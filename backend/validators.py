# medical-data-analysis/backend/validators.py
"""医疗数据校验、变量类型推断、医学合理范围异常检测。

提供三件事：
1. validate_medical_dataset：中英文医学字段词库匹配，判断是否为医疗数据；
2. infer_variable_types：推断变量类型（数值 / 分类 / 数字编码分类 / 文本 / 日期），
   自动排除 ID 列与常量列；
3. detect_medical_outliers：结合医学参考区间与 IQR 识别异常值。
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

from config import HIGH_MISSING_THRESHOLD, REFERENCE_ALIASES, REFERENCE_RANGES

# ------------------------------------------------------------------
# 全局医学词库（用于判断标准名是否为医学指标）
# ------------------------------------------------------------------
_GLOBAL_LEXICON_CACHE = None

def _load_global_lexicon():
    """加载全局医学词库，返回 {standard_name: entry} 映射。"""
    global _GLOBAL_LEXICON_CACHE
    if _GLOBAL_LEXICON_CACHE is not None:
        return _GLOBAL_LEXICON_CACHE
    
    try:
        import json
        import os
        lexicon_path = os.path.join(os.path.dirname(__file__), 'automl', 'lexicon', 'global_lexicon.json')
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 构建 standard_name -> entry 映射
        lexicon = {}
        for entry in data.get('entries', []):
            std_name = entry.get('standard_name', '').strip().lower()
            if std_name:
                lexicon[std_name] = entry
        
        _GLOBAL_LEXICON_CACHE = lexicon
        return lexicon
    except Exception:
        # 如果加载失败，返回空字典
        _GLOBAL_LEXICON_CACHE = {}
        return {}

def _is_medical_feature(standard_name: str | None) -> bool:
    """检查标准名是否为已知的医学特征（非 ID 列）。
    
    如果标准名在全局词库中且 role_hint 不是 'id'，则认为是医学特征。
    """
    if not standard_name:
        return False
    
    lexicon = _load_global_lexicon()
    std_key = standard_name.strip().lower()
    
    entry = lexicon.get(std_key)
    if not entry:
        return False
    
    role = entry.get('role_hint', '')
    # 如果 role_hint 不是 'id'，则认为是医学特征
    return role != 'id'

# ------------------------------------------------------------------
# 中英文医学字段词库（用于医疗数据校验与结局变量识别）
# ------------------------------------------------------------------
MEDICAL_LEXICON = {
    "人口学": [
        "age", "gender", "sex", "male", "female", "ethnic", "race",
        "年龄", "性别", "民族", "婚育", "职业",
    ],
    "体格与生命体征": [
        "height", "weight", "bmi", "body_mass", "sbp", "dbp", "bp",
        "blood_pressure", "heart_rate", "pulse", "pulse_rate", "temperature",
        "respiratory", "rr", "spo2", "oxygen", "waist", "hip", "waist_hip",
        "身高", "体重", "体质指数", "收缩压", "舒张压", "血压", "心率", "脉搏",
        "体温", "呼吸", "血氧", "腰围", "臀围",
    ],
    "实验室检验": [
        "glucose", "hba1c", "cholesterol", "chol", "triglyceride", "tg", "tc", "hdl", "ldl",
        "hemoglobin", "hgb", "wbc", "rbc", "platelet", "plt", "creatinine", "crea", "bun",
        "urea", "uric_acid", "alt", "ast", "alp", "ggt", "crp", "albumin", "alb",
        "bilirubin", "tbil", "dbil", "inr", "aptt", "pt", "sodium", "potassium",
        "calcium", "chloride", "magnesium", "troponin", "ck_mb", "bnp", "d_dimer",
        "ferritin", "fpg", "ogtt", "c_peptide", "tsh", "ft3", "ft4", "t3", "t4",
        "psa", "cea", "afp", "esr", "hct", "mcv", "mch", "rdw", "pdw", "mpv",
        "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil",
        "lactate", "amylase", "lipase", "vitamin", "folate", "protein", "globulin",
        "sgot", "sgpt", "alkphos", "il6", "pct",
        "空腹血糖", "血糖", "糖化血红蛋白", "总胆固醇", "胆固醇", "甘油三酯",
        "高密度脂蛋白", "低密度脂蛋白", "血红蛋白", "白细胞", "红细胞", "血小板",
        "肌酐", "尿素氮", "尿酸", "转氨酶", "白蛋白", "胆红素", "凝血",
        "电解质", "钠", "钾", "钙", "肌钙蛋白", "血气", "乳酸", "淀粉酶",
        "检验", "化验", "指标",
    ],
    "诊断与疾病": [
        "diagnosis", "disease", "tumor", "cancer", "carcinoma", "adenoma",
        "diabetes", "hypertension", "stroke", "pneumonia", "hepatitis",
        "anemia", "copd", "sepsis", "infection", "ckd", "cad",
        "myocardial", "infarction", "arrhythmia", "cardiac", "pulmonary",
        "renal", "hepatic", "asthma", "allergy", "immunotherapy", "malaria",
        "tuberculosis", "hiv", "aids", "covid", "cirrhosis", "fibrosis",
        "depression", "anxiety", "autism", "alzheimer", "parkinson",
        "epilepsy", "seizure", "lesion", "mortality",
        "heart", "liver", "kidney", "lung", "breast", "thyroid", "prostate",
        "ovarian", "cervical", "brain", "bone", "skin", "blood", "plasma",
        "serum", "obesity", "arthritis", "osteoporosis", "transplant", "dialysis",
        "诊断", "疾病", "肿瘤", "癌", "糖尿病", "高血压", "脑卒中", "冠心病",
        "肺炎", "肝炎", "肾病", "贫血", "感染", "并发症", "合并症", "病史",
        "心肌", "梗死", "心律失常", "肝硬化", "哮喘", "过敏", "抑郁", "焦虑",
    ],
    "生理信号与可穿戴": [
        "eeg", "ecg", "ekg", "emg", "ppg", "accelerometer", "gyroscope",
        "gait", "posture", "wearable", "physiological", "heart_sound",
        "脑电", "心电", "肌电", "脉搏波", "加速度计", "陀螺仪", "步态", "体征",
    ],
    "分期分级与病理": [
        "stage", "grade", "t_stage", "n_stage", "m_stage", "pathology",
        "histology", "differentiation", "ki67", "er", "pr", "her2",
        "分期", "分级", "分型", "病理", "分化", "浸润", "转移", "淋巴结",
    ],
    "治疗与用药": [
        "treatment", "therapy", "surgery", "operation", "drug", "dose",
        "medication", "chemotherapy", "radiotherapy", "targeted", "immuno",
        "治疗", "手术", "用药", "药物", "剂量", "化疗", "放疗", "靶向", "免疫",
        "给药", "方案", "疗程",
    ],
    # 专科检查与常用英文缩写：公开数据集（心脏病、超声、营养流行病学）里高频出现
    "专科检查与常用缩写": [
        "cp", "trestbps", "restecg", "thalach", "exang", "oldpeak", "slope", "thal",
        "fbs", "chol", "lvdd", "lvef", "epss",
        "pericardial", "fractional", "wall_score",
        "wall_index", "retinol", "retplasma", "betaplasma", "smoking", "smoker",
        "echo", "spirometry", "dlco", "holter", "cardiotocography", "ctg",
        "胸痛", "心悸", "气促", "晕厥", "水肿", "超声", "心超",
    ],
    # 心理/精神量表条目：产后抑郁等问卷数据的列名通常就是症状短语
    "心理与行为": [
        "depression", "anxiety", "anxious", "irritable", "tearful", "guilt",
        "suicide", "appetite", "sleeping", "sleep", "bonding", "stress", "mental",
        "抑郁", "焦虑", "易怒", "哭泣", "内疚", "自杀", "食欲", "睡眠", "压力",
    ],
    "结局与随访": [
        "outcome", "mortality", "death", "survival", "os", "dfs", "pfs",
        "recurrence", "relapse", "metastasis", "remission", "response",
        "readmission", "follow_up", "followup", "event", "status", "censor",
        "duration", "at_risk", "events", "time_to_event",
        "结局", "预后", "死亡", "生存", "复发", "缓解", "疗效", "随访",
        "住院", "出院", "入院", "再入院", "事件",
    ],
    "医疗场景": [
        "patient", "case", "cohort", "hospital", "icu", "ward", "clinic",
        "admission", "discharge", "record", "sample", "specimen", "lab",
        "examination", "radiology", "symptom", "sign", "score", "bmi",
        "medical", "clinical", "health", "healthcare", "biomed", "biomedical",
        "患者", "病人", "病例", "队列", "医院", "病案", "病历", "样本", "标本",
        "科室", "门诊", "急诊", "体检", "症状", "体征", "评分",
    ],
}

# 结局变量候选词（用于 P1 结局变量智能识别，此处一并维护以免词库分裂）
# 真实医疗/公开数据集里最常见的结局列名（class/category/result/target/label/grade/stage）
# 必须先命中词库，否则任何二值列都会因为“二值加分”被误荐为结局（性别、种族等）。
OUTCOME_KEYWORDS = [
    "outcome", "mortality", "death", "died", "survival", "status", "event",
    "recurrence", "relapse", "metastasis", "response", "remission", "censor",
    "readmission", "complication", "infection", "diagnosis", "disease",
    "class", "category", "target", "label", "result", "grade", "stage", "severity",
    "pxc50", "ic50", "ec50", "activity", "potency", "affinity", "viability",
    "sensitivity", "resistance",
    "结局", "预后", "死亡", "生存", "复发", "转移", "疗效", "缓解", "事件",
    "并发症", "感染", "诊断", "是否", "分级", "分期", "结果", "分类", "类别",
]

# 明确的非结局（应排除）字段
NON_OUTCOME_KEYWORDS = ["id", "编号", "姓名", "name", "date", "日期", "time", "phone", "phone_no"]
# 人口学/标识列：不应被自动推荐为研究结局（用户仍可手动选择）
NON_OUTCOME_DEMOGRAPHIC = ["sex", "gender", "race", "ethnicity", "marital", "insurance",
                           "age", "性别", "种族", "民族", "婚姻", "保险", "年龄"]
# 标注/评分/受试者/实验条件类字段：是协变量或标注元数据，不能当结局
NON_OUTCOME_ANNOTATION = ["scorer", "rater", "annotator", "observer", "automatic", "manual",
                          "subject", "order", "pair", "cell_type", "stimulus", "inhibitor",
                          "评分者", "标注", "受试者", "受试", "受试者编号"]
# 治疗/分组类协变量：可能是结局（如“治疗反应”），因此只降权不硬排除
SOFT_COVARIATE_TERMS = ["treat", "treatment", "therapy", "arm", "group", "allocation",
                        "visit", "center", "site", "dose", "dosage", "total",
                        "治疗", "分组", "访视", "剂量", "合计", "总计"]

# 常见的“目标列”精确名：命中时额外加权，避免被同名业务字段抢走
_TARGET_LIKE_NAMES = {"class", "category", "target", "label", "result", "outcome",
                      "diagnosis", "status"}
# 目标列常见后缀：binaryClass、class、label、target 等（OpenML/UCI 常见命名）
_TARGET_LIKE_SUFFIXES = ("class", "label", "target", "outcome", "status", "diagnosis",
                         "category", "result")
# 纯数字/单符号列名：多为无表头文件被误当表头（首行数据变成列名），不可作为结局
_NUMERIC_COLNAME_RE = re.compile(r"^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?$")
# 特征族：同一前缀 + 递增数字后缀（FCFP6_1024_0..1023、g_123、col_1..col_N）
_FAMILY_SUFFIX_RE = re.compile(r"(?:[_\-\s]*\d+)+$")

# 组学特征列命名（GENE1835X、g_202387_at、probe_xxx、miRNA/SND 等）：
# 这些数据集列名本身不含临床词，但整体就是基因表达/变异矩阵，属于生物医学数据
_OMICS_COLNAME_RE = re.compile(
    r"(?:^|_)(?:gene\d+x|g_\d+(?:_at)?|probe|snp|mirna|expression|methylation)", re.I)

# 疑似 ID 的字段名特征：ascii 的 id/no/code 须以非字母为界（避免误伤 covid、node 等），
# 中文编号类词按子串匹配
_ID_PATTERN = re.compile(
    r"(?:^|[^a-z])(?:id|no|code)(?:$|[^a-z])|编号|序号|流水号|病案号|住院号|病历号", re.I)

# 强临床指标：命中任一即足以判定为医疗数据（用于 WDBC 这类字段名多为测量型的数据集）
STRONG_CLINICAL_TERMS = [
    "diagnosis", "diagnose", "malignant", "benign", "cancer", "carcinoma", "tumor",
    "tumour", "neoplasm", "nuclei", "nucleus", "biopsy", "cytology", "histology",
    "mortality", "death", "survival", "outcome", "prognosis", "recurrence",
    "metastasis", "relapse", "remission", "complication", "sepsis",
    "诊断", "结局", "预后", "死亡", "生存", "复发", "转移", "恶性", "良性",
    "肿瘤", "癌", "病理", "活检", "并发症",
]

# 文件名强提示词：这些词本身足以说明数据属于医疗/临床场景。
# 公开医疗数据集经常把列名匿名化为 V1..Vn / attribute1..N（如 heart-switzerland、
# lung-cancer），此时字段命中必然为 0；若仍要求“至少命中 1 个字段”，真实医疗数据
# 会被大量误判。仅在文件名命中本表时才允许“0 字段命中”放行，普通词库仍维持原规则。
STRONG_NAME_TERMS = [
    # 器官 / 专科 / 疾病
    "heart", "heartdisease", "cardiac", "coronary", "cardiovascular", "arrhythmia",
    "ecg", "ekg", "liver", "liverdisease", "hepatic", "hepat", "hepatitis", "cirrhosis",
    "kidney", "kidneydisease", "renal", "nephro", "ckd", "dialysis",
    "lung", "lungcancer", "lungdisease", "pulmonary", "copd", "pneumonia",
    "respiratory", "asthma", "breast", "breastcancer", "mammogra", "mammographic",
    "mammary", "cervical", "ovarian", "uterine", "prostate", "endometri",
    "cancer", "tumor", "tumour", "carcinoma", "leukemia", "lymphoma", "melanoma",
    "sarcoma", "glioma", "metastasis", "neoplasm",
    "diabetes", "diabetic", "glucose", "insulin", "hba1c",
    "thyroid", "thyroiddisease", "hypothyroid", "hyperthyroid",
    "parkinson", "parkinsons", "alzheimer", "epilepsy", "seizure", "stroke",
    "brain", "brainsize", "neuro", "autism", "depression", "mental", "psychiat",
    "hiv", "aids", "covid", "covid19", "sars", "malaria", "tuberculosis",
    "sepsis", "infection", "viruses",
    "anemia", "anaemia", "hemoglobin", "platelet", "blood", "transfusion", "plasma",
    "dermatology", "skin", "audiology", "hearing", "sleep", "apnea",
    "obesity", "bmi", "arthritis", "osteoporosis", "vertebra", "spine",
    "fracture", "hip", "knee", "surgery", "surgical", "postoperative",
    "cardiotocography",
    "transplant", "transplantation", "icu", "intensive", "mortality", "survival",
    # 场景 / 人群
    "patient", "clinical", "hospital", "medical", "health", "healthcare",
    "diagnosis", "disease", "pregnancy", "pregnan", "fetal", "neonatal",
    "births", "fertility", "maternal",
    # 治疗 / 检查 / 生物医学
    "drug", "vaccine", "immunotherapy", "allergy", "mri", "ultrasound",
    "mammography", "radiology", "spect", "echocardi", "biomed", "biomedical",
    "protein", "phosphoprotein", "miceprotein", "genom", "genomics", "gene",
    "microarray",
]

# 仅这些足够具体的强提示词允许做“去分隔符子串匹配”，用于 ParkinsonSpeech、
# SKdailyCOVID19 这类粘连命名；patient/hospital 等通用场景词仍只走词边界，
# 避免把 hospitality / patiently 这类普通文件名误判成医疗数据。
COMPOUND_NAME_TERMS = [
    "covid19", "parkinson", "phosphoprotein", "miceprotein", "brainsize",
    "lungcancer", "breastcancer", "heartdisease", "kidneydisease",
    "liverdisease", "thyroiddisease", "cardiotocography", "echocardiogram",
    "mammography", "dermatology", "audiology", "tuberculosis", "alzheimer",
    "osteoporosis", "immunotherapy", "transplantation", "biomedical",
    "microarray", "genomics",
]

# 通用场景词：只说明“可能和医疗场景有关”，但不能单独放行匿名化数据。
# 例如 medical_devices_sales、health_insurance_claims、patient_satisfaction
# 文件名含 medical/health/patient，但列全是销售额/评分时不应判为医疗数据。
# 这些词仍参与普通文件名提示（需至少 1 个字段命中），但不享受“0 字段放行”。
GENERIC_NAME_TERMS = {"patient", "hospital", "medical", "health", "healthcare"}


def strong_clinical_fields(df: pd.DataFrame, rename_map: dict | None = None) -> list:
    """返回命中强临床指标词的字段名列表。
    
    同时检查原始列名和标准化列名（如果提供了 rename_map），以支持非标准命名的医疗数据。
    """
    rename_map = rename_map or {}
    hits = []
    for col in df.columns:
        norm = normalize_field(col)
        std_name = rename_map.get(str(col))
        std_norm = normalize_field(std_name) if std_name else None
        
        if any(_term_hits(norm, [t]) for t in STRONG_CLINICAL_TERMS):
            hits.append(str(col))
        elif std_norm and any(_term_hits(std_norm, [t]) for t in STRONG_CLINICAL_TERMS):
            hits.append(str(col))
    return hits


def normalize_field(name) -> str:
    """字段名标准化：去空白、小写。"""
    return str(name).strip().lower()


def _term_hits(field_norm: str, terms):
    """返回字段命中该组词库的词列表（英文按词边界，中文按子串）。"""
    hits = []
    for t in terms:
        tl = t.lower()
        if re.fullmatch(r"[a-z0-9_\-]+", tl):
            if re.search(r"(^|[^a-z])" + re.escape(tl) + r"([^a-z]|$)", field_norm):
                hits.append(t)
        else:
            if tl in field_norm:
                hits.append(t)
    return hits


def _strong_name_hits(name_norm: str) -> list:
    """文件名强提示词命中：词边界命中 + 去分隔符后的长词子串命中。

    公开数据集文件名常把单词粘连（ParkinsonSpeech、SKdailyCOVID19、
    PhosphoproteinChallenge），仅靠词边界会漏掉。压缩子串匹配只对
    COMPOUND_NAME_TERMS 中足够具体的词启用，避免 "patient"/"hospital"
    这类通用词误伤 hospitality / patiently 等普通文件名。
    """
    hits = list(_term_hits(name_norm, STRONG_NAME_TERMS))
    compact = re.sub(r"[^a-z0-9]", "", name_norm)
    for term in COMPOUND_NAME_TERMS:
        if term not in hits and term in compact:
            hits.append(term)
    return hits


def match_medical_fields(df: pd.DataFrame, rename_map: dict | None = None) -> dict:
    """逐字段匹配医学词库，返回命中情况与整体匹配度。
    
    同时检查原始列名和标准化列名（如果提供了 rename_map），以支持非标准命名的医疗数据。
    """
    rename_map = rename_map or {}
    fields = []
    matched = 0
    for col in df.columns:
        norm = normalize_field(col)
        std_name = rename_map.get(str(col))
        std_norm = normalize_field(std_name) if std_name else None
        
        hits_by_category = {}
        for category, terms in MEDICAL_LEXICON.items():
            # 检查原始列名
            hits = _term_hits(norm, terms)
            # 检查标准列名（如果有）
            if std_norm:
                std_hits = _term_hits(std_norm, terms)
                hits.extend([h for h in std_hits if h not in hits])
            if hits:
                hits_by_category[category] = hits
        if hits_by_category:
            matched += 1
        fields.append({
            "field": str(col),
            "matched": bool(hits_by_category),
            "categories": list(hits_by_category.keys()),
            "terms": [t for hits in hits_by_category.values() for t in hits][:6],
            "standard_name": std_name,
        })
    total = max(1, len(df.columns))
    score = round(matched / total, 4)
    matched_terms = list(dict.fromkeys(
        t for f in fields if f["matched"] for t in f["terms"]))
    return {
        "total_fields": len(df.columns),
        "matched_fields": matched,
        "match_score": score,
        "fields": fields,
        "matched_field_names": [f["field"] for f in fields if f["matched"]],
        "matched_terms": matched_terms[:12],
    }


def validate_medical_dataset(df: pd.DataFrame, rename_map: dict | None = None, 
                             min_score: float = 0.30, min_fields: int = 2,
                             dataset_name: str | None = None) -> dict:
    """校验数据集是否为医疗数据。

    判定规则（四条满足其一即为医疗数据）：
      1. 常规规则：字段匹配度 ≥ min_score 且命中字段数 ≥ min_fields；
      2. 强临床指标规则：存在诊断/结局/良恶性等强临床字段（如 WDBC 的 diagnosis，
         其形态学测量字段名本身不含医学词，但诊断字段足以判定为医疗数据）；
      3. 文件名提示规则：文件名含医学词（如 heart/cancer/patient），且满足以下之一：
         - 至少命中 1 个医学字段（普通词库或通用场景词均可）；
         - 文件名命中具体强提示词（如 heart/lung-cancer/kidney），此时允许列名全部
           匿名化、字段命中为 0（公开医疗数据集常见 V1..Vn / attribute1..N 命名）。
         注意 patient/hospital/medical/health/healthcare 属于通用场景词，
         必须有至少 1 个字段命中，不能单凭文件名放行。
      4. 组学矩阵规则：列名呈现 GENE*X / g_*_at / probe 等组学特征。

    匹配度不足时不阻断流程，而是返回 terminate / continue 两个选项由用户决定，
    手动放行会在报告中注明。
    
    rename_map: 列名标准化映射 {原名: 标准名}，用于同时检查原始列名和标准列名。
    dataset_name: 上传文件名，用于补充“文件名含医学词”的证据。
    """
    rename_map = rename_map or {}
    info = match_medical_fields(df, rename_map)
    normal_rule = (info["match_score"] >= min_score) and (info["matched_fields"] >= min_fields)
    strong_hits = strong_clinical_fields(df, rename_map)

    name_hits = []
    strong_name_hits = []
    if dataset_name:
        base = re.split(r"[\\/]", str(dataset_name))[-1]
        base = re.sub(r"\.\w+$", "", base)
        lexicon_terms = [t for terms in MEDICAL_LEXICON.values() for t in terms]
        name_norm = normalize_field(base)
        strong_name_hits = _strong_name_hits(name_norm)
        name_hits = list(dict.fromkeys(
            _term_hits(name_norm, lexicon_terms) + strong_name_hits))
    # 普通文件名提示仍需至少 1 个字段命中；
    # 只有“具体强提示词”（器官/疾病/检查）允许 0 字段命中（列名匿名化场景）；
    # patient/hospital/medical/health 等通用场景词不享受该豁免。
    specific_name_hits = [t for t in strong_name_hits
                          if t not in GENERIC_NAME_TERMS]
    name_rule = bool(name_hits) and (info["matched_fields"] >= 1
                                     or bool(specific_name_hits))
    # 组学矩阵规则：列名多为 GENE*X / g_*_at / probe 等探针名，整体属于生物医学数据
    omics_fields = [str(c) for c in df.columns if _OMICS_COLNAME_RE.search(normalize_field(c))]
    omics_rule = (info["total_fields"] >= 20
                  and len(omics_fields) >= max(10, int(0.2 * info["total_fields"])))
    is_medical = bool(normal_rule or strong_hits or name_rule or omics_rule)

    if is_medical and not normal_rule and not strong_hits and omics_rule:
        message = (f"共 {info['total_fields']} 个字段，常规词库仅命中 {info['matched_fields']} 个"
                   f"（匹配度 {info['match_score']:.0%}），但检测到 {len(omics_fields)} 个组学特征列"
                   f"（如 {omics_fields[0]}、{omics_fields[1] if len(omics_fields) > 1 else '—'}），"
                   "判定为医疗/生物医学数据；医学参考区间对基因表达量不适用。")
    elif is_medical and not normal_rule and not strong_hits:
        message = (f"共 {info['total_fields']} 个字段，医学字段命中 {info['matched_fields']} 个"
                   f"（匹配度 {info['match_score']:.0%}），"
                   f"但文件名包含医学提示词 {'、'.join(name_hits[:5])}，已判定为医疗数据；"
                   "如不符合预期可终止分析。")
    elif is_medical and not normal_rule:
        message = (f"共 {info['total_fields']} 个字段，常规词库仅命中 {info['matched_fields']} 个"
                   f"（匹配度 {info['match_score']:.0%}），但检测到强临床指标字段 "
                   f"{'、'.join(strong_hits)}，已判定为医疗数据；"
                   f"因字段多为测量型变量，医学参考区间可能不适用。")
    elif is_medical:
        message = (f"共 {info['total_fields']} 个字段，命中医学词库 {info['matched_fields']} 个"
                   f"（匹配度 {info['match_score']:.0%}），已识别为医疗数据。"
                   + (f"命中术语：{'、'.join(info['matched_terms'][:6])}。" if info["matched_terms"] else ""))
    else:
        message = (f"共 {info['total_fields']} 个字段，仅命中医学词库 {info['matched_fields']} 个"
                   f"（匹配度 {info['match_score']:.0%}，需 ≥{min_score:.0%} 且 ≥{min_fields} 个字段，"
                   f"或含诊断/结局等强临床字段）。该数据集可能不是医疗数据，"
                   f"统计模型与医学参考区间可能不适用。")
    return {
        "is_medical": is_medical,
        "match_score": info["match_score"],
        "matched_fields": info["matched_fields"],
        "total_fields": info["total_fields"],
        "matched_field_names": info["matched_field_names"],
        "matched_terms": info["matched_terms"],
        "name_hint": name_hits[:5],
        "strong_clinical_fields": strong_hits,
        "omics_fields": len(omics_fields),
        "rule": ("strong_clinical" if (is_medical and not normal_rule and strong_hits) else
                 ("filename" if (is_medical and not normal_rule and name_rule) else
                  ("omics" if (is_medical and not normal_rule and omics_rule) else
                   ("normal" if is_medical else "none")))),
        "field_details": info["fields"],
        "message": message,
        "options": ["terminate", "continue"],
        "threshold": {"min_score": min_score, "min_fields": min_fields},
    }


# ------------------------------------------------------------------
# 变量类型推断
# ------------------------------------------------------------------
# 数字编码分类变量的判定阈值
MAX_CODE_LEVELS = 10          # 取值水平数不超过该值才可能是编码分类
MAX_CODE_UNIQUE_RATIO = 0.05  # 唯一值占比低于该值才可能是编码分类
ID_UNIQUE_RATIO = 0.90        # 唯一值占比高于该值判定为 ID 列


def _is_numeric_code(s: pd.Series) -> bool:
    """判断整数型字段是否为数字编码的分类变量（如性别 0/1、疾病分级 1/2/3）。"""
    vals = s.dropna()
    if vals.empty:
        return False
    n_unique = vals.nunique()
    if n_unique > MAX_CODE_LEVELS or n_unique < 2:
        return False
    if len(vals) > 0 and n_unique / len(vals) > MAX_CODE_UNIQUE_RATIO:
        return False
    # 必须全部为整数取值（舍入后与原值一致）
    arr = vals.to_numpy(dtype=float)
    return bool(np.all(np.isclose(arr, np.round(arr))))


def _is_id_like(s: pd.Series, name: str, standard_name: str | None = None) -> bool:
    """疑似 ID 列：字段名（原名或标准名）带编号特征，或（高唯一 + 全整数取值）。

    非整数取值的浮点列（连续测量值）绝不判为 ID——WDBC 类数据集中
    perimeter_mean 等特征取值近乎唯一，但它们是正儿八经的连续变量。
    
    如果 standard_name 指向医学指标（有参考区间的），即使唯一值占比高也不判为 ID。
    如果 standard_name 在全局医学词库中且 role_hint 不是 'id'，也不判为 ID。
    """
    # 先检查原名，再检查标准名（如果提供了）
    if _ID_PATTERN.search(str(name)):
        return True
    if standard_name and _ID_PATTERN.search(str(standard_name)):
        return True
    
    # 如果标准名是医学指标（有参考区间或在词库中），不判定为 ID 列
    if standard_name:
        std_key = str(standard_name).strip().lower()
        if std_key in REFERENCE_ALIASES or std_key in REFERENCE_RANGES:
            return False
        # 检查全局医学词库
        if _is_medical_feature(standard_name):
            return False
    
    vals = s.dropna()
    if vals.empty or not pd.api.types.is_numeric_dtype(vals):
        return False
    arr = vals.to_numpy(dtype=float)
    if not np.all(np.isclose(arr, np.round(arr))):
        return False
    # 仅“连续编号”（取值恰为 1..n / 0..n-1 这样无跳号的整数序列）才算 ID；
    # 计数、金额、剂量等整数测量值即使每行都不同，也不能因为“唯一值多”被当成 ID 排除
    # （否则像 AIDS、Total 这类真实计数型结局会整列消失）。
    uniq = np.unique(arr)
    if uniq.size < 2:
        return False
    return bool(np.all(uniq == np.arange(uniq.min(), uniq.min() + uniq.size)))


def _to_numeric_series(s: pd.Series):
    """尝试把对象列转为数值列，成功率 >=90% 时返回转换结果，否则返回 None。"""
    converted = pd.to_numeric(s, errors="coerce")
    non_missing = s.notna().sum()
    if non_missing == 0:
        return None
    ok = converted.notna().sum()
    if ok / max(1, non_missing) >= 0.9 and ok > 0:
        return converted
    return None


def infer_variable_types(df: pd.DataFrame, overrides: dict | None = None,
                          rename_map: dict | None = None) -> list:
    """推断每个字段的类型，输出供前端确认的清单。

    返回字段：name / inferred_type / n_unique / n_missing / missing_rate /
    high_missing / sample_values / excluded / exclude_reason / reference_range /
    confirmed_type。

    high_missing 只做警示（缺失比例 > HIGH_MISSING_THRESHOLD），不自动排除该列，
    是否纳入分析由用户在确认页决定。

    rename_map: 列名标准化映射 {原名: 标准名}，用于参考区间查找。
    """
    overrides = overrides or {}
    rename_map = rename_map or {}
    n_rows = len(df)
    result = []
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        missing = int(s.isna().sum())
        missing_rate = round(missing / n_rows, 4) if n_rows else 0.0
        non_null = s.dropna()
        samples = [str(v) for v in non_null.head(3).tolist()]

        excluded, reason, inferred = False, None, None
        std_name = rename_map.get(str(col))

        if n_rows and missing_rate >= 1.0:
            excluded, reason, inferred = True, "全列缺失", "constant"
        elif n_unique <= 1:
            excluded, reason, inferred = True, "常量列（唯一值 ≤ 1）", "constant"
        elif n_rows and (n_unique / n_rows) > ID_UNIQUE_RATIO and _is_id_like(s, col, std_name):
            excluded, reason, inferred = True, "疑似 ID 列（字段名或取值特征判定）", "id"
        elif pd.api.types.is_datetime64_any_dtype(s):
            inferred = "datetime"
            excluded, reason = True, "日期/时间型字段，暂不参与统计建模"
        elif pd.api.types.is_bool_dtype(s):
            inferred = "categorical"
        elif pd.api.types.is_numeric_dtype(s):
            inferred = "categorical" if _is_numeric_code(s) else "numeric"
        else:
            conv = _to_numeric_series(s)
            if conv is not None:
                inferred = "categorical" if _is_numeric_code(conv) else "numeric"
            elif n_unique <= max(20, int(0.05 * max(1, n_rows))):
                inferred = "categorical"
            else:
                inferred = "text"
                excluded, reason = True, "自由文本字段，不参与统计建模"

        confirmed = overrides.get(str(col)) or inferred
        result.append({
            "name": str(col),
            "inferred_type": inferred,
            "confirmed_type": confirmed,
            "n_unique": n_unique,
            "n_missing": missing,
            "missing_rate": missing_rate,
            # 只警示不排除：缺失比例超过阈值的列在确认页标记，纳入与否由用户决定
            "high_missing": bool(n_rows and missing_rate > HIGH_MISSING_THRESHOLD),
            "sample_values": samples,
            "excluded": excluded,
            "exclude_reason": reason,
            "reference_range": reference_of(col, std_name),
            "standard_name": std_name,
        })
    return result


def reference_of(column_name: str, standard_name: str | None = None):
    """返回字段对应的医学参考区间配置（无则 None）。
    
    查找策略（按优先级）：
    1. 如果有 standard_name，先用它在 REFERENCE_ALIASES 中查找对应的英文 key
    2. 用原始列名在 REFERENCE_ALIASES 中查找对应的英文 key
    3. 用原始列名或标准名直接在 REFERENCE_RANGES 中查找（兼容直接用英文 key 的情况）
    """
    # 优先用标准名查找别名映射
    if standard_name:
        std_key = str(standard_name).strip().lower()
        # 先用标准名在别名表中查找英文 key
        if std_key in REFERENCE_ALIASES:
            ref_key = REFERENCE_ALIASES[std_key]
            if ref_key in REFERENCE_RANGES:
                return REFERENCE_RANGES[ref_key]
        # 如果标准名本身就是 REFERENCE_RANGES 的 key（英文标准名）
        if std_key in REFERENCE_RANGES:
            return REFERENCE_RANGES[std_key]
    
    # 回退：用原始列名查找
    col_key = str(column_name).strip().lower()
    # 在别名表中查找
    if col_key in REFERENCE_ALIASES:
        ref_key = REFERENCE_ALIASES[col_key]
        if ref_key in REFERENCE_RANGES:
            return REFERENCE_RANGES[ref_key]
    # 直接在 REFERENCE_RANGES 中查找
    return REFERENCE_RANGES.get(col_key)


def detect_medical_outliers(df: pd.DataFrame, variables: list, rename_map: dict | None = None) -> list:
    """对每个数值变量检测异常值：医学参考区间 + IQR 统计离群。"""
    rename_map = rename_map or {}
    report = []
    for col in variables:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        q1, q3 = np.percentile(valid, [25, 75])
        iqr = q3 - q1
        lo_stat, hi_stat = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((valid < lo_stat) | (valid > hi_stat)).sum())

        std_name = rename_map.get(str(col))
        ref = reference_of(col, std_name)
        n_below = n_above = 0
        ref_used = None
        if ref:
            n_below = int((valid < ref["low"]).sum())
            n_above = int((valid > ref["high"]).sum())
            ref_used = {"low": ref["low"], "high": ref["high"], "unit": ref["unit"],
                        "label": ref["label"], "source": ref["source"]}

        report.append({
            "field": str(col),
            "n_valid": int(valid.size),
            "min": float(valid.min()), "max": float(valid.max()),
            "iqr_bounds": [float(lo_stat), float(hi_stat)],
            "n_iqr_outlier": n_iqr,
            "reference_range": ref_used,
            "n_below_reference": n_below,
            "n_above_reference": n_above,
        })
    return report


def _family_key(name: str) -> str:
    """特征族前缀：去掉末尾连续的数字后缀（_0.._1023、123 等）。"""
    return _FAMILY_SUFFIX_RE.sub("", normalize_field(name)).strip("_- ")


def _feature_family_members(names) -> set:
    """识别「同一前缀 + 递增数字」的特征族（分子指纹、匿名列 col_1..N 等）。

    同前缀列数需达到 max(8, 5% 列数) 才认定为特征族，避免少量同名指标被误判。
    """
    if len(names) < 8:
        return set()
    keys = {str(n): _family_key(n) for n in names}
    counts = Counter(k for k in keys.values() if k)
    threshold = max(8, int(len(names) * 0.05))
    return {n for n, k in keys.items() if k and counts.get(k, 0) >= threshold}


def suggest_outcome_candidates(df: pd.DataFrame, schema: list) -> list:
    """扫描字段名与词库，推断最可能的结局变量候选（P1 使用）。

    排序根因修复：旧规则里「任意二值列」都能拿到 0.6 分，而真正的连续目标列
    （如 MEDIAN_PXC50、Count）没有任何词库命中时得 0 分，直接被丢弃；宽表里的
    特征族（FCFP6_1024_x、col_1..N）又会靠二值加分抢到第一位。新规则：
    1) 先硬排除标注/评分/受试者/实验条件列与人口学列（含 Age/AGE）；
    2) 识别特征族；存在族外列时，族内列不再作为候选（保护真实目标列）；
    3) 排序优先级：结局词库 > 结构证据（族外列/族尾列）> 目标名后缀 > 二值加分；
    4) 无任何强/弱候选时，用族外数值列兜底（自表尾向前），避免出现“无结局”。
    """
    schema_list = list(schema)
    names = [str(it["name"]) for it in schema_list if not it.get("excluded")]
    family = _feature_family_members(names)
    # 只有存在「族外列」时才屏蔽族内列；若整表都是 col_1..col_N（匿名表），族内列仍是唯一可选项
    non_family = [n for n in names if n not in family]
    family_blocked = family if non_family else set()

    eligible = []
    for pos, item in enumerate(schema_list):
        if item.get("excluded"):
            continue
        name = str(item["name"])
        # 无表头文件的首行数据会被当成列名（如 "1.0000000e+000"、"+"、"#"）：
        # 这类列名不是有意义的目标列，直接跳过，避免把噪声列当成研究结局。
        if _NUMERIC_COLNAME_RE.match(name.strip()) or name.strip() in {"+", "-", "#", "?", ""}:
            continue
        if name in family_blocked:
            continue

        std_name = item.get("standard_name")
        norms = [normalize_field(name)] + ([normalize_field(std_name)] if std_name else [])
        # 排除非结局字段（英文缩写按词边界匹配，避免 "os" 误伤 "glucose"、"dose" 等）
        if any(_term_hits(n, NON_OUTCOME_KEYWORDS) for n in norms):
            continue
        # 人口学/标识列（含 Age/AGE）不自动推荐为结局：性别、年龄等二值列极易抢占排名
        if any(_term_hits(n, NON_OUTCOME_DEMOGRAPHIC) for n in norms):
            continue
        # 评分者/标注者/受试者/实验条件列：属于协变量或标注元数据，不能作为预测目标
        if any(_term_hits(n, NON_OUTCOME_ANNOTATION) for n in norms):
            continue

        hits = []
        for n in norms:
            hits.extend([k for k in _term_hits(n, OUTCOME_KEYWORDS) if k not in hits])
        soft = []
        for n in norms:
            soft.extend([k for k in _term_hits(n, SOFT_COVARIATE_TERMS) if k not in soft])

        eligible.append({
            "pos": pos,
            "field": name,
            "norm": normalize_field(name),
            "item": item,
            "hits": hits,
            "soft": soft,
            "n_unique": int(item.get("n_unique") or 0),
        })

    scored = []
    for e in eligible:
        item = e["item"]
        n_unique = e["n_unique"]
        score = 0.0
        reasons = []
        if e["hits"]:
            score += 2.0
            reasons.append("字段名命中结局词库：" + "、".join(e["hits"]))
        if e["norm"] in _TARGET_LIKE_NAMES or e["norm"].endswith(_TARGET_LIKE_SUFFIXES):
            score += 0.5
            reasons.append("字段名符合常见目标列命名（class/label/target/status 等）")
        # 结构证据：宽表中唯一的族外列，通常就是被上千个特征列包围的真实目标
        if family_blocked and non_family and e["field"] not in family_blocked:
            score += 0.6
            reasons.append("位于宽表特征族之外（其余列为同族特征）")
            if e["field"] == non_family[-1]:
                score += 0.4
                reasons.append("特征族之后的末列（公开数据集常见的目标列位置）")
        # 二值/少分类只做轻微加分，不允许“任意二值列”凭空成为首选
        if n_unique == 2:
            score += 0.3
            reasons.append(f"取值水平数为 {n_unique}，适合作为分类结局")
        elif item.get("confirmed_type") == "categorical" and 2 < n_unique <= 5:
            score += 0.2
            reasons.append(f"取值水平数为 {n_unique}，可作为多分类结局")
        if e["soft"]:
            score -= 1.0
            reasons.append("疑似治疗/分组类协变量，已降权：" + "、".join(e["soft"]))
        if score > 0:
            scored.append({
                "field": e["field"],
                "score": round(score, 2),
                "reasons": reasons,
                "suggested_type": item.get("confirmed_type"),
                "_pos": e["pos"],
            })

    scored.sort(key=lambda x: (-x["score"], x["_pos"]))
    candidates = scored

    # 兜底：没有任何候选时（典型是“无标签词 + 连续数值列”的表），按表尾向前给出数值列，
    # 避免出现“未识别到结局候选”而无法建模；分数最低，不会盖过词库命中的真实结局。
    if not candidates:
        fallback = []
        # 先选未被判为协变量的数值列，再退回包含协变量的列（如 Total/合计）
        for skip_soft in (True, False):
            for e in reversed(eligible):
                if e["item"].get("confirmed_type") != "numeric":
                    continue
                if skip_soft and e["soft"]:
                    continue
                if any(c["field"] == e["field"] for c in fallback):
                    continue
                fallback.append({
                    "field": e["field"],
                    "score": 0.1,
                    "reasons": ["无明确结局词，按数值列兜底（优先靠近表尾的列）"],
                    "suggested_type": e["item"].get("confirmed_type"),
                    "_pos": e["pos"],
                })
                if len(fallback) >= 3:
                    break
            if len(fallback) >= 3:
                break
        candidates = fallback

    for c in candidates:
        c.pop("_pos", None)
    # 宽表里二值列很多，候选列表截断到 20 条，避免前端出现上千条候选
    return candidates[:20]
