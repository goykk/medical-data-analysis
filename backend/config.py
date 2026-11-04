# medical-data-analysis/backend/config.py
"""集中配置：运行模式、端口、上传限制、随机种子、统计常量与医学参考区间。

所有可变参数均可通过环境变量覆盖，代码内不出现任何密钥或真实数据。
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv 为可选依赖
    pass


# ---------------- 运行模式 ----------------
# local：本地单用户模式，数据不出本机，历史存 SQLite
# cloud：云上模式（华为云 ECS + RDS + Redis + OBS）
MODE = os.getenv("MODE", "local").strip().lower()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8002"))

# 前端固定来源（CORS），严禁使用 "*"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:3001")
# 浏览器把 127.0.0.1 与 localhost 视为不同来源，只配一个会出现
# "页面能打开（GET 不预检）、POST 全部失败（预检被拒）"的经典问题 —— 默认两者都允许。
_DEFAULT_ORIGINS = [FRONTEND_ORIGIN]
for _a, _b in (("127.0.0.1", "localhost"), ("localhost", "127.0.0.1")):
    if _a in FRONTEND_ORIGIN:
        _alt = FRONTEND_ORIGIN.replace(_a, _b)
        if _alt not in _DEFAULT_ORIGINS:
            _DEFAULT_ORIGINS.append(_alt)
CORS_ORIGINS = [o.strip() for o in
                os.getenv("CORS_ORIGINS", ",".join(_DEFAULT_ORIGINS)).split(",") if o.strip()]

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv", ".txt", ".json", ".sav", ".dta"}

# API 访问令牌：非空时，除 /api/health 外的所有 /api/* 请求必须携带
# X-API-Token 头（或 Authorization: Bearer <token>）。默认空 = 不启用（本地单用户）。
# 云上模式（MODE=cloud，公网可达）必须设置，否则接口完全裸露。
API_TOKEN = os.getenv("API_TOKEN", "").strip()

# 内存中最多保留的数据集数量：超出后按"最旧的先淘汰"释放，当前激活数据集豁免。
# 写死常量，不做配置项暴露、不做前端可调。
MAX_DATASETS_IN_MEMORY = 8

# 文本类文件编码候选顺序（依次尝试）
ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]

# ---------------- 统计常量 ----------------
ALPHA = 0.05                # 显著性水平
RANDOM_SEED = 42            # 固定随机种子，保证可复现
SHAPIRO_MAX_N = 500         # n>500 时 Shapiro-Wilk 结果仅作参考
MIN_N_FOR_MODELING = 50     # 建模样本量红线
MIN_EPV = 10                # 每个变量所需最少事件数（events per variable）
# 高缺失列警示阈值：缺失比例超过此值仅在确认页标记警示，绝不自动排除
HIGH_MISSING_THRESHOLD = 0.8
VIF_THRESHOLD = 10          # 共线性风险阈值
FDR_METHOD = "fdr_bh"       # Benjamini-Hochberg 校正
# 组间比较（含单变量筛选）护栏：平均每组样本量低于该值时，统计量必然退化
# （Kruskal-Wallis 的 η² 趋近 1、p 值无意义），必须拒绝计算而非静默输出。
# 不设单独的水平数上限——30 家医院的多中心研究（每中心样本充足）是正当用法，
# 退化与否取决于"平均每组样本量"，而非水平数本身。
MIN_AVG_GROUP_SIZE = 2

# ---------------- 医学参考区间 ----------------
# 说明：以下为成人常规检验项目参考区间，用于图形叠加与异常值提示，
# 依据《全国临床检验操作规程》（第 4 版）与常用临床检验学教材通用区间；
# 不同实验室方法学存在差异，实际临床判断应以本院检验科参考区间为准。
# key 为标准化字段名或常见中文别名，low/high 为参考区间上下限（含）。
REFERENCE_RANGES = {
    "bmi":            {"low": 18.5, "high": 24.0, "unit": "kg/m²", "label": "体质指数",
                       "source": "中国成人体质指数分类标准 WS/T 428-2013"},
    "sbp":            {"low": 90.0, "high": 140.0, "unit": "mmHg", "label": "收缩压",
                       "source": "《中国高血压防治指南（2018 年修订版）》"},
    "dbp":            {"low": 60.0, "high": 90.0, "unit": "mmHg", "label": "舒张压",
                       "source": "《中国高血压防治指南（2018 年修订版）》"},
    "glucose":        {"low": 3.9, "high": 6.1, "unit": "mmol/L", "label": "空腹血糖",
                       "source": "《中国 2 型糖尿病防治指南（2020 年版）》"},
    "hba1c":          {"low": 4.0, "high": 6.0, "unit": "%", "label": "糖化血红蛋白",
                       "source": "《中国 2 型糖尿病防治指南（2020 年版）》"},
    "tc":             {"low": 3.1, "high": 5.2, "unit": "mmol/L", "label": "总胆固醇",
                       "source": "《中国成人血脂异常防治指南（2016 年修订版）》"},
    "tg":             {"low": 0.4, "high": 1.7, "unit": "mmol/L", "label": "甘油三酯",
                       "source": "《中国成人血脂异常防治指南（2016 年修订版）》"},
    "hdl":            {"low": 1.0, "high": 1.8, "unit": "mmol/L", "label": "高密度脂蛋白胆固醇",
                       "source": "《中国成人血脂异常防治指南（2016 年修订版）》"},
    "ldl":            {"low": 1.5, "high": 3.4, "unit": "mmol/L", "label": "低密度脂蛋白胆固醇",
                       "source": "《中国成人血脂异常防治指南（2016 年修订版）》"},
    "hemoglobin":     {"low": 115.0, "high": 150.0, "unit": "g/L", "label": "血红蛋白",
                       "source": "《全国临床检验操作规程》第 4 版（成人女性参考区间）"},
    "wbc":            {"low": 3.5, "high": 9.5, "unit": "×10⁹/L", "label": "白细胞计数",
                       "source": "《全国临床检验操作规程》第 4 版"},
    "platelet":       {"low": 125.0, "high": 350.0, "unit": "×10⁹/L", "label": "血小板计数",
                       "source": "《全国临床检验操作规程》第 4 版"},
    "alt":            {"low": 7.0, "high": 40.0, "unit": "U/L", "label": "丙氨酸氨基转移酶",
                       "source": "《全国临床检验操作规程》第 4 版"},
    "ast":            {"low": 13.0, "high": 35.0, "unit": "U/L", "label": "天门冬氨酸氨基转移酶",
                       "source": "《全国临床检验操作规程》第 4 版"},
    "creatinine":     {"low": 41.0, "high": 81.0, "unit": "μmol/L", "label": "血肌酐",
                       "source": "《全国临床检验操作规程》第 4 版（成人女性参考区间）"},
    "bun":            {"low": 2.6, "high": 7.5, "unit": "mmol/L", "label": "尿素氮",
                       "source": "《全国临床检验操作规程》第 4 版"},
    "uric_acid":      {"low": 155.0, "high": 357.0, "unit": "μmol/L", "label": "血尿酸",
                       "source": "《全国临床检验操作规程》第 4 版（成人女性参考区间）"},
    "crp":            {"low": 0.0, "high": 8.0, "unit": "mg/L", "label": "C 反应蛋白",
                       "source": "《全国临床检验操作规程》第 4 版"},
}

# 字段名 -> 参考区间 key 的别名映射（中英文常见列名）
REFERENCE_ALIASES = {
    "体质指数": "bmi", "bmi": "bmi", "body_mass_index": "bmi",
    "收缩压": "sbp", "sbp": "sbp", "systolic": "sbp", "systolic_bp": "sbp",
    "舒张压": "dbp", "dbp": "dbp", "diastolic": "dbp", "diastolic_bp": "dbp",
    "空腹血糖": "glucose", "血糖": "glucose", "fpg": "glucose", "glucose": "glucose", "glu": "glucose",
    "糖化血红蛋白": "hba1c", "hba1c": "hba1c",
    "总胆固醇": "tc", "胆固醇": "tc", "tc": "tc", "cholesterol": "tc",
    "甘油三酯": "tg", "tg": "tg", "triglyceride": "tg",
    "高密度脂蛋白": "hdl", "hdl": "hdl", "hdl_c": "hdl",
    "低密度脂蛋白": "ldl", "ldl": "ldl", "ldl_c": "ldl",
    "血红蛋白": "hemoglobin", "hb": "hemoglobin", "hemoglobin": "hemoglobin",
    "白细胞": "wbc", "白细胞计数": "wbc", "wbc": "wbc",
    "血小板": "platelet", "血小板计数": "platelet", "plt": "platelet", "platelet": "platelet",
    "谷丙转氨酶": "alt", "alt": "alt",
    "谷草转氨酶": "ast", "ast": "ast",
    "肌酐": "creatinine", "血肌酐": "creatinine", "cr": "creatinine", "creatinine": "creatinine",
    "尿素氮": "bun", "bun": "bun",
    "尿酸": "uric_acid", "血尿酸": "uric_acid", "ua": "uric_acid",
    "c反应蛋白": "crp", "crp": "crp",
}


def reference_range_for(column_name: str):
    """按字段名（支持中英文别名）查找医学参考区间，未匹配返回 None。"""
    if not column_name:
        return None
    key = str(column_name).strip().lower()
    std = REFERENCE_ALIASES.get(key, key)
    return REFERENCE_RANGES.get(std)
