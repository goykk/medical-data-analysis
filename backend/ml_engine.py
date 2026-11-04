# medical-data-analysis/backend/ml_engine.py
"""机器学习建模与评估：sklearn Pipeline、模型对比、评估三件套、SHAP、新样本预测。

准确性红线：
- 预处理与模型一律使用 sklearn 成熟接口，严禁手写训练逻辑；
- 所有评估指标基于**测试集**预测结果计算，图表标题注明 n=测试集样本量；
- 仅在 DCA 与 Hosmer-Lemeshow 检验两处按已批准例外手写公式（无成熟库实现），
  二者均配手工核算的单元测试（见 tests/test_dca_hl.py）；
- SHAP：树模型用 TreeExplainer，线性模型用 LinearExplainer，禁用 KernelExplainer。
"""

import base64
import io
import logging
import os
import re
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             get_scorer, mean_squared_error, r2_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MIN_EPV, MIN_N_FOR_MODELING, RANDOM_SEED

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# dataset_id 白名单：只允许字母/数字/下划线/连字符（生产环境为 uuid4().hex[:12]）。
# 字符集不含 "."、"/"、"\"，从根上杜绝 DELETE /api/ml/model/{dataset_id} 等入口的路径穿越。
_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

log = logging.getLogger(__name__)


def _model_path(dataset_id: str) -> str:
    """模型文件路径：校验 id 格式，并确认真实路径仍在 MODEL_DIR 内（双重防护）。"""
    if not _DATASET_ID_RE.fullmatch(str(dataset_id)):
        raise MLError("非法的数据集 id")
    path = os.path.realpath(os.path.join(MODEL_DIR, f"model_{dataset_id}.joblib"))
    if os.path.dirname(path) != os.path.realpath(MODEL_DIR):
        raise MLError("非法的模型路径")
    return path

# 建模一律单进程执行（n_jobs=1）：本模块由 FastAPI 请求线程直接调用，
# 若用 joblib/loky 派生子进程，Windows 下 worker 会被异常终止
# （"A worker process ... was unexpectedly terminated"），整次建模直接 500。
# 医疗数据集量级下串行开销可以忽略，稳定性优先。
N_JOBS = 1

# 大数据集阈值：训练集超过此行数时分层采样，避免训练时间过长
LARGE_DATASET_THRESHOLD = 50000
LARGE_DATASET_TRAIN_SIZE = 30000

# ---- 建模最低样本量 ----
# 低于 30 行无法稳定切分训练/测试；分类任务再要求“最少类别 ≥ 5”，否则连 5 折
# 分层交叉验证都无法保证每折有该类别。旧实现直接以 min_class < 10 跳过，导致
# 782 行、907 行的数据集也被标成“样本量严重不足”——现在只在确实无法建模时跳过，
# 5~10 例改为“指标不稳定”的显式警示。
MIN_N_FOR_SPLIT = 30
MIN_CLASS_FOR_MODELING = 5
MIN_CLASS_WARN = 10
# 回归按结局分位数分层切分只用于极小样本（<100 行）：这类数据随机切分最容易把
# 极端结局全分到测试集而出现 R²<0；100 行以上随机切分足够均匀，强分层反而会改变
# 原有指标口径，因此不做干预。
REGRESSION_STRATIFY_MAX_N = 100

# 规模自适应：训练集超过该行数时切到快速模式（3 折 CV + 150 棵树）。
# 医疗数据在 2 万行以上时，5 折 × 双模型 × 校准的串行拟合是主要耗时来源。
FAST_MODE_ROWS = 20000
CV_FOLDS = 5
FAST_CV_FOLDS = 3
RF_TREES = 300
FAST_RF_TREES = 150

# 训练结果缓存：同一数据集版本 + 同一特征/结局/任务/口径的重复请求直接命中。
# 只缓存最终产物，不缓存“全量数据 fit 出的变换矩阵”——那会把测试折信息带进预处理，
# 造成数据泄漏；交叉验证仍需每折独立拟合。
TRAIN_CACHE_SIZE = 6
_TRAIN_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()

# ---- P1：宽表降维 + 结构性泄露自动剔除 ----
# 宽表先做方差过滤 / 单变量 F 检验 / 共线性去冗，只保留 top-k（数值列 + 类别列合并计算）。
SCREEN_TRIGGER_FEATURES = 50      # 特征数超过该值才启用筛选
SCREEN_TOP_K = 40                 # 筛选后保留的特征上限（分类任务默认）
SCREEN_TOP_K_REGRESSION = 80      # 回归宽表可放宽：连续结局对特征更敏感，40 列会明显损失 R²
SCREEN_TOP_K_MIN_ROWS = 500       # 仅当训练集 ≥ 该行数时才放宽回归 top-k，避免小样本过拟合
SCREEN_CORR_MAX = 0.95            # 相关系数超过该值的冗余列剔除
# 自动剔除阈值刻意取 0.99（报告用的 0.95 只做提示）：
# WDBC 的 radius_worst 等真实强预测因子单变量 AUC≈0.97，不能因为“预测力强”就删掉。
LEAK_AUTO_DROP_THRESHOLD = 0.99
# 与自动建模 prepare 使用同一个警告阈值（automl.leakage.DEFAULT_LEAKAGE_THRESHOLD=0.95），
# 避免出现“prepare 报了泄露、常规训练却毫无提示”的口径分裂。
LEAK_WARN_THRESHOLD = 0.95

# 概率校准方法：isotonic 需要充足样本（每类约 ≥1000）才能稳定拟合，
# 小样本用 Platt（sigmoid）平滑，避免等渗回归过拟合把概率压成 0/1 台阶。
ISOTONIC_MIN_N = 1000

# 数据集 -> 模型产物（内存索引）。上传新数据集时旧模型立即失效。
MODEL_STORE: dict = {}
# 数据集 -> 最近一次“训练成功但模型被丢弃”的记录（如回归 R²<0）；
# 预测页据此显示具体原因，而不是笼统的“请先完成建模”。
MODEL_DISCARDED: dict = {}


class MLError(Exception):
    """建模无法执行时抛出，携带面向用户的明确原因。"""


# ------------------------------------------------------------------
# 预处理与模型
# ------------------------------------------------------------------

def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """缺失值填补 + 编码 + 标准化；全部封装在 Pipeline 内，只在训练折上拟合。"""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ], remainder="drop")


def _candidate_models(task: str, n_estimators: int = RF_TREES,
                      fast_mode: bool = False) -> dict:
    """候选模型。

    standard（默认）：可解释线性模型 + 随机森林（300 棵树，SHAP 原生支持）。
    fast_mode（大样本）：可解释线性模型 + HistGradientBoosting——直方图提升在 3 万行
    以上通常比 300 棵随机森林快一个数量级；代价是 HGB 无原生 importance/SHAP，
    由置换重要性与明确的 SHAP 降级提示兜底。
    """
    if task == "classification":
        models = {
            # 医疗场景优先可解释模型：逻辑回归作为首选候选
            "logistic": ("逻辑回归（Logistic Regression）",
                         LogisticRegression(max_iter=2000, class_weight="balanced",
                                            random_state=RANDOM_SEED)),
        }
        if fast_mode:
            models["hist_gb"] = ("直方图梯度提升（HistGradientBoosting）",
                                 HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1,
                                                                early_stopping=True,
                                                                random_state=RANDOM_SEED))
        else:
            models["random_forest"] = ("随机森林（Random Forest）",
                                       RandomForestClassifier(n_estimators=n_estimators,
                                                              min_samples_leaf=2,
                                                              class_weight="balanced_subsample",
                                                              random_state=RANDOM_SEED,
                                                              n_jobs=N_JOBS))
    else:
        models = {
            "linear": ("多元线性回归（Linear Regression）", LinearRegression()),
        }
        if fast_mode:
            models["hist_gb"] = ("直方图梯度提升（HistGradientBoosting）",
                                 HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1,
                                                               early_stopping=True,
                                                               random_state=RANDOM_SEED))
        else:
            models["random_forest"] = ("随机森林回归（Random Forest）",
                                       RandomForestRegressor(n_estimators=n_estimators,
                                                             min_samples_leaf=2,
                                                             random_state=RANDOM_SEED,
                                                             n_jobs=N_JOBS))
    return models


def _train_cache_key(cache_key, features, target, task, variable_types, with_shap):
    """缓存指纹：数据集版本 + 特征集合 + 结局 + 任务 + 变量类型口径 + 是否算 SHAP。"""
    vt = tuple(sorted((f, variable_types.get(f)) for f in features))
    return (cache_key, tuple(sorted(features)), target, task, vt, bool(with_shap))


def clear_cache(dataset_id: str | None = None) -> int:
    """清除训练结果缓存；给定 dataset_id 时只清该数据集（key 形如 `id:版本号`）。"""
    if not dataset_id:
        n = len(_TRAIN_CACHE)
        _TRAIN_CACHE.clear()
        return n
    keys = [k for k in _TRAIN_CACHE
            if isinstance(k[0], str) and k[0].split(":", 1)[0] == dataset_id]
    for k in keys:
        _TRAIN_CACHE.pop(k, None)
    return len(keys)


def _split(X, y, task: str):
    """划分训练/测试集。

    回归任务按结局分位数分层切分：小样本随机切分很容易出现“测试集的结局全落在
    训练集没见过的区间”，导致 R² 直接为负（QSAR 几十行的数据就是这种情形）。
    分位数太细导致每层样本不足时自动退回普通随机切分。
    """
    if task == "classification":
        return train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
    stratify = None
    try:
        if len(y) < REGRESSION_STRATIFY_MAX_N:
            bins = pd.qcut(pd.Series(np.asarray(y, dtype=float)), q=5, labels=False,
                           duplicates="drop")
            if pd.Series(bins).nunique() >= 2 and pd.Series(bins).value_counts().min() >= 4:
                stratify = np.asarray(bins)
    except Exception:
        stratify = None
    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=stratify)


def _auto_drop_leaky_features(df: pd.DataFrame, features: list, target: str, task: str):
    """自动剔除结构性泄露列：单变量判别力 > 0.99（接近目标副本/由结局反推）。

    返回 (保留特征, 自动剔除列, 仅告警列)。告警列指判别力落在 0.95~0.99 的特征：
    它们与 prepare 的泄露判定口径一致，但不自动删除（WDBC 的 radius_worst 等真实强
    预测因子 AUC≈0.97，删掉会损失临床信息），只把风险显式提示给用户。
    """
    try:
        from automl.leakage import detect_leakage
        findings = detect_leakage(df[features + [target]], target=target, task=task,
                                  threshold=LEAK_WARN_THRESHOLD)
    except Exception as exc:
        log.warning("泄露列检测失败（已跳过自动剔除）：%s", exc)
        return features, [], []
    dropped = [f.column for f in findings
               if f.verdict == "leak" and f.column in features
               and f.score >= LEAK_AUTO_DROP_THRESHOLD]
    warned = [f for f in findings
              if f.column in features and LEAK_WARN_THRESHOLD <= f.score < LEAK_AUTO_DROP_THRESHOLD]
    kept = [f for f in features if f not in dropped]
    return kept, dropped, warned


def _screen_wide_features(X_train, y_train, numeric: list, categorical: list, task: str):
    """宽表降维：方差过滤 + 单变量 F 检验 + 共线性去冗，返回 (数值列, 类别列, 信息)。

    只在训练集上计算分数，测试集不参与，避免用测试标签做特征选择。

    修复点：旧实现只对数值列打分，类别列原样全量保留；当宽表全是 0/1 指纹位这类
    “数字编码分类变量”时，numeric 为空、categorical 上千，`SCREEN_TOP_K - len(cat)`
    退化为 1，筛选等于没做（QSAR 1026 列 × 数百行训练要几分钟）。现在类别列按
    “编码后单变量 F 检验”参与排序，并与数值列合并后统一取 top-K。
    """
    from sklearn.feature_selection import f_classif, f_regression

    # 回归任务对连续结局更敏感，40 列会明显损失 R²；但小样本放宽会过拟合，
    # 因此只在训练集 ≥ SCREEN_TOP_K_MIN_ROWS 时放宽到 SCREEN_TOP_K_REGRESSION。
    top_k = SCREEN_TOP_K
    if task != "classification" and len(X_train) >= SCREEN_TOP_K_MIN_ROWS:
        top_k = max(SCREEN_TOP_K, SCREEN_TOP_K_REGRESSION)
    # 回归还要受样本量约束：特征数不超过训练集行数的 1/5（最少保留 5 个）。
    # 24~60 行的宽表配 40 个特征会严重过拟合（曾出现 R²=-11），按 1/5 缩放后
    # 既压住过拟合，又不至于把 100~400 行的正常数据砍得过狠。
    if task != "classification":
        top_k = min(top_k, max(5, len(X_train) // 5))

    info = {"method": "方差过滤 + 单变量F检验（含类别列编码后打分） + 共线性(>0.95)去冗",
            "top_k": top_k}

    num = []
    for f in numeric:
        s = pd.to_numeric(X_train[f], errors="coerce")
        if s.notna().sum() < max(10, int(0.02 * len(s))) or s.nunique(dropna=True) <= 1:
            continue
        num.append(f)
    info["variance_dropped"] = [f for f in numeric if f not in num]

    # 类别列：同样先做缺失/常量过滤，再编码为整数参与单变量打分
    cat = []
    for f in categorical:
        s = X_train[f]
        if s.notna().sum() < max(10, int(0.02 * len(s))) or s.nunique(dropna=True) <= 1:
            continue
        cat.append(f)
    info["categorical_dropped"] = [f for f in categorical if f not in cat][:20]

    if not num and not cat:
        return [], [], info

    blocks, names = [], []
    if num:
        M_num = X_train[num].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        blocks.append(M_num)
        names.extend(num)
    if cat:
        cols = []
        for f in cat:
            codes, _ = pd.factorize(X_train[f], sort=True)   # 缺失 -> -1
            codes = np.where(codes < 0, np.nan, codes).astype(float)
            cols.append(codes)
        blocks.append(np.column_stack(cols))
        names.extend(cat)

    M = np.column_stack(blocks)
    with np.errstate(all="ignore"):
        med = np.nanmedian(M, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    miss = np.where(~np.isfinite(M))
    M[miss] = med[miss[1]]
    try:
        scores, _ = (f_classif(M, y_train) if task == "classification"
                     else f_regression(M, y_train))
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        log.warning("宽表单变量筛选失败，退化为原顺序：%s", exc)
        scores = np.ones(M.shape[1], dtype=float)

    order = np.argsort(-scores)
    # 共线性只在数值列之间检查（类别列不做相关系数去冗）
    corr = None
    if len(num) > 1:
        try:
            corr = np.corrcoef(M[:, :len(num)], rowvar=False)
            if np.ndim(corr) == 0:
                corr = None
        except Exception:
            corr = None

    picked = []
    picked_num_idx = []
    num_pos = {f: i for i, f in enumerate(num)}
    for i in order:
        if len(picked) >= max(top_k, 1):
            break
        name = names[int(i)]
        ni = num_pos.get(name)
        if ni is not None and corr is not None and picked_num_idx:
            if np.nanmax(np.abs(corr[ni, picked_num_idx])) > SCREEN_CORR_MAX:
                continue
        picked.append(int(i))
        if ni is not None:
            picked_num_idx.append(ni)

    sel_names = [names[i] for i in picked]
    sel_num = [f for f in sel_names if f in num_pos]
    sel_cat = [f for f in sel_names if f not in num_pos]
    info.update({
        "top_scores": [{"feature": names[int(i)], "score": round(float(scores[int(i)]), 3)}
                       for i in order[:10]],
    })
    return sel_num, sel_cat, info


# ------------------------------------------------------------------
# 评估指标
# ------------------------------------------------------------------

def _classification_metrics(y_true, y_pred, y_prob, classes) -> dict:
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    binary = len(classes) == 2
    if binary:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else None
        specificity = tn / (tn + fp) if (tn + fp) else None
        auc = float(roc_auc_score(y_true, y_prob[:, 1])) if y_prob is not None else None
    else:
        sensitivity = specificity = None
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")) \
            if y_prob is not None else None
    per_class = {}
    for cls in classes:
        key = str(cls)
        if key in report:
            per_class[key] = {
                "recall": round(report[key]["recall"], 4),
                "precision": round(report[key]["precision"], 4),
                "f1": round(report[key]["f1-score"], 4),
                "support": int(report[key]["support"]),
            }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "sensitivity": round(float(sensitivity), 4) if sensitivity is not None else None,
        "specificity": round(float(specificity), 4) if specificity is not None else None,
        "confusion_matrix": cm,
        "per_class": per_class,
        "macro_avg_recall": round(report["macro avg"]["recall"], 4) if "macro avg" in report else None,
    }


def _regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": round(rmse, 4), "r2": round(float(r2_score(y_true, y_pred)), 4),
            "mae": round(float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))), 4)}


def roc_analysis(y_true, y_prob, n_test: int) -> dict:
    """ROC 曲线 + AUC + 约登指数最佳截断值（基于测试集）。"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # roc_curve 首个阈值为 +inf（表示“全部判为阴性”），替换为 1.0 以便 JSON 序列化
    thresholds = np.where(np.isfinite(thresholds), thresholds, 1.0)
    auc = float(roc_auc_score(y_true, y_prob))
    youden = tpr - fpr
    best = int(np.argmax(youden))
    return {
        "fpr": [round(float(v), 5) for v in fpr],
        "tpr": [round(float(v), 5) for v in tpr],
        "thresholds": [round(float(v), 5) for v in thresholds],
        "auc": round(auc, 4),
        "youden_threshold": round(float(thresholds[best]), 4),
        "youden_index": round(float(youden[best]), 4),
        "sensitivity_at_best": round(float(tpr[best]), 4),
        "specificity_at_best": round(float(1 - fpr[best]), 4),
        "n": n_test,
        "title": f"ROC 曲线（测试集 n={n_test}）",
    }


def hl_groups(n: int) -> int:
    """Hosmer-Lemeshow 分组数随样本量自适应。

    固定 10 组问题明显：n<100 时每组不足 10 例，检验退化成噪声；n>500 时
    10 组又过粗，抓不到局部校准偏差。经验取值：n<50→3，50–99→5，
    100–500→10，>500→20。
    """
    if n < 50:
        return 3
    if n < 100:
        return 5
    if n <= 500:
        return 10
    return 20


def hosmer_lemeshow(y_true, y_prob, groups: int | None = None) -> dict:
    """Hosmer-Lemeshow 拟合优度检验（手写实现，已批准例外，配手工核算单元测试）。

    H = Σ_g (O₁g − E₁g)² / [ n_g · p̄_g · (1 − p̄_g) ]，自由度 = g − 2，
    p 值由 scipy.stats.chi2 生存函数给出（库调用）。
    groups 缺省时按样本量自适应（见 hl_groups）。
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    groups = int(groups) if groups else hl_groups(n)
    if n < groups * 2:
        groups = max(2, n // 5)
    order = np.argsort(p)
    y_sorted, p_sorted = y[order], p[order]
    idx = np.array_split(np.arange(n), groups)

    stat, table = 0.0, []
    for part in idx:
        if part.size == 0:
            continue
        n_g = part.size
        o1 = float(y_sorted[part].sum())
        e1 = float(p_sorted[part].sum())
        pbar = e1 / n_g
        denom = n_g * pbar * (1 - pbar)
        if denom > 1e-12:
            stat += (o1 - e1) ** 2 / denom
        table.append({"n": int(n_g), "observed": round(o1, 3), "expected": round(e1, 3),
                      "mean_prob": round(pbar, 4)})

    from scipy import stats as _st
    dof = max(1, groups - 2)
    p_value = float(_st.chi2.sf(stat, dof))
    # p 保持完整精度：前端依据 p>0.05 判断校准好坏，round(p,4) 在 0.05 边界会翻转结论
    from analyzer import full_precision_p
    return {"statistic": round(float(stat), 4), "dof": int(dof), "p": full_precision_p(p_value),
            "groups": len(table), "table": table,
            "note": "Hosmer-Lemeshow 检验为按批准例外手写实现，已配手工核算单元测试；"
                    "p 值由 scipy.stats.chi2 生存函数给出。p>0.05 提示校准良好。"}


def calibration_analysis(y_true, y_prob, n_test: int, n_bins: int = 10) -> dict:
    """校准曲线（sklearn.calibration_curve）+ Hosmer-Lemeshow 检验。

    曲线分箱固定 10 箱便于横向对比；HL 的分组数按样本量自适应（见 hl_groups）。
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    hl = hosmer_lemeshow(y_true, y_prob)
    return {
        "prob_true": [round(float(v), 5) for v in prob_true],
        "prob_pred": [round(float(v), 5) for v in prob_pred],
        "hosmer_lemeshow": hl,
        "n": n_test,
        "title": f"校准曲线（测试集 n={n_test}）",
    }


def decision_curve_analysis(y_true, y_prob, n_points: int = 99) -> dict:
    """决策曲线分析 DCA（手写实现，已批准例外，配手工核算单元测试）。

    净获益 NB(pt) = TP/n − (FP/n)·[pt/(1−pt)]；
    “全部干预”参考线 NB_all = 患病率 − (1−患病率)·[pt/(1−pt)]；“全不干预”恒为 0。
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    prevalence = float(y.mean())
    thresholds = np.linspace(0.01, 0.99, n_points)

    model, treat_all = [], []
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        w = pt / (1.0 - pt)
        model.append(tp / n - fp / n * w)
        treat_all.append(prevalence - (1.0 - prevalence) * w)

    best_i = int(np.argmax(model)) if model else 0
    return {
        "thresholds": [round(float(t), 4) for t in thresholds],
        "net_benefit_model": [round(float(v), 5) for v in model],
        "net_benefit_all": [round(float(v), 5) for v in treat_all],
        "net_benefit_none": [0.0] * len(thresholds),
        "prevalence": round(prevalence, 4),
        "best_threshold": round(float(thresholds[best_i]), 4),
        "best_net_benefit": round(float(model[best_i]), 5) if model else None,
        "n": int(len(y)),
        "title": f"决策曲线分析 DCA（测试集 n={int(len(y))}）",
        "note": "DCA 为按批准例外手写实现，已配手工核算单元测试。",
    }


def n_test_of(y) -> int:
    return int(len(y))


def multiclass_evaluation(y_prob_multi, y_test, classes, n_test: int) -> dict:
    """多分类 one-vs-rest 评估三件套。

    - ROC / 校准：每个类别一条曲线（该类为正类、其余为负类）；
    - DCA：每个类别一条净获益曲线，另给各类算术平均的宏平均曲线用于总览。
    返回结构与二分类的 roc / calibration / dca 兼容（roc.auc 为 OvR 宏平均 AUC），
    使前端原有渲染逻辑在二分类与多分类下走同一套代码。
    """
    prob = np.asarray(y_prob_multi, dtype=float)
    y = np.asarray(y_test)
    if prob.ndim != 2 or prob.shape[1] != len(classes):
        raise MLError("多分类概率矩阵与类别数不一致")
    if prob.shape[1] < 3:
        raise MLError("至少需要 3 个类别才能进行多分类评估")

    roc_curves, cal_curves, dca_curves = [], [], []
    for idx, cls in enumerate(classes):
        y_bin = (y == idx).astype(int)
        p = prob[:, idx]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue          # 测试集缺少该类，OvR 曲线无定义，跳过该类别
        roc_curves.append({"label": str(cls), **roc_analysis(y_bin, p, n_test)})
        cal_curves.append({"label": str(cls), **calibration_analysis(y_bin, p, n_test)})
        dca_curves.append({"label": str(cls), **decision_curve_analysis(y_bin, p)})

    if not roc_curves:
        raise MLError("测试集中没有可用于多分类评估的类别")

    macro_auc = round(float(np.mean([c["auc"] for c in roc_curves])), 4)
    nb_matrix = np.vstack([np.asarray(d["net_benefit_model"], dtype=float) for d in dca_curves])
    nb_macro = nb_matrix.mean(axis=0)
    thresholds = list(dca_curves[0]["thresholds"])
    best_i = int(np.argmax(nb_macro))

    return {
        "roc": {
            "multi": True, "classes": roc_curves, "auc": macro_auc, "macro_auc": macro_auc,
            "n": n_test,
            "title": f"ROC 曲线（测试集 n={n_test}，多分类 OvR，宏平均 AUC={macro_auc}）",
        },
        "calibration": {
            "multi": True, "classes": cal_curves, "n": n_test,
            "title": f"校准曲线（测试集 n={n_test}，多分类 OvR）",
        },
        "dca": {
            "multi": True,
            "classes": [{"label": d["label"], "best_threshold": d["best_threshold"],
                         "best_net_benefit": d["best_net_benefit"], "prevalence": d["prevalence"]}
                        for d in dca_curves],
            "thresholds": thresholds,
            "net_benefit_model": [round(float(v), 5) for v in nb_macro],
            "net_benefit_all": dca_curves[0]["net_benefit_all"],
            "net_benefit_none": [0.0] * len(thresholds),
            "prevalence": dca_curves[0]["prevalence"],
            "best_threshold": round(float(thresholds[best_i]), 4),
            "best_net_benefit": round(float(nb_macro[best_i]), 5),
            "n": n_test,
            "title": f"决策曲线 DCA（测试集 n={n_test}，各类 OvR 净获益宏平均）",
            "note": "多分类 DCA 曲线为各类 one-vs-rest 净获益的算术平均，单类结果见 classes。",
        },
    }


def _cv_scores_with_oof(pipe, X, y, cv, scoring: str, task: str, n_classes: int | None):
    """一次交叉验证同时产出折分数与 OOF 概率（校准直接复用这批折，不再单独拟合）。

    返回 (每折分数 ndarray, OOF 概率矩阵|None)。分类任务的 OOF 概率用于校准；
    回归任务只算 R² 折分数。
    """
    scorer = get_scorer(scoring)
    scores = []
    oof = None
    if task == "classification" and n_classes:
        oof = np.full((len(y), int(n_classes)), np.nan, dtype=float)
    for tr, va in cv.split(X, y):
        fold = clone(pipe)
        fold.fit(X.iloc[tr], y[tr])
        # 单折评分失败不得打断整次训练：少数类在某一折的验证集中缺席时，
        # 多分类 roc_auc_ovr 会因 y_true/y_score 类别数不一致直接抛错（曾造成 500）。
        # 记 NaN 并在上层标记“交叉验证未产出有效结果”，测试集指标仍正常返回。
        try:
            scores.append(float(scorer(fold, X.iloc[va], y[va])))
        except Exception as exc:
            log.warning("交叉验证单折评分失败（记为 NaN）：%s", exc)
            scores.append(float("nan"))
        if oof is not None:
            proba = np.asarray(fold.predict_proba(X.iloc[va]), dtype=float)
            order = list(getattr(fold.named_steps["model"], "classes_",
                                 range(proba.shape[1])))
            for col, cls in enumerate(order):
                oof[va, int(cls)] = proba[:, col]
    return np.asarray(scores, dtype=float), oof


def _calibrate_oof(oof, y_train, probs_test, method: str, n_classes: int, folds: int):
    """用 OOF 概率拟合校准器，把测试集原始概率映射为校准概率。

    二分类直接对正类概率做 isotonic / Platt；多分类按 one-vs-rest 逐类校准后归一化。
    """
    out = np.zeros_like(probs_test, dtype=float)
    eps = 1e-9

    def _fit_one(p_oof, y_bin, p_test):
        p_oof = np.clip(p_oof, eps, 1 - eps)
        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            cal.fit(p_oof, y_bin)
            return np.clip(cal.predict(np.clip(p_test, 0, 1)), eps, 1 - eps)
        z = np.log(p_oof / (1 - p_oof)).reshape(-1, 1)
        cal = LogisticRegression(max_iter=1000).fit(z, y_bin)
        pt = np.clip(p_test, eps, 1 - eps)
        zt = np.log(pt / (1 - pt)).reshape(-1, 1)
        return np.clip(cal.predict_proba(zt)[:, 1], eps, 1 - eps)

    if n_classes == 2:
        cp = _fit_one(oof[:, 1], (y_train == 1).astype(int), probs_test[:, 1])
        out[:, 1], out[:, 0] = cp, 1 - cp
        return out, f"OOF校准（{method}, 复用 {folds} 折 CV）"

    for k in range(n_classes):
        out[:, k] = _fit_one(oof[:, k], (y_train == k).astype(int), probs_test[:, k])
    total = out.sum(axis=1, keepdims=True)
    out = np.divide(out, total, out=np.full_like(out, 1.0 / n_classes), where=total > 0)
    return np.clip(out, eps, 1 - eps), f"OOF校准（{method}, 复用 {folds} 折 CV，多分类 OvR）"


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, features: list, target: str,
                       task: str = "classification", variable_types: dict | None = None,
                       with_shap: bool = True, cache_key: str | None = None) -> dict:
    """训练并评估模型。task: classification | regression。

    cache_key 由调用方给出（数据集 id + 版本号），相同数据集版本、相同特征/结局/任务/
    变量类型口径的重复训练会直接命中缓存；换数据、换字段、换口径都会自然失效。
    """
    variable_types = variable_types or {}
    features = [f for f in features if f != target and f in df.columns]
    if not features:
        raise MLError("请至少选择一个特征变量")
    if target not in df.columns:
        raise MLError(f"结局变量「{target}」不存在")

    cache_fp = _train_cache_key(cache_key, features, target, task, variable_types, with_shap) \
        if cache_key else None
    if cache_fp is not None and cache_fp in _TRAIN_CACHE:
        _TRAIN_CACHE.move_to_end(cache_fp)
        hit = dict(_TRAIN_CACHE[cache_fp])
        hit["from_cache"] = True
        return hit

    # 结构性泄露自动剔除：特征若几乎等价于结局（判别力 >0.99），继续参与训练只会给出
    # 虚高指标（Absenteeism 的 AUC=0.999、NHANES 的 AUC=1.000 都是这么来的）。
    features, leak_dropped, leak_warned = _auto_drop_leaky_features(df, features, target, task)
    if not features:
        # 全列泄露属于“数据无法建模”，不是系统错误：返回 skipped 让前端按跳过展示，
        # 避免用户把它当成 500/异常反复重试。
        return {"skipped": True, "task": task, "n_total": int(len(df)),
                "reason": (f"所有候选特征都被判定为疑似泄露列（单变量判别力 > "
                           f"{LEAK_AUTO_DROP_THRESHOLD}），无法建模。"
                           "请更换特征列，或确认这些列是否由结局派生。"),
                "skip_detail": {"n_total": int(len(df)),
                                "target": target,
                                "leak_columns": leak_dropped[:20],
                                "leak_threshold": LEAK_AUTO_DROP_THRESHOLD},
                "warnings": [f"已判定为泄露并剔除的特征："
                             + "、".join(leak_dropped[:8])
                             + ("等" if len(leak_dropped) > 8 else "") + "。"]}

    # 仅剔除结局缺失的行（结局无法填补）；特征缺失保留，交给 Pipeline 内的
    # imputer 在训练折上拟合，避免在划分前整行丢弃造成信息损失与选择性偏倚。
    data = df[features + [target]].dropna(subset=[target])
    data = data.copy()
    # 防御性类型收敛：确认页允许把某列显式设为 numeric，但原列仍可能是 object
    # （如混入 "NaNN"/"未知" 等脏值）。此类列若直接进 SimpleImputer(median) 会抛
    # ValueError 并让整个 /api/ml/train 返回 500；这里统一把无法转换的值转成缺失，
    # 由 imputer 填补，与解析阶段的处理保持一致。
    for f in features:
        if variable_types.get(f) != "categorical":
            col = data[f]
            if col.dtype == object or pd.api.types.is_string_dtype(col):
                data[f] = pd.to_numeric(col, errors="coerce")
    n_total = int(len(data))
    X = data[features]
    y_raw = data[target]

    if task == "classification":
        y = y_raw.astype(str).to_numpy()
        classes = sorted(pd.unique(y))
        if len(classes) < 2:
            raise MLError("分类任务要求结局至少 2 个类别")
        y_idx = np.array([classes.index(v) for v in y])
        class_counts = {c: int((y == c).sum()) for c in classes}
    else:
        y_idx = pd.to_numeric(y_raw, errors="coerce").to_numpy(float)
        if np.isnan(y_idx).any():
            raise MLError("回归任务的结局变量存在无法转换为数值的取值")
        classes, class_counts = None, None

    categorical = [f for f in features if variable_types.get(f) == "categorical"]
    numeric = [f for f in features if f not in categorical]

    # ---- 宽表：0/1 二值列按数值参与建模 ----
    # 0/1 指纹位、二值标志列会被类型推断判成“数字编码分类”，宽表里上千列走 OneHot
    # 后训练慢一个量级，且宽表筛选（只打分数值列）对它们完全无效。这里仅在宽表路径
    # 把它们按数值处理：统计页仍显示为分类，建模时用 0/1 原值，省掉独热与上千万次
    # 类别比较。窄表保持原语义（性别 0/1 仍按分类）。
    binary_as_numeric = []
    if len(features) > SCREEN_TRIGGER_FEATURES:
        for f in list(categorical):
            vals = pd.to_numeric(data[f], errors="coerce").dropna()
            if vals.empty:
                continue
            uniq = np.unique(vals.to_numpy(dtype=float))
            if uniq.size <= 2 and set(uniq.tolist()) <= {0.0, 1.0}:
                data[f] = pd.to_numeric(data[f], errors="coerce")
                binary_as_numeric.append(f)
        if binary_as_numeric:
            categorical = [f for f in categorical if f not in binary_as_numeric]
            numeric = numeric + binary_as_numeric

    # ---- 样本量红线 ----
    warnings_ = []
    if leak_dropped:
        warnings_.append(
            f"检测到 {len(leak_dropped)} 个结构性泄露列（单变量判别力 > {LEAK_AUTO_DROP_THRESHOLD}），"
            "已自动从特征中剔除：" + "、".join(leak_dropped[:8]) +
            ("等" if len(leak_dropped) > 8 else "") + "。请确认这些列是否由结局派生。"
        )
    if leak_warned:
        warnings_.append(
            f"检测到 {len(leak_warned)} 个特征单变量判别力 > {LEAK_WARN_THRESHOLD}"
            f"（未达自动剔除线 {LEAK_AUTO_DROP_THRESHOLD}），指标可能虚高，请复核是否为结局派生："
            + "、".join(f"{f.column}({f.metric.upper()}={f.score:.3f})" for f in leak_warned[:8])
            + ("等" if len(leak_warned) > 8 else "") + "。"
        )
    if binary_as_numeric:
        warnings_.append(
            f"宽表优化：{len(binary_as_numeric)} 个 0/1 二值列已按数值参与建模（统计页仍显示为分类），"
            "以避免上千列独热编码带来的额外开销。"
        )
    if task == "classification":
        min_class = min(class_counts.values())
        min_class_name = min(class_counts, key=class_counts.get)
        # 类别值可能是很长的自由文本（报告/备注），提示里截断避免把整句话塞进原因
        min_class_label = str(min_class_name)[:24] + ("…" if len(str(min_class_name)) > 24 else "")
    else:
        min_class, min_class_name, min_class_label = None, None, None
    if n_total < MIN_N_FOR_MODELING:
        warnings_.append(f"样本量 n={n_total} < {MIN_N_FOR_MODELING}，样本量不足，模型结果仅供参考。")

    # ---- 样本量红线：只在“确实无法稳定建模”时跳过，并给出可核对的具体数字 ----
    if n_total < MIN_N_FOR_SPLIT:
        return {"skipped": True, "task": task, "n_total": n_total,
                "reason": (f"样本量严重不足：有效样本 n={n_total} < {MIN_N_FOR_SPLIT}，"
                           "无法稳定切分训练/测试集，请仅参考统计描述与组间比较结果。"),
                "skip_detail": {"n_total": n_total, "min_n": MIN_N_FOR_SPLIT},
                "warnings": warnings_}
    if task == "classification" and min_class < MIN_CLASS_FOR_MODELING:
        return {"skipped": True, "task": task, "n_total": n_total,
                "reason": (f"样本量严重不足：最少类别「{min_class_label}」仅 {min_class} 例"
                           f"（< {MIN_CLASS_FOR_MODELING}），无法进行分层交叉验证，"
                           "请仅参考统计描述与组间比较结果。"),
                "skip_detail": {"n_total": n_total, "min_class": min_class,
                                "min_class_name": min_class_label,
                                "min_class_threshold": MIN_CLASS_FOR_MODELING},
                "warnings": warnings_}
    if task == "classification" and min_class < MIN_CLASS_WARN:
        warnings_.append(
            f"最少类别「{min_class_label}」仅 {min_class} 例（< {MIN_CLASS_WARN}），"
            "交叉验证与 AUC 波动会较大，结果仅供参考。"
        )

    # ---- 大数据集采样 ----
    # 超过阈值时对训练集进行分层采样，测试集保持完整以评估泛化能力
    X_train_full, X_test, y_train_full, y_test = _split(X, y_idx, task)
    
    if len(X_train_full) > LARGE_DATASET_THRESHOLD:
        from sklearn.model_selection import train_test_split
        rng = np.random.default_rng(RANDOM_SEED)
        
        if task == "classification":
            # 分层采样保持类别比例
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full,
                train_size=LARGE_DATASET_TRAIN_SIZE,
                stratify=y_train_full,
                random_state=RANDOM_SEED
            )
        else:
            # 回归任务随机采样
            indices = rng.choice(len(X_train_full), LARGE_DATASET_TRAIN_SIZE, replace=False)
            X_train = X_train_full.iloc[indices]
            y_train = y_train_full[indices]
        
        warnings_.append(
            f"训练集样本量过大（{len(X_train_full)} 行），已采样 {LARGE_DATASET_TRAIN_SIZE} 行用于训练以加速计算。"
            f"测试集保持 {len(X_test)} 行完整数据用于评估。"
        )
    else:
        X_train, y_train = X_train_full, y_train_full
    
    n_train, n_test = int(len(X_train)), int(len(X_test))

    # ---- P1：宽表降维（只在训练集上算分，避免用测试标签做特征选择） ----
    features_used = features
    screening = {"triggered": False}
    if len(features) > SCREEN_TRIGGER_FEATURES:
        sel_num, sel_cat, sinfo = _screen_wide_features(X_train, y_train, numeric, categorical, task)
        cols = sel_num + sel_cat
        if cols and len(cols) < len(features):
            X_train = X_train[cols]
            X_test = X_test[cols]
            numeric, categorical = sel_num, sel_cat
            features_used = cols
            dropped = [f for f in features if f not in cols]
            screening = {"triggered": True, "original": len(features), "selected": len(cols),
                         "dropped": dropped, "dropped_count": len(dropped), **sinfo}
            warnings_.append(
                f"宽表自动筛选：{len(features)} 个特征 → 保留 {len(cols)} 个"
                f"（数值列方差过滤 + 数值/类别列单变量 F 检验 + 共线性 >{SCREEN_CORR_MAX} 去冗，"
                f"合并后统一取 top-{sinfo.get('top_k', SCREEN_TOP_K)}）；"
                f"已剔除 {len(dropped)} 列" + ("（如 " + "、".join(dropped[:5]) + "）" if dropped else "") + "。"
            )

    # ---- EPV 按实际参与建模的特征数计算 ----
    n_features = len(features_used)
    if task == "classification":
        epv = min_class / max(1, n_features)
    else:
        epv = n_total / max(1, n_features)
    if epv < MIN_EPV:
        warnings_.append(f"EPV={epv:.1f} < {MIN_EPV}（{'最少类别事件数' if task == 'classification' else '样本量'}"
                         f"÷特征数），模型结果仅供参考。")

    # ---- 规模自适应：大样本自动降折、减树，缩短串行拟合等待 ----
    fast_mode = n_train > FAST_MODE_ROWS
    folds = FAST_CV_FOLDS if fast_mode else CV_FOLDS
    # 分层交叉验证要求每折至少含 1 个少数类样本；按“训练集”（而非全量）的类别
    # 计数降折，避免切分后少数类只剩 3~4 例却仍用 5 折（会触发 sklearn 警告并让
    # 多分类 AUC 评分失败）。
    if task == "classification":
        counts_train = np.bincount(np.asarray(y_train, dtype=int), minlength=len(classes))
        folds = max(2, min(folds, int(counts_train.min()) if counts_train.size else folds))
    trees = FAST_RF_TREES if fast_mode else RF_TREES
    if fast_mode:
        warnings_.append(
            f"训练集 n={n_train} > {FAST_MODE_ROWS}，已切换快速训练模式：{folds} 折交叉验证，"
            "并用 HistGradientBoosting 替代随机森林以缩短等待时间；"
            "指标口径不变，仍为测试集评估 + 训练折交叉验证。"
        )

    results = []
    fitted = {}
    models_map = _candidate_models(task, trees, fast_mode=fast_mode)
    for key, (label, estimator) in models_map.items():
        pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", estimator)])
        if task == "classification":
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
            # 类别数 >2 时 scoring="roc_auc" 会因缺 multi_class 参数被 sklearn 拒绝，
            # 每一折都记 NaN（仅 warning），NaN 再进响应体会让 JSON 序列化直接 500。
            # roc_auc_ovr 内置 scorer 在二分类与多分类下都成立。
            scoring = "roc_auc" if len(classes) == 2 else "roc_auc_ovr"
            cv_scores, oof_prob = _cv_scores_with_oof(
                pipe, X_train, y_train, cv, scoring, task, len(classes))
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
            cv_scores, oof_prob = _cv_scores_with_oof(
                pipe, X_train, y_train, cv, "r2", task, None)
        cv_scores = np.asarray(cv_scores, dtype=float)
        # 交叉验证若出现 NaN/Inf 不得进入响应体：starlette 序列化器 allow_nan=False 会直接 500
        cv_ok = bool(np.isfinite(cv_scores).all())
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test) if task == "classification" else None
        if task == "classification":
            metrics_ = _classification_metrics(y_test, y_pred, y_prob, range(len(classes)))
            primary = metrics_["roc_auc"]
        else:
            metrics_ = _regression_metrics(y_test, y_pred)
            primary = metrics_["r2"]
        results.append({
            "name": key, "label": label,
            "cv_metric": "roc_auc" if task == "classification" else "r2",
            "cv_mean": round(float(cv_scores.mean()), 4) if cv_ok else None,
            "cv_std": round(float(cv_scores.std()), 4) if cv_ok else None,
            "cv_text": (f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}" if cv_ok
                        else "交叉验证未产出有效结果"),
            "test_metrics": metrics_,
            "primary": primary,
        })
        fitted[key] = {"pipeline": pipe, "y_pred": y_pred, "y_prob": y_prob,
                       "oof_prob": oof_prob}

    # ---- 最优模型：性能接近时优先可解释模型 ----
    interpretable = "logistic" if task == "classification" else "linear"
    best = max(results, key=lambda r: (r["primary"] if r["primary"] is not None else -np.inf))
    interp = next(r for r in results if r["name"] == interpretable)
    if best["name"] != interpretable and \
            abs((best["primary"] or 0) - (interp["primary"] or 0)) <= 0.02:
        reason = (f"{best['label']} 指标略高（{best['primary']} vs {interp['primary']}），"
                  f"但差距 ≤0.02，医疗场景优先可解释模型，故推荐 {interp['label']}。")
        best = interp
    else:
        reason = f"{best['label']} 在测试集上指标最优（{best['primary']}）。"
    best_key = best["name"]
    pipe = fitted[best_key]["pipeline"]

    # 完美 AUC 的显式警示：泄露列已按 >0.99 自动剔除，但小样本仍可能过拟合或存在
    # 未被检出的派生列，不能让用户误以为模型“临床可用”。
    best_r2 = None
    if task == "classification":
        best_auc = best["test_metrics"].get("roc_auc")
        if best_auc is not None and float(best_auc) >= 0.995:
            warnings_.append(
                f"最佳模型测试集 AUC={float(best_auc):.3f} 接近完美（n_test={n_test}），"
                "需排查结局泄露（如诊断派生列）或过拟合；请结合校准曲线与临床合理性复核。"
            )
    else:
        # 回归 R²<0 意味着模型还不如“直接用均值”，必须显式提示而不是把负分当正常结果展示
        best_r2 = best["test_metrics"].get("r2")
        if best_r2 is not None and float(best_r2) < 0:
            warnings_.append(
                f"最佳模型测试集 R²={float(best_r2):.3f} < 0：预测效果不如直接用均值，"
                f"可能因样本量小（n={n_total}）、特征与结局关系弱或过拟合；不建议直接用于预测。"
            )

    # 是否可用于新样本预测：回归 R²<0 的模型等于反预测，前端据此禁用预测入口，
    # 避免用户把“越预测越偏离”的结果当临床参考。
    predictive = not (task == "regression" and best_r2 is not None and float(best_r2) < 0)

    payload = {
        "task": task, "target": target, "features": features_used,
        "numeric_features": numeric, "categorical_features": categorical,
        "classes": classes, "class_counts": class_counts,
        "n_total": n_total, "n_train": n_train, "n_test": n_test,
        "models": results, "best_model": {"name": best_key, "label": best["label"], "reason": reason},
        "warnings": warnings_, "epv": round(epv, 2),
        "predictive": predictive,
        "random_seed": RANDOM_SEED,
        "feature_importance": _feature_importance(pipe, numeric, categorical, task, classes,
                                                   X_test, y_test),
        "speed": {"fast_mode": fast_mode, "cv_folds": folds,
                  "estimator": "hist_gb" if fast_mode else "random_forest",
                  "rf_trees": None if fast_mode else trees,
                  "train_rows": n_train, "sampled": len(X_train) < len(X_train_full)},
        "feature_screening": screening,
        "leak_columns": leak_dropped,
        "leak_warnings": [{"column": f.column, "metric": f.metric, "score": f.score}
                          for f in leak_warned],
    }

    # ---- 分类任务：评估三件套（全部基于测试集）----
    if task == "classification":
        # 概率校准：直接复用上面 CV 的同一批折产生的 OOF 概率拟合校准器，
        # 不再让 CalibratedClassifierCV 额外再拟合一整套折（少 folds 次拟合）。
        # 校准器只用训练集的 OOF 概率拟合，测试集仅做评估。
        n_cal = int(len(X_train))
        method = "isotonic" if n_cal >= ISOTONIC_MIN_N else "sigmoid"
        cal_probs, cal_label = None, None
        oof = fitted[best_key].get("oof_prob")
        if oof is not None and np.isfinite(oof).all():
            try:
                cal_probs, cal_label = _calibrate_oof(
                    oof, y_train, np.asarray(pipe.predict_proba(X_test), dtype=float),
                    method, len(classes), folds)
            except Exception as exc:
                log.warning("OOF 校准失败，退回 CalibratedClassifierCV：%s", exc)
        if cal_probs is None:
            # 兜底：OOF 不完整（极少见）时退回 sklearn 标准校准，保证接口可用
            from sklearn.calibration import CalibratedClassifierCV
            cal_cv = max(2, min(folds, min(class_counts.values())))
            cal_pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)),
                                 ("model", _candidate_models(task, trees,
                                                             fast_mode=fast_mode)[best_key][1])])
            calibrated = CalibratedClassifierCV(cal_pipe, method=method, cv=cal_cv)
            calibrated.fit(X_train, y_train)
            cal_probs = calibrated.predict_proba(X_test)
            cal_label = f"CalibratedClassifierCV({method}, cv={cal_cv})"

        if len(classes) == 2:
            y_test_bin = (y_test == 1).astype(int)
            cal_prob = cal_probs[:, 1]
            payload["roc"] = roc_analysis(y_test_bin, cal_prob, n_test)
            payload["calibration"] = calibration_analysis(y_test_bin, cal_prob, n_test)
            payload["dca"] = decision_curve_analysis(y_test_bin, cal_prob)
        else:
            # 多分类没有单一正类：按 one-vs-rest 为每个类别各算一条 ROC / 校准曲线，
            # DCA 则取各类 OvR 净获益的宏平均（单类净获益仍在 dca.classes 中可见）。
            payload.update(multiclass_evaluation(cal_probs, y_test, classes, n_test))
        payload["calibrated"] = True
        payload["calibration_model"] = cal_label

    # ---- SHAP ----
    if with_shap:
        try:
            payload["shap"] = shap_global(pipe, X_test, task, numeric, categorical)
        except Exception as exc:
            # 不静默吞异常：接口仍可用（前端降级展示），但服务端必须留完整堆栈供排查
            log.warning("SHAP 计算失败（已降级，不影响其他指标）：%s", exc, exc_info=True)
            payload["shap"] = {
                "error": f"SHAP 计算失败：{exc}",
                "error_type": type(exc).__name__,
                "hint": "样本量过小或当前模型不支持该解释器时，可关闭 SHAP 后重新训练",
            }

    payload["_fitted"] = {"pipeline": pipe, "X_train": X_train}
    payload["from_cache"] = False

    # 结果缓存：只存最终产物（含已拟合 pipeline），重复请求秒回；超容量按 LRU 淘汰。
    if cache_fp is not None:
        _TRAIN_CACHE[cache_fp] = payload
        while len(_TRAIN_CACHE) > TRAIN_CACHE_SIZE:
            _TRAIN_CACHE.popitem(last=False)
    return payload


# ------------------------------------------------------------------
# 特征重要性与 SHAP
# ------------------------------------------------------------------

def _transformed_names(pipe, numeric, categorical):
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    mapping = []
    for name in names:
        body = name.split("__", 1)[1]
        if name.startswith("num__"):
            mapping.append((name, body, None))
        else:
            owner = next((c for c in categorical if body.startswith(c + "_")), body)
            level = body[len(owner) + 1:] if body.startswith(owner + "_") else body
            mapping.append((name, owner, level))
    return mapping


def _feature_importance(pipe, numeric, categorical, task, classes,
                        X_eval=None, y_eval=None) -> list:
    """按原始变量聚合的特征重要性。

    树/线性模型用原生 importance/系数；HistGradientBoosting 没有原生重要性，
    退化为测试集上的置换重要性（permutation_importance，按原始列计算，无需哑变量映射）。
    """
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
        mapping = _transformed_names(pipe, numeric, categorical)
        agg = {}
        for i, (_col, owner, _lvl) in enumerate(mapping):
            agg[owner] = agg.get(owner, 0.0) + float(raw[i])
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        raw = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        mapping = _transformed_names(pipe, numeric, categorical)
        agg = {}
        for i, (_col, owner, _lvl) in enumerate(mapping):
            agg[owner] = agg.get(owner, 0.0) + float(raw[i])
    elif X_eval is not None and y_eval is not None and len(X_eval):
        try:
            scoring = "r2" if task == "regression" else (
                "roc_auc" if len(classes or []) == 2 else "roc_auc_ovr")
            pi = permutation_importance(pipe, X_eval, y_eval, scoring=scoring,
                                        n_repeats=3, random_state=RANDOM_SEED, n_jobs=N_JOBS)
            raw = np.clip(np.asarray(pi.importances_mean, dtype=float), 0, None)
            agg = {col: float(raw[i]) for i, col in enumerate(X_eval.columns)}
        except Exception as exc:
            log.warning("置换重要性计算失败（返回空重要性）：%s", exc)
            return []
    else:
        return []
    total = sum(agg.values()) or 1.0
    items = [{"variable": k, "importance": round(v / total, 4), "raw": round(v, 6)}
             for k, v in agg.items()]
    items.sort(key=lambda x: -x["importance"])
    return items


def shap_global(pipe, X: pd.DataFrame, task: str, numeric: list,
                categorical: list, max_points: int = 1000) -> dict:
    """全局 SHAP：树模型 TreeExplainer / 线性模型 LinearExplainer，哑变量按原变量聚合。"""
    import shap

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    if isinstance(model, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        raise MLError("推荐模型为 HistGradientBoosting，暂不支持 SHAP 解释；"
                      "若需要 SHAP 请改用逻辑回归/随机森林，或关闭 SHAP 后重新训练。")
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=float)
    mapping = _transformed_names(pipe, pipe.named_steps["prep"].transformers_[0][2] or [],
                                 pipe.named_steps["prep"].transformers_[1][2] or [])

    # 蜂群图样本量超 1000 时后端采样
    idx = np.arange(Xt.shape[0])
    if Xt.shape[0] > max_points:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(Xt.shape[0], max_points, replace=False)

    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt[idx])
        ev = explainer.expected_value
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        if isinstance(ev, (list, np.ndarray)):
            ev = float(np.ravel(ev)[1]) if np.ravel(ev).size > 1 else float(np.ravel(ev)[0])
        method = "TreeExplainer"
    else:
        explainer = shap.LinearExplainer(model, Xt)
        sv = explainer.shap_values(Xt[idx])
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(np.ravel(ev)[0])
        method = "LinearExplainer"

    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    ev = float(ev)

    # 哑变量按原始变量聚合
    grouped = {}
    for i, (_col, owner, _lvl) in enumerate(mapping):
        grouped.setdefault(owner, []).append(sv[:, i])
    features = []
    for owner, cols in grouped.items():
        summed = np.sum(np.vstack(cols), axis=0)
        features.append({
            "variable": owner,
            "mean_abs_shap": round(float(np.mean(np.abs(summed))), 5),
            "values": [round(float(v), 5) for v in summed],
        })
    features.sort(key=lambda x: -x["mean_abs_shap"])
    return {"method": method, "expected_value": round(ev, 5), "n_sample": int(len(idx)),
            "features": features,
            "note": "独热编码哑变量的 SHAP 值已按原始变量聚合；"
                    f"样本量超过 {max_points} 时后端采样 {max_points} 点绘制蜂群图。"}


# ------------------------------------------------------------------
# 模型持久化与新样本预测
# ------------------------------------------------------------------

def save_model(dataset_id: str, payload: dict) -> dict:
    """保存最优模型 Pipeline、特征清单与数值特征训练集范围；新训练覆盖旧模型。

    注意：不得从 payload 中 pop 训练产物——payload 可能是缓存里的对象，
    pop 后再命中缓存就会丢掉 `_fitted`，导致「模型未训练，无法保存」。
    """
    fitted = payload.get("_fitted")
    if fitted is None:
        raise MLError("模型未训练，无法保存")
    pipe = fitted["pipeline"]
    X_train = fitted["X_train"]
    numeric = payload["numeric_features"]
    ranges = {}
    for col in numeric:
        s = pd.to_numeric(X_train[col], errors="coerce").dropna()
        if not s.empty:
            ranges[col] = [round(float(s.min()), 4), round(float(s.max()), 4)]
    categories = {}
    for col in payload["categorical_features"]:
        categories[col] = sorted(str(v) for v in pd.unique(X_train[col].dropna()))
    # SHAP 背景数据（训练集变换结果，最多 200 行），供 LinearExplainer 作为 masker
    try:
        bg = pipe.named_steps["prep"].transform(X_train.head(200))
        bg = bg.toarray() if hasattr(bg, "toarray") else np.asarray(bg, dtype=float)
    except Exception:
        bg = None

    bundle = {
        "pipeline": pipe,
        "task": payload["task"],
        "target": payload["target"],
        "features": payload["features"],
        "numeric_features": numeric,
        "categorical_features": payload["categorical_features"],
        "shap_background": bg,
        "numeric_ranges": ranges,
        "categories": categories,
        "classes": payload.get("classes"),
        "best_model": payload["best_model"],
        "predictive": payload.get("predictive", True),
        "test_metrics": next(m["test_metrics"] for m in payload["models"]
                             if m["name"] == payload["best_model"]["name"]),
        "n_train": payload["n_train"], "n_test": payload["n_test"],
        "random_seed": RANDOM_SEED,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = _model_path(dataset_id)
    import joblib
    joblib.dump(bundle, path)
    MODEL_STORE[dataset_id] = {"path": path, "bundle": bundle}
    MODEL_DISCARDED.pop(dataset_id, None)      # 新模型可用，旧的“已丢弃”状态失效
    return {"dataset_id": dataset_id, "path": path, "saved_at": bundle["created_at"],
            "features": bundle["features"], "numeric_ranges": ranges, "categories": categories}


def load_model(dataset_id: str):
    """读取数据集对应的模型；不存在则返回 None（前端显示“请先完成当前数据集的建模”）。"""
    entry = MODEL_STORE.get(dataset_id)
    if entry:
        return entry["bundle"]
    try:
        path = _model_path(dataset_id)
    except MLError:
        return None                     # 非法 id 视为无模型，不触碰文件系统
    if os.path.exists(path):
        import joblib
        bundle = joblib.load(path)
        MODEL_STORE[dataset_id] = {"path": path, "bundle": bundle}
        return bundle
    return None


def invalidate_model(dataset_id: str) -> None:
    """上传新数据集后旧模型立即失效（同时清掉该数据集的训练结果缓存）。"""
    MODEL_STORE.pop(dataset_id, None)
    MODEL_DISCARDED.pop(dataset_id, None)
    path = _model_path(dataset_id)
    if os.path.exists(path):
        os.remove(path)
    clear_cache(dataset_id)


def mark_model_discarded(dataset_id: str, payload: dict) -> dict:
    """记录“训练完成但模型被丢弃”的状态（回归 R²<0 不落盘），供预测页展示原因。"""
    r2 = None
    best_name = (payload.get("best_model") or {}).get("name")
    for m in payload.get("models") or []:
        if m.get("name") == best_name:
            r2 = m.get("test_metrics", {}).get("r2")
            break
    rec = {"task": payload.get("task"), "target": payload.get("target"),
           "r2": r2, "n_total": payload.get("n_total"),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "reason": "测试集 R²<0（不如直接用均值），模型不具备预测能力，已丢弃"}
    MODEL_DISCARDED[dataset_id] = rec
    return rec


def discarded_model_info(dataset_id: str):
    """最近一次被丢弃的模型记录；无则返回 None。"""
    return MODEL_DISCARDED.get(dataset_id)


def predict_new_sample(dataset_id: str, sample: dict) -> dict:
    """新样本预测：返回预测类别/数值、概率、SHAP 力图（base64）与外推警示。"""
    bundle = load_model(dataset_id)
    if bundle is None:
        raise MLError("当前数据集尚未完成建模，请先在「分析看板」完成模型训练")
    if bundle.get("predictive") is False:
        raise MLError("该模型测试集 R²<0，预测效果不如直接用均值，不具备预测能力；"
                      "请更换特征列或改用其他模型重新训练")

    features = bundle["features"]
    row = {f: sample.get(f, np.nan) for f in features}
    X_new = pd.DataFrame([row], columns=features)
    for col in bundle["categorical_features"]:
        X_new[col] = X_new[col].astype(object).where(X_new[col].notna(), None)

    warnings_ = []
    for col, (lo, hi) in bundle["numeric_ranges"].items():
        val = pd.to_numeric(X_new[col], errors="coerce").iloc[0]
        if pd.isna(val):
            continue
        if val < lo or val > hi:
            warnings_.append(f"「{col}」输入值 {val} 超出训练集范围 [{lo}, {hi}]，"
                             f"属于外推预测，结果可靠性下降。")
    for col, levels in bundle["categories"].items():
        val = X_new[col].iloc[0]
        if val is not None and str(val) not in levels:
            warnings_.append(f"「{col}」取值「{val}」未在训练集中出现过，将按未知类别处理。")

    pipe = bundle["pipeline"]
    result = {"task": bundle["task"], "target": bundle["target"], "warnings": warnings_}

    if bundle["task"] == "classification":
        proba = pipe.predict_proba(X_new)[0]
        classes = bundle["classes"]
        pred_idx = int(np.argmax(proba))
        result.update({
            "predicted_class": classes[pred_idx],
            "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, proba)},
            "positive_class": classes[1] if len(classes) == 2 else None,
            "positive_probability": round(float(proba[1]), 4) if len(classes) == 2 else None,
        })
    else:
        value = float(pipe.predict(X_new)[0])
        result.update({"predicted_value": round(value, 4)})

    try:
        result["shap_force_plot"] = shap_force_plot_base64(pipe, X_new, bundle)
    except Exception as exc:
        result["shap_force_plot"] = None
        result["shap_error"] = f"SHAP 力图生成失败：{exc}"

    result["disclaimer"] = "科研演示用途，不构成临床诊断依据。"
    return result


def shap_force_plot_base64(pipe, X_new: pd.DataFrame, bundle: dict) -> str:
    """用 matplotlib 生成静态 SHAP 力图并返回 base64 PNG（禁止前端引入 SHAP JS）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字段名需要在力图中正常显示（按系统可用字体依次回退）
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
            if candidate in available:
                plt.rcParams["font.sans-serif"] = [candidate]
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    import shap

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    Xt = prep.transform(X_new)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=float)
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]

    bg = bundle.get("shap_background")
    masker = np.asarray(bg, dtype=float) if bg is not None and np.asarray(bg).size else Xt

    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
        ev = explainer.expected_value
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        if isinstance(ev, (list, np.ndarray)):
            ev = float(np.ravel(ev)[1]) if np.ravel(ev).size > 1 else float(np.ravel(ev)[0])
    else:
        explainer = shap.LinearExplainer(model, masker)
        sv = explainer.shap_values(Xt)
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(np.ravel(ev)[0])

    sv = np.asarray(sv, dtype=float).reshape(-1)
    plt.figure(figsize=(11, 3.2))
    try:
        try:
            shap.plots.force(ev, sv, feature_names=names, matplotlib=True, show=False)
        except TypeError:
            shap.plots.force(ev, sv, feature_names=names, matplotlib=True)
    except Exception:
        # 力图渲染失败时退回条形图，保证前端始终有图可展示
        order = np.argsort(-np.abs(sv))[:12]
        plt.barh([names[i] for i in order][::-1], sv[order][::-1],
                 color=["#C53030" if v < 0 else "#1E5AA8" for v in sv[order][::-1]])
        plt.title("SHAP 贡献（Top 12）")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close("all")
    return base64.b64encode(buf.getvalue()).decode("ascii")
