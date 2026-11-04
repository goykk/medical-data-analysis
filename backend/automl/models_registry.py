# medical-data-analysis/backend/automl/models_registry.py
"""模型清单：元数据、默认超参、参数说明、SHAP 策略、依赖检查。

自动模式（router）与手动模式共用这一张表，避免两处维护。
所有模型统一：random_state=RANDOM_SEED、并行线程/进程一律 1（见 ml_engine.N_JOBS 的教训）。
"""
from __future__ import annotations

import importlib

from config import RANDOM_SEED
from automl.errors import AutoMLError
from automl.types import (ModelSpec, TASK_CLASSIFICATION, TASK_CLUSTERING,
                          TASK_REGRESSION, TASK_SURVIVAL)

__all__ = ["MODELS", "available_models", "build_estimator", "validate_model_data_fit"]

TASK_LABELS = {
    TASK_CLASSIFICATION: "分类",
    TASK_REGRESSION: "回归",
    TASK_CLUSTERING: "聚类",
    TASK_SURVIVAL: "生存分析",
}

# 依赖 import 名 -> 展示名（requirements 中的包名）
DEP_NAMES = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
             "sksurv": "scikit-survival"}


def _dep(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------
# 注册表
# ------------------------------------------------------------------
MODELS: dict[str, ModelSpec] = {
    # ---------------- 表格分类 ----------------
    "logistic": ModelSpec(
        key="logistic", label="逻辑回归（Logistic Regression）", family="linear",
        task_types=(TASK_CLASSIFICATION,),
        params={"max_iter": 2000, "random_state": RANDOM_SEED},
        param_docs=[{"name": "C", "label": "正则强度倒数", "default": 1.0, "desc": "越大拟合越强，越小越平滑"},
                    {"name": "max_iter", "label": "最大迭代", "default": 2000, "desc": ""}],
        supports_shap="linear",
        shap_note="LinearExplainer 或直接输出标准化系数（OR=exp(β)）",
        notes="医疗场景首选可解释模型",
    ),
    "random_forest": ModelSpec(
        key="random_forest", label="随机森林（Random Forest）", family="tree",
        task_types=(TASK_CLASSIFICATION, TASK_REGRESSION),
        params={"n_estimators": 300, "min_samples_leaf": 2, "random_state": RANDOM_SEED, "n_jobs": 1},
        param_docs=[{"name": "n_estimators", "label": "树数量", "default": 300, "desc": ""},
                    {"name": "min_samples_leaf", "label": "叶最小样本", "default": 2, "desc": "越大越抗过拟合"}],
        supports_shap="tree", shap_note="TreeExplainer",
        notes="",
    ),
    "xgboost": ModelSpec(
        key="xgboost", label="XGBoost", family="tree",
        task_types=(TASK_CLASSIFICATION, TASK_REGRESSION),
        params={"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8, "nthread": 1,
                "random_state": RANDOM_SEED},
        param_docs=[{"name": "max_depth", "label": "树深", "default": 5, "desc": ""},
                    {"name": "learning_rate", "label": "学习率", "default": 0.1, "desc": ""}],
        supports_shap="tree", shap_note="TreeExplainer", requires="xgboost",
    ),
    "lightgbm": ModelSpec(
        key="lightgbm", label="LightGBM", family="tree",
        task_types=(TASK_CLASSIFICATION, TASK_REGRESSION),
        params={"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.1,
                "n_jobs": 1, "random_state": RANDOM_SEED},
        param_docs=[{"name": "num_leaves", "label": "叶节点数", "default": 31, "desc": ""},
                    {"name": "learning_rate", "label": "学习率", "default": 0.1, "desc": ""}],
        supports_shap="tree", shap_note="TreeExplainer", requires="lightgbm",
    ),
    "catboost": ModelSpec(
        key="catboost", label="CatBoost", family="tree",
        task_types=(TASK_CLASSIFICATION, TASK_REGRESSION),
        params={"iterations": 300, "depth": 6, "learning_rate": 0.1,
                "thread_count": 1, "random_state": RANDOM_SEED},
        param_docs=[{"name": "depth", "label": "树深", "default": 6, "desc": ""},
                    {"name": "learning_rate", "label": "学习率", "default": 0.1, "desc": ""}],
        supports_shap="tree", shap_note="TreeExplainer", requires="catboost",
    ),
    # ---------------- 表格回归 ----------------
    "ridge": ModelSpec(
        key="ridge", label="岭回归（Ridge）", family="linear",
        task_types=(TASK_REGRESSION,),
        params={"alpha": 1.0, "random_state": RANDOM_SEED},
        param_docs=[{"name": "alpha", "label": "L2 正则强度", "default": 1.0, "desc": ""}],
        supports_shap="linear", shap_note="LinearExplainer 或标准化系数",
    ),
    # ---------------- 高维小样本（稀疏线性族） ----------------
    "lasso": ModelSpec(
        key="lasso", label="Lasso", family="linear_sparse",
        task_types=(TASK_CLASSIFICATION, TASK_REGRESSION),
        params={"alpha": 1.0, "max_iter": 2000, "random_state": RANDOM_SEED},
        param_docs=[{"name": "alpha", "label": "L1 正则强度", "default": 1.0, "desc": ""}],
        supports_shap="linear", shap_note="LinearExplainer 或标准化系数",
        notes="高维小样本路线：L1 正则天然做特征选择",
    ),
    "elastic_net": ModelSpec(
        key="elastic_net", label="弹性网络（ElasticNet）", family="linear_sparse",
        task_types=(TASK_CLASSIFICATION, TASK_REGRESSION),
        params={"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 2000, "random_state": RANDOM_SEED},
        param_docs=[{"name": "l1_ratio", "label": "L1 占比", "default": 0.5, "desc": "0 纯 L2，1 纯 L1"}],
        supports_shap="linear", shap_note="LinearExplainer 或标准化系数",
        notes="高维小样本路线",
    ),
    "linear_svm": ModelSpec(
        key="linear_svm", label="线性 SVM", family="svm",
        task_types=(TASK_CLASSIFICATION, TASK_REGRESSION),
        params={"max_iter": 5000, "random_state": RANDOM_SEED},
        param_docs=[{"name": "C", "label": "软间隔系数", "default": 1.0, "desc": ""}],
        supports_shap="linear", shap_note="LinearExplainer 或系数",
        notes="高维小样本路线",
    ),
    # ---------------- 无标签聚类 ----------------
    "kmeans": ModelSpec(
        key="kmeans", label="KMeans 聚类", family="cluster",
        task_types=(TASK_CLUSTERING,),
        params={"n_clusters": 3, "n_init": 10, "random_state": RANDOM_SEED},
        param_docs=[{"name": "n_clusters", "label": "簇数 k", "default": "自动选 2..10", "desc": ""}],
        supports_shap="none", shap_note="聚类不输出 SHAP",
        notes="k 由轮廓系数 + CH 指数自动选择",
    ),
    "agglomerative": ModelSpec(
        key="agglomerative", label="层次聚类（Agglomerative）", family="cluster",
        task_types=(TASK_CLUSTERING,),
        params={"n_clusters": 3},
        param_docs=[{"name": "n_clusters", "label": "簇数 k", "default": "自动选 2..10", "desc": ""}],
        supports_shap="none", shap_note="聚类不输出 SHAP",
    ),
    "dbscan": ModelSpec(
        key="dbscan", label="DBSCAN", family="cluster",
        task_types=(TASK_CLUSTERING,),
        params={"eps": 0.5, "min_samples": 5},
        param_docs=[{"name": "eps", "label": "邻域半径", "default": 0.5, "desc": ""}],
        supports_shap="none", shap_note="聚类不输出 SHAP",
        notes="仅支持全量评估，无新样本预测能力（排行榜中标注）",
    ),
    "gmm": ModelSpec(
        key="gmm", label="高斯混合模型（GMM）", family="mixture",
        task_types=(TASK_CLUSTERING,),
        params={"n_components": 3, "random_state": RANDOM_SEED},
        param_docs=[{"name": "n_components", "label": "分量数", "default": "自动选 2..10", "desc": ""}],
        supports_shap="none", shap_note="聚类不输出 SHAP",
    ),
    # ---------------- 生存分析 ----------------
    "cox_ph": ModelSpec(
        key="cox_ph", label="Cox 比例风险模型", family="survival",
        task_types=(TASK_SURVIVAL,),
        params={"alpha": 0.1, "ties": "breslow"},
        param_docs=[{"name": "alpha", "label": "L2 正则强度", "default": 0.1, "desc": ""}],
        supports_shap="none",
        shap_note="输出标准化回归系数（exp(β)=风险比）",
        requires="sksurv",
    ),
    "random_survival_forest": ModelSpec(
        key="random_survival_forest", label="随机生存森林（Random Survival Forest）", family="survival",
        task_types=(TASK_SURVIVAL,),
        params={"n_estimators": 100, "min_samples_leaf": 5, "random_state": RANDOM_SEED, "n_jobs": 1},
        param_docs=[{"name": "n_estimators", "label": "树数量", "default": 100, "desc": ""}],
        supports_shap="tree", shap_note="TreeExplainer（按累积风险函数近似）",
        requires="sksurv",
    ),
}


# ------------------------------------------------------------------
# 可用性与构造
# ------------------------------------------------------------------
def available_models(task: str | None = None, check_deps: bool = True) -> list:
    """可运行模型清单；依赖缺失时标注 available=False + 原因，不静默消失。"""
    out = []
    for spec in MODELS.values():
        if task and task not in spec.task_types:
            continue
        available, reason = True, ""
        if check_deps and spec.requires and not _dep(spec.requires):
            available, reason = False, f"依赖 {DEP_NAMES.get(spec.requires, spec.requires)} 未安装"
        out.append(spec.to_dict(available, reason))
    return out


def validate_model_data_fit(key: str, task: str, profile=None) -> None:
    """模型与数据不适配时抛出带中文原因与建议的 AutoMLError（禁止静默失败）。"""
    spec = MODELS.get(key)
    if spec is None:
        names = "、".join(MODELS.keys())
        raise AutoMLError(f"未知模型「{key}」", suggestion=f"可选模型：{names}")
    if task not in spec.task_types:
        raise AutoMLError(
            f"模型「{spec.label}」不适用于{TASK_LABELS.get(task, task)}任务",
            suggestion=f"该模型仅支持：{'、'.join(TASK_LABELS.get(t, t) for t in spec.task_types)}")
    if spec.requires and not _dep(spec.requires):
        raise AutoMLError(
            f"模型「{spec.label}」依赖 {DEP_NAMES.get(spec.requires, spec.requires)} 未安装",
            suggestion=f"请先安装 {DEP_NAMES.get(spec.requires, spec.requires)} 后重试")


def _construct(key: str, task: str, p: dict):
    """按 (key, task) 构造估计器。lasso/elastic_net/linear_svm 因任务不同而取不同实现。"""
    if key == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**p)
    if key == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**p)
    if key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        return (RandomForestClassifier if task == TASK_CLASSIFICATION else RandomForestRegressor)(**p)
    if key == "xgboost":
        import xgboost as xgb
        if task == TASK_CLASSIFICATION:
            p.setdefault("eval_metric", "logloss"); p.setdefault("verbosity", 0)
            return xgb.XGBClassifier(**p)
        p.setdefault("verbosity", 0)
        return xgb.XGBRegressor(**p)
    if key == "lightgbm":
        import lightgbm as lgb
        p.setdefault("verbose", -1)
        return (lgb.LGBMClassifier if task == TASK_CLASSIFICATION else lgb.LGBMRegressor)(**p)
    if key == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        p.setdefault("verbose", False)
        return (CatBoostClassifier if task == TASK_CLASSIFICATION else CatBoostRegressor)(**p)
    if key == "lasso":
        if task == TASK_CLASSIFICATION:
            from sklearn.linear_model import LogisticRegression
            alpha = float(p.pop("alpha", 1.0))
            # sklearn 1.8 起 penalty="l1" 已弃用：用 l1_ratio=1.0 表达纯 L1
            return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0 / alpha, **p)
        from sklearn.linear_model import Lasso
        return Lasso(**p)
    if key == "elastic_net":
        if task == TASK_CLASSIFICATION:
            from sklearn.linear_model import SGDClassifier
            return SGDClassifier(loss="log_loss", penalty="elasticnet", **p)
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**p)
    if key == "linear_svm":
        if task == TASK_CLASSIFICATION:
            from sklearn.svm import LinearSVC
            return LinearSVC(dual="auto", **p)
        from sklearn.svm import LinearSVR
        return LinearSVR(dual="auto", **p)
    if key == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(**p)
    if key == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(**p)
    if key == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(**p)
    if key == "gmm":
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(**p)
    if key == "cox_ph":
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        return CoxPHSurvivalAnalysis(**p)
    if key == "random_survival_forest":
        from sksurv.ensemble import RandomSurvivalForest
        return RandomSurvivalForest(**p)
    raise AutoMLError(f"未知模型「{key}」")


# 用户可调数值参数的取值范围：防 n_estimators=10**9 这类资源耗尽输入
_PARAM_RANGES = {
    "n_estimators": (1, 5000), "max_depth": (1, 50), "num_leaves": (2, 500),
    "max_iter": (100, 100000), "min_samples_leaf": (1, 1000), "min_samples_split": (2, 1000),
    "learning_rate": (1e-4, 2.0), "C": (1e-4, 1e4),
}

# 安全参数锁定：并行度固定 1（Windows 子进程教训）、随机种子统一，用户不可覆盖
_LOCKED_PARAMS = {"n_jobs", "nthread", "random_state", "thread_count"}


def sanitize_params(key: str, params: dict | None) -> dict:
    """手动模式用户超参白名单过滤。

    仅接受该模型 param_docs/默认参数中声明过的键；数值按 _PARAM_RANGES 钳制；
    n_jobs/random_state 等安全参数拒绝覆盖；非数值原样通过（如 penalty）。
    """
    if not params:
        return {}
    spec = MODELS.get(key)
    if spec is None:
        return {}
    allowed = {d["name"] for d in (spec.param_docs or [])} | set(spec.params.keys())
    out = {}
    for k, v in params.items():
        if k in _LOCKED_PARAMS or k not in allowed:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k in _PARAM_RANGES:
            lo, hi = _PARAM_RANGES[k]
            v = min(max(v, lo), hi)
        out[k] = v
    return out


def build_estimator(key: str, task: str, params: dict | None = None, imbalance: str | None = None):
    """构造估计器；imbalance="balanced" 时对线性/随机森林启用类权重（其余由赛马折内 SMOTE 处理）。"""
    spec = MODELS.get(key)
    if spec is None:
        validate_model_data_fit(key, task)   # 触发未知模型错误
    validate_model_data_fit(key, task)
    p = dict(spec.params)
    if params:
        p.update(sanitize_params(key, params))
    if imbalance == "balanced" and task == TASK_CLASSIFICATION:
        if key in ("logistic", "lasso", "elastic_net"):
            p["class_weight"] = "balanced"
        elif key == "random_forest":
            p["class_weight"] = "balanced_subsample"
    return _construct(key, task, p)
