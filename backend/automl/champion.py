# medical-data-analysis/backend/automl/champion.py
"""模块五：冠军模型深度分析。

全量重训 + 特征重要性 Top15 + SHAP + 自动生成中文解读。
SHAP 兼容性（陷阱清单）：
- 树模型（RF/XGB/LGBM/CatBoost）-> TreeExplainer；
- 线性模型（Logistic/Ridge/Lasso/ElasticNet/LinearSVC/SGD）-> LinearExplainer 或直接标准化系数；
- 其他 -> KernelExplainer 并在报告中预警耗时；
- 独热哑变量的 SHAP 值先按原始特征聚合，再取 Top15。
生存模型：Cox 输出标准化系数（exp(β)=风险比），随机生存森林输出特征重要性，不做 SHAP。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import RANDOM_SEED
from automl.errors import AutoMLError, DataShapeError
from automl.models_registry import MODELS, build_estimator
from automl.racer import (build_preprocessor, _configure_imbalance, _drop_all_missing,
                          fit_pipeline, imbalance_weight_mode)
from automl.types import (ChampionDetail, TASK_CLASSIFICATION, TASK_REGRESSION,
                          TASK_SURVIVAL)

__all__ = ["analyze_champion", "interpret"]


# ------------------------------------------------------------------
# 输出列 -> 原始特征 的索引映射（One-Hot 哑变量按原特征聚合）
# ------------------------------------------------------------------
def _feature_index_map(prep, numeric: list, categorical: list) -> list:
    mapping = list(numeric)
    try:
        oh = prep.named_transformers_["cat"].named_steps["oh"]
        cats = oh.categories_ if hasattr(oh, "categories_") else None
    except Exception:
        cats = None
    for i, c in enumerate(categorical):
        k = len(cats[i]) if cats is not None and i < len(cats) else 1
        mapping.extend([c] * k)
    return mapping


def _aggregate(raw_values, mapping: list) -> dict:
    agg = {}
    for i, owner in enumerate(mapping):
        agg[owner] = agg.get(owner, 0.0) + float(raw_values[i])
    return agg


def _attr_or_none(obj, name):
    """安全读取可选属性：仅对“属性不存在 / 明确未实现”返回 None。

    不能用 hasattr 探测：sksurv 的 RandomSurvivalForest.feature_importances_
    是 NotImplementedError property，hasattr 只吞 AttributeError，异常会直接
    冒到 /api/automl/analyze（生存分析冠军解释阶段 500）。

    这里只捕获 AttributeError 与 NotImplementedError；其他异常（如模型
    内部状态损坏导致的 ValueError/RuntimeError）必须继续向上抛出，
    不能因为“属性可选”就把真实故障静默吞掉。
    """
    try:
        return getattr(obj, name)
    except (AttributeError, NotImplementedError):
        return None


def _importance_values(model):
    for attr in ("feature_importances_", "importance_"):
        v = _attr_or_none(model, attr)
        if v is not None:
            return np.asarray(v, dtype=float)
    c = _attr_or_none(model, "coef_")
    if c is not None:
        c = np.asarray(c, dtype=float)
        return np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)
    return None


def _importance_top(model, mapping, top_k=15) -> list:
    raw = _importance_values(model)
    if raw is None or raw.shape[0] != len(mapping):
        return []
    agg = _aggregate(raw, mapping)
    total = sum(agg.values()) or 1.0
    items = [{"variable": k, "importance": round(v / total, 4), "raw": round(v, 6)}
             for k, v in agg.items()]
    items.sort(key=lambda x: -x["importance"])
    return items[:top_k]


def _shap_top(est, Xt, mapping, top_k=15):
    import shap
    n = Xt.shape[0]
    idx = np.arange(n)
    if n > 1000:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(n, 1000, replace=False)
    Xs = np.asarray(Xt[idx], dtype=float)
    warning = None
    if _attr_or_none(est, "feature_importances_") is not None:
        explainer, method = shap.TreeExplainer(est), "TreeExplainer"
    elif _attr_or_none(est, "coef_") is not None:
        explainer, method = shap.LinearExplainer(est, Xs), "LinearExplainer"
    else:
        bg = shap.sample(Xs, min(50, len(Xs)), random_state=RANDOM_SEED)
        explainer, method = shap.KernelExplainer(est.predict, bg), "KernelExplainer"
        warning = "使用 KernelExplainer，计算较慢，结果仅作参考"
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:
        # shap 0.5x 对分类器返回 (n, 特征, 类别)：按类别轴取绝对均值塌成 (n, 特征)
        sv = np.abs(sv).mean(axis=2)
    if sv.ndim != 2 or sv.shape[1] != len(mapping):
        return {"method": method, "warning": "SHAP 输出形状异常，跳过", "n_sample": int(len(idx)), "features": []}
    mean_abs = np.abs(sv).mean(axis=0)
    agg = _aggregate(mean_abs, mapping)
    items = sorted([{"variable": k, "mean_abs_shap": round(v, 5)} for k, v in agg.items()],
                   key=lambda x: -x["mean_abs_shap"])
    return {"method": method, "warning": warning, "n_sample": int(len(idx)),
            "features": items[:top_k]}


# ------------------------------------------------------------------
# 数据准备（与 racer 一致，但只做一次全量拟合）
# ------------------------------------------------------------------
def _prepare(df, profile):
    features = profile.feature_columns()
    if not features:
        raise DataShapeError("没有可用的特征列")
    X = df[features].copy()
    y_raw = df[profile.target]
    mask = y_raw.notna().to_numpy()
    X, y_raw = X[mask], y_raw[mask]
    col_roles = {c.name: c.role for c in profile.columns}
    numeric = [f for f in features if col_roles.get(f) == "numeric"]
    categorical = [f for f in features if col_roles.get(f) == "categorical"]
    X, dropped, numeric, categorical = _drop_all_missing(X, numeric, categorical)
    if profile.task_type == TASK_CLASSIFICATION:
        classes = sorted(pd.unique(y_raw.astype(str)))
        if len(classes) < 2:
            raise DataShapeError(f"结局变量「{profile.target}」只有单一取值")
        y = np.array([classes.index(str(v)) for v in y_raw], dtype=int)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise DataShapeError(f"回归结局「{profile.target}」存在无法转换为数值的取值")
        classes = None
    return X, y, numeric, categorical, dropped, classes


def _survival_prepare(df, profile):
    from sksurv.util import Surv
    features = profile.feature_columns()
    col_roles = {c.name: c.role for c in profile.columns}
    numeric = [f for f in features if col_roles.get(f) == "numeric"]
    categorical = [f for f in features if col_roles.get(f) == "categorical"]
    X = df[features].copy()
    t = pd.to_numeric(df[profile.time_col], errors="coerce")
    e = pd.to_numeric(df[profile.event_col], errors="coerce")
    mask = t.notna() & e.notna()
    X = X[mask]
    X, dropped, numeric, categorical = _drop_all_missing(X, numeric, categorical)
    y = Surv.from_arrays(event=e[mask].to_numpy(dtype=bool), time=t[mask].to_numpy(dtype=float))
    return X, y, numeric, categorical, dropped


# ------------------------------------------------------------------
# 冠军分析
# ------------------------------------------------------------------
def analyze_champion(df: pd.DataFrame, profile, model_key: str, params: dict | None = None,
                     with_shap: bool = True, top_k: int = 15,
                     metric_name: str | None = None, metric_value: float | None = None) -> ChampionDetail:
    spec = MODELS.get(model_key)
    if spec is None:
        raise AutoMLError(f"未知模型「{model_key}」")

    if profile.task_type == TASK_SURVIVAL:
        X, y, numeric, categorical, dropped = _survival_prepare(df, profile)
        prep = build_preprocessor(numeric, categorical)
        t0 = time.time()
        est = build_estimator(model_key, TASK_SURVIVAL, params)
        pipe = Pipeline([("prep", prep), ("model", est)])
        pipe.fit(X, y)
        model = pipe.named_steps["model"]
        mapping = _feature_index_map(prep, numeric, categorical)
        imp = _importance_top(model, mapping, top_k)
        shap = {"method": "coefficients" if model_key == "cox_ph" else "importance",
                "warning": "生存模型不做 SHAP，输出标准化系数 / 特征重要性", "n_sample": len(X),
                "features": []}
        notes = ([f"已剔除全缺失列：{'、'.join(dropped)}"] if dropped else [])
        if model_key == "random_survival_forest" and not imp:
            notes.append("随机生存森林不提供内置特征重要性，本次跳过特征归因。")
        detail = ChampionDetail(model_key, spec.label, dict(spec.params), metric_name,
                                metric_value, imp, shap, "", time.time() - t0, notes)
        detail.interpretation = interpret(detail, profile)
        return detail

    X, y, numeric, categorical, dropped, classes = _prepare(df, profile)
    prep = build_preprocessor(numeric, categorical)
    t0 = time.time()
    est = build_estimator(model_key, profile.task_type, params)
    n_classes = len(classes) if classes is not None else None
    est, imb_note = _configure_imbalance(est, model_key, profile, n_classes)
    weighted = imbalance_weight_mode(model_key, profile, n_classes)
    if weighted and imb_note is None:
        imb_note = (f"少数类占比 {profile.imbalance_ratio:.0%} < 10%，已改用平衡样本权重"
                    "（XGBoost 多分类不支持 scale_pos_weight）")
    pipe = Pipeline([("prep", prep), ("model", est)])
    fit_pipeline(pipe, X, y, weighted)
    model = pipe.named_steps["model"]
    mapping = _feature_index_map(prep, numeric, categorical)
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=float)

    imp = _importance_top(model, mapping, top_k)
    shap = _shap_top(model, Xt, mapping, top_k) if with_shap else None
    notes = []
    if dropped:
        notes.append(f"已剔除全缺失列：{'、'.join(dropped)}")
    if imb_note:
        notes.append(imb_note)
    if shap and shap.get("warning"):
        notes.append(shap["warning"])

    detail = ChampionDetail(model_key, spec.label, dict(spec.params), metric_name,
                            metric_value, imp, shap, "", time.time() - t0, notes)
    detail.interpretation = interpret(detail, profile)
    return detail


def interpret(champion: ChampionDetail, profile) -> str:
    """自动生成中文解读：模型为何胜出 + 关键特征 + 局限性提示。"""
    task = profile.task_type
    top = [t["variable"] for t in champion.importance_top[:5]]
    feat_txt = "、".join(top) if top else "暂无可解释特征"

    metric_desc = ""
    if champion.metric_name and champion.metric_value is not None:
        metric_desc = f"（{champion.metric_name}={champion.metric_value}）"

    if task == TASK_CLASSIFICATION:
        head = f"推荐分类模型「{champion.label}」在交叉验证中以 {metric_desc} 胜出。"
    elif task == TASK_REGRESSION:
        head = f"推荐回归模型「{champion.label}」在交叉验证中以 {metric_desc} 胜出。"
    else:
        head = f"推荐生存模型「{champion.label}」在交叉验证中以 {metric_desc} 胜出。"

    shap_note = ""
    if champion.shap and champion.shap.get("method") in ("TreeExplainer", "LinearExplainer"):
        shap_note = f"SHAP 采用 {champion.shap['method']}。"
    elif champion.shap:
        shap_note = f"特征归因方式：{champion.shap['method']}。"

    body = f"影响最大的特征依次为：{feat_txt}。" if top else ""
    return f"{head}{shap_note}{body}本结果仅供研究参考，不能替代临床医生判断。"
