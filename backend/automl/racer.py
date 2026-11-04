# medical-data-analysis/backend/automl/racer.py
"""模块四：交叉验证赛马。

分类/回归：5 折交叉验证，指标按任务匹配（分类 AUC/AUPRC，回归 RMSE/MAE/R²）；
聚类：全量拟合，轮廓系数 + CH 指数，自动选 k，DBSCAN 标注无新样本预测能力；
生存：5 折，C-index。
医疗安全（逐条落实）：
- 患者级划分：GroupKFold，同一患者记录绝不跨折；
- SMOTE/类权重仅在训练折内生效（imblearn.Pipeline / estimator 层类权重）；
- 字符串标签先编码；单一取值抛 DataShapeError；
- 数值中位数插补、类别"缺失"作独立类；全缺失列自动剔除并记录；
- 少数类 <10% 自动启用类权重（或折内 SMOTE）；
- 统一 random_state 可复现；单模型失败捕获记录、剔除后继续。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             calinski_harabasz_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     StratifiedShuffleSplit)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_SEED
from automl.errors import AutoMLError, DataShapeError, ModelFitError
from automl.models_registry import MODELS, build_estimator
from automl.types import (RaceRow, TASK_CLASSIFICATION, TASK_CLUSTERING,
                          TASK_REGRESSION, TASK_SURVIVAL)

__all__ = ["build_cv", "make_sampler", "race_models",
           "evaluate_clustering", "evaluate_survival", "build_preprocessor"]

DEFAULT_SAMPLE_THRESHOLD = 100_000


# ------------------------------------------------------------------
# 预处理
# ------------------------------------------------------------------
def _categorical_to_str(X):
    """类别列统一转字符串，并把缺失标为「缺失」独立类。

    必须先转字符串：数值编码的类别列（如 0/1 性别、1/2/3 分级）是 int64，
    SimpleImputer 用字符串填充会抛 ValueError（sklearn 1.8 起强制 dtype 一致）。
    """
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.where(pd.isna(arr), "缺失", arr)
    return np.asarray(out, dtype=str)


def build_preprocessor(numeric: list, categorical: list) -> ColumnTransformer:
    """数值中位数插补 + 标准化；类别列（含数值编码类别）转字符串、缺失作独立类、独热。
    全部封装在 Pipeline 内，只在训练折上拟合（防数据泄漏）。"""
    from sklearn.preprocessing import FunctionTransformer

    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat = Pipeline([
        ("to_str", FunctionTransformer(_categorical_to_str, feature_names_out="one-to-one")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)],
                             remainder="drop")


def _drop_all_missing(df: pd.DataFrame, numeric: list, categorical: list):
    """剔除全缺失列（否则中位数插补后仍为 NaN，StandardScaler 产出 NaN）。"""
    dropped = [c for c in df.columns if df[c].isna().all()]
    if dropped:
        df = df.drop(columns=dropped)
        numeric = [c for c in numeric if c not in dropped]
        categorical = [c for c in categorical if c not in dropped]
    return df, dropped, numeric, categorical


# ------------------------------------------------------------------
# 交叉验证划分
# ------------------------------------------------------------------
def build_cv(profile, n_splits: int = 5, patient_id_col: str | None = None):
    """患者 ID 存在 -> GroupKFold（同一患者不跨折）；否则分类/生存分层，回归 KFold。"""
    if patient_id_col:
        return GroupKFold(n_splits=n_splits)
    if profile.task_type in (TASK_CLASSIFICATION, TASK_SURVIVAL):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


def _folds(cv, X, stratify, groups):
    if groups is not None:
        return list(cv.split(X, stratify, groups=groups))
    return list(cv.split(X, stratify))


def make_sampler(profile, strategy: str = "auto"):
    """返回折内采样器。仅分类任务且少数类<10%时启用；'smote' 显式用 SMOTE，
    'class_weight'/'auto' 用 estimator 层类权重（见 _configure_imbalance），'none' 关闭。
    返回的 SMOTE 必须放进 imblearn.Pipeline，严禁在划分前对全量数据过采样。"""
    if profile.task_type != TASK_CLASSIFICATION:
        return None
    if not profile.imbalance_ratio or profile.imbalance_ratio >= 0.1:
        return None
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    return None


def imbalance_weight_mode(key: str, profile, n_classes: int | None,
                         sampler=None) -> bool:
    """是否需要逐样本平衡权重（仅 XGBoost 多分类）。

    XGBoost 3.x 的 sklearn 接口既没有 `class_weight`，`scale_pos_weight` 又只对
    二分类生效，多分类的不平衡只能通过 fit 时的 `sample_weight` 补偿。
    折内已有 SMOTE 时不叠加权重，避免双重补偿。
    """
    if profile.task_type != TASK_CLASSIFICATION or key != "xgboost":
        return False
    if sampler is not None:
        return False
    ratio = profile.imbalance_ratio
    if not ratio or ratio >= 0.1:
        return False
    return n_classes is not None and n_classes > 2


def fit_pipeline(pipe, X, y, weighted: bool = False):
    """统一 fit 入口：加权时把平衡样本权重经 Pipeline 传给最终估计器。"""
    if weighted:
        from sklearn.utils.class_weight import compute_sample_weight
        return pipe.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    return pipe.fit(X, y)


def _configure_imbalance(est, key: str, profile, n_classes: int | None = None):
    """少数类<10% 时自动启用类权重（estimator 层，天然只在训练折内生效）。"""
    if profile.task_type != TASK_CLASSIFICATION:
        return est, None
    ratio = profile.imbalance_ratio
    if not ratio or ratio >= 0.1:
        return est, None
    scale = (1.0 - ratio) / ratio
    note = f"少数类占比 {ratio:.0%} < 10%，已自动启用类权重"
    try:
        if key in ("logistic", "lasso", "elastic_net", "linear_svm"):
            est.set_params(class_weight="balanced")
        elif key == "random_forest":
            est.set_params(class_weight="balanced_subsample")
        elif key == "xgboost":
            # scale_pos_weight 只在二分类下有定义：多分类时 XGBoost 会直接拒绝该参数，
            # 强行设置会让整个候选模型训练失败
            if n_classes is not None and n_classes != 2:
                return est, None
            est.set_params(scale_pos_weight=scale)
        elif key == "lightgbm":
            est.set_params(class_weight="balanced")
        elif key == "catboost":
            est.set_params(auto_class_weights="Balanced")
        else:
            return est, None
        return est, note
    except Exception:
        return est, None


# ------------------------------------------------------------------
# 采样（>10 万行先抽样赛马）
# ------------------------------------------------------------------
def _sample_for_race(X, y, task, threshold, groups=None):
    """行数超过阈值时先抽样赛马（分组向量同步裁剪）。返回 (X, y, groups, note)。"""
    n = len(X)
    if n <= threshold:
        return X, y, groups, None
    note = f"样本量 {n} > {threshold}，先抽样赛马"
    try:
        if task == TASK_CLASSIFICATION:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=threshold, random_state=RANDOM_SEED)
            sel, _ = next(sss.split(X, y))
            note += "（按标签分层抽样）"
        else:
            bins = pd.qcut(pd.Series(y), q=10, duplicates="drop", labels=False).to_numpy()
            sss = StratifiedShuffleSplit(n_splits=1, train_size=threshold, random_state=RANDOM_SEED)
            sel, _ = next(sss.split(X, bins))
            note += "（按标签分位数分箱后分层抽样）"
    except Exception:
        rng = np.random.default_rng(RANDOM_SEED)
        sel = rng.choice(n, min(threshold, n), replace=False)
        note += "（分层不可行，退化为随机抽样）"
    sel = np.asarray(sel, dtype=int)
    return X.iloc[sel], y[sel], (groups[sel] if groups is not None else None), note


# ------------------------------------------------------------------
# 打分
# ------------------------------------------------------------------
def _ci(scores):
    arr = np.asarray(scores, dtype=float)
    if arr.size < 2 or np.isnan(arr).any():
        return None, None
    m, sd = arr.mean(), arr.std(ddof=1)
    se = sd / np.sqrt(arr.size)
    return round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)


def _score_matrix(model, X):
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    return np.asarray(model.decision_function(X))


def _class_metrics(model, X, y, n_classes):
    sm = _score_matrix(model, X)
    if n_classes == 2:
        s = sm[:, 1] if sm.ndim == 2 else sm
        auc = roc_auc_score(y, s)
        auprc = average_precision_score(y, s)
    else:
        auc = roc_auc_score(y, sm, multi_class="ovr", average="macro")
        # 显式 label-binarize 后再算宏平均 AUPRC（sklearn<1.5 不会自动二值化，会直接抛错；
        # 1.5+ 的自动处理与此等价，数值已实测一致）
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y)
        y_bin = label_binarize(y, classes=classes)
        if sm.ndim == 2 and sm.shape[1] == y_bin.shape[1]:
            auprc = average_precision_score(y_bin, sm, average="macro")
        else:
            auprc = None        # 折内缺类导致列数对不上时跳过，不阻塞整榜
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    # labels 固定为全部类别：折内缺类时矩阵形状仍为 n_classes×n_classes，便于逐折求均值
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    return auc, auprc, acc, cm


def _regression_metrics(model, X, y):
    pred = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    return rmse, mae, r2


# ------------------------------------------------------------------
# 有监督赛马（分类 / 回归）
# ------------------------------------------------------------------
def _race_supervised(df, profile, candidate_keys, patient_id_col, n_splits,
                     sample_threshold, imbalance_strategy, params: dict | None = None):
    target = profile.target
    features = profile.feature_columns()
    if not features:
        raise DataShapeError("没有可用的特征列", suggestion="请检查列类型判定与排除结果")

    X = df[features].copy()
    y_raw = df[target]
    mask = y_raw.notna().to_numpy()
    X, y_raw = X[mask], y_raw[mask]
    if patient_id_col:
        groups = df.loc[mask, patient_id_col].to_numpy()
    else:
        groups = None

    col_roles = {c.name: c.role for c in profile.columns}
    numeric = [f for f in features if col_roles.get(f) == "numeric"]
    categorical = [f for f in features if col_roles.get(f) == "categorical"]

    if profile.task_type == TASK_CLASSIFICATION:
        classes = sorted(pd.unique(y_raw.astype(str)))
        if len(classes) < 2:
            raise DataShapeError(f"结局变量「{target}」只有单一取值，无法建模")
        y = np.array([classes.index(str(v)) for v in y_raw], dtype=int)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise DataShapeError(f"回归结局「{target}」存在无法转换为数值的取值")
        classes = None

    X, dropped, numeric, categorical = _drop_all_missing(X, numeric, categorical)
    X, y, groups, sampling_note = _sample_for_race(X, y, profile.task_type, sample_threshold, groups)

    sampler = make_sampler(profile, strategy="smote" if imbalance_strategy == "smote" else "auto")
    cv = build_cv(profile, n_splits, patient_id_col)
    prep = build_preprocessor(numeric, categorical)

    rows = []
    for key in candidate_keys:
        spec = MODELS.get(key)
        label = spec.label if spec else key
        if spec is None or profile.task_type not in spec.task_types:
            rows.append(RaceRow(key, label, "", 0.0, status="failed",
                                error_message="模型不适配当前任务"))
            continue
        try:
            est = build_estimator(key, profile.task_type, params)
            est, imb_note = _configure_imbalance(
                est, key, profile, len(classes) if classes is not None else None)
            if sampler is not None:
                from imblearn.pipeline import Pipeline as ImbPipeline
                pipe = ImbPipeline([("prep", prep), ("sampler", sampler), ("model", est)])
            else:
                pipe = Pipeline([("prep", prep), ("model", est)])

            weighted = imbalance_weight_mode(
                key, profile, len(classes) if classes is not None else None, sampler)
            if weighted and imb_note is None:
                imb_note = (f"少数类占比 {profile.imbalance_ratio:.0%} < 10%，已改用平衡样本权重"
                            "（XGBoost 多分类不支持 scale_pos_weight）")

            folds = _folds(cv, X, y, groups)
            aucs, sec_auc, sec_acc = [], [], []
            fold_cms = []
            t0 = time.time()
            for tr, te in folds:
                fit_pipeline(pipe, X.iloc[tr], y[tr], weighted)
                if profile.task_type == TASK_CLASSIFICATION:
                    a, p, ac, cm = _class_metrics(pipe, X.iloc[te], y[te], len(classes))
                    aucs.append(a); sec_auc.append(p); sec_acc.append(ac)
                    fold_cms.append(cm)
                else:
                    rmse, mae, r2 = _regression_metrics(pipe, X.iloc[te], y[te])
                    aucs.append(rmse); sec_auc.append(mae); sec_acc.append(r2)
            seconds = time.time() - t0

            if profile.task_type == TASK_CLASSIFICATION:
                primary, name = float(np.mean(aucs)), "auc"
                hi_better = True
                auprc_vals = [v for v in sec_auc if v is not None]
                extra = {"auprc": round(float(np.mean(auprc_vals)), 4) if auprc_vals else None,
                         "accuracy": round(float(np.mean(sec_acc)), 4),
                         # 医疗场景用户要看假阴性/假阳性，而非只有聚合 AUC：
                         # 逐折矩阵 + 均值矩阵（行=真实、列=预测，标签顺序同 classes）
                         "classes": [str(c) for c in classes],
                         "confusion_matrices": [c.tolist() for c in fold_cms],
                         "confusion_matrix_mean": np.round(np.mean(fold_cms, axis=0), 2).tolist()}
            else:
                primary, name = float(np.mean(aucs)), "rmse"
                hi_better = False
                extra = {"mae": round(float(np.mean(sec_auc)), 4),
                         "r2": round(float(np.mean(sec_acc)), 4)}
            lo, hi = _ci(aucs)
            notes = []
            if sampling_note:
                notes.append(sampling_note)
            if imb_note:
                notes.append(imb_note)
            if dropped:
                notes.append(f"已剔除全缺失列：{'、'.join(dropped)}")
            rows.append(RaceRow(key, label, name, round(primary, 4), lo, hi,
                                seconds, "ok", metrics=extra,
                                higher_is_better=hi_better, notes=notes))
        except Exception as exc:   # 单模型失败：记录、剔除、继续
            rows.append(RaceRow(key, label, "", 0.0, status="failed",
                                error_message=str(exc)[:200]))

    ok = [r for r in rows if r.status == "ok"]
    failed = [r for r in rows if r.status != "ok"]
    reverse = ok[0].higher_is_better if ok else True
    ok.sort(key=lambda r: r.metric_value, reverse=reverse)
    return ok + failed


# ------------------------------------------------------------------
# 聚类评估
# ------------------------------------------------------------------
def evaluate_clustering(df, profile, candidate_keys) -> list:
    features = profile.feature_columns()
    if not features:
        raise DataShapeError("没有可用的特征列")
    col_roles = {c.name: c.role for c in profile.columns}
    numeric = [f for f in features if col_roles.get(f) == "numeric"]
    categorical = [f for f in features if col_roles.get(f) == "categorical"]
    X = df[features].copy()
    X, dropped, numeric, categorical = _drop_all_missing(X, numeric, categorical)
    if len(X) < 4:
        raise DataShapeError("样本量不足，无法聚类", suggestion="聚类至少需要 4 条记录")
    prep = build_preprocessor(numeric, categorical)
    Xt = prep.fit_transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=float)

    rows = []
    k_max = min(10, len(Xt) - 1)
    for key in candidate_keys:
        spec = MODELS.get(key)
        label = spec.label if spec else key
        if spec is None or TASK_CLUSTERING not in spec.task_types:
            rows.append(RaceRow(key, label, "silhouette", 0.0, status="failed",
                                error_message="模型不适配聚类任务"))
            continue
        try:
            t0 = time.time()
            if key == "dbscan":
                est = build_estimator(key, TASK_CLUSTERING)
                labels = est.fit_predict(Xt)
                core = labels[labels != -1]
                if len(np.unique(core)) < 2:
                    raise AutoMLError("DBSCAN 未能分出 ≥2 个簇，请调大 eps 或减小 min_samples")
                sil = silhouette_score(Xt[labels != -1], core, random_state=RANDOM_SEED)
                ch = calinski_harabasz_score(Xt[labels != -1], core)
                notes = ["DBSCAN 无新样本预测能力，仅支持全量评估并在排行榜标注该限制"]
            else:
                param = {"kmeans": "n_clusters", "agglomerative": "n_clusters",
                         "gmm": "n_components"}[key]
                best_k = best_sil = best_ch = None
                for k in range(2, k_max + 1):
                    est = build_estimator(key, TASK_CLUSTERING)
                    est.set_params(**{param: k})
                    lbl = est.fit_predict(Xt)
                    if len(np.unique(lbl)) < 2:
                        continue
                    sil = silhouette_score(Xt, lbl, random_state=RANDOM_SEED)
                    ch = calinski_harabasz_score(Xt, lbl)
                    if best_sil is None or sil > best_sil:
                        best_sil, best_k, best_ch = sil, k, ch
                if best_k is None:
                    raise AutoMLError("未能选出有效簇数 k")
                sil, ch = best_sil, best_ch
                notes = [f"自动选择簇数 k={best_k}"]
            rows.append(RaceRow(key, label, "silhouette", round(float(sil), 4),
                                None, None, time.time() - t0, "ok",
                                metrics={"ch_index": round(float(ch), 4)},
                                notes=notes))
        except Exception as exc:
            rows.append(RaceRow(key, label, "silhouette", 0.0, status="failed",
                                error_message=str(exc)[:200]))

    ok = [r for r in rows if r.status == "ok"]
    failed = [r for r in rows if r.status != "ok"]
    ok.sort(key=lambda r: r.metric_value, reverse=True)
    return ok + failed


# ------------------------------------------------------------------
# 生存评估
# ------------------------------------------------------------------
def evaluate_survival(df, profile, candidate_keys, patient_id_col=None, n_splits=5) -> list:
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    features = profile.feature_columns()
    if not features:
        raise DataShapeError("没有可用的特征列")
    col_roles = {c.name: c.role for c in profile.columns}
    numeric = [f for f in features if col_roles.get(f) == "numeric"]
    categorical = [f for f in features if col_roles.get(f) == "categorical"]

    time_col, event_col = profile.time_col, profile.event_col
    X = df[features].copy()
    t = pd.to_numeric(df[time_col], errors="coerce")
    e = pd.to_numeric(df[event_col], errors="coerce")
    mask = t.notna() & e.notna()
    X = X[mask]
    t = t[mask].astype(float)
    e = e[mask].astype(int)
    X, dropped, numeric, categorical = _drop_all_missing(X, numeric, categorical)
    groups = df.loc[mask, patient_id_col].to_numpy() if patient_id_col else None

    y = Surv.from_arrays(event=e.to_numpy(dtype=bool), time=t.to_numpy(dtype=float))
    prep = build_preprocessor(numeric, categorical)
    cv = build_cv(profile, n_splits, patient_id_col)

    rows = []
    for key in candidate_keys:
        spec = MODELS.get(key)
        label = spec.label if spec else key
        if spec is None or TASK_SURVIVAL not in spec.task_types:
            rows.append(RaceRow(key, label, "cindex", 0.0, status="failed",
                                error_message="模型不适配生存分析任务"))
            continue
        try:
            est = build_estimator(key, TASK_SURVIVAL)
            pipe = Pipeline([("prep", prep), ("model", est)])
            scores, t0 = [], time.time()
            ev = e.to_numpy(dtype=bool)
            tt = t.to_numpy(dtype=float)
            for tr, te in _folds(cv, X, e.to_numpy(), groups):
                pipe.fit(X.iloc[tr], y[tr])
                risk = pipe.predict(X.iloc[te])
                scores.append(concordance_index_censored(ev[te], tt[te], risk)[0])
            m = float(np.mean(scores))
            lo, hi = _ci(scores)
            notes = [f"已剔除全缺失列：{'、'.join(dropped)}"] if dropped else []
            rows.append(RaceRow(key, label, "cindex", round(m, 4), lo, hi,
                                time.time() - t0, "ok",
                                metrics={"fold_scores": [round(float(s), 4) for s in scores]},
                                notes=notes))
        except Exception as exc:
            rows.append(RaceRow(key, label, "cindex", 0.0, status="failed",
                                error_message=str(exc)[:200]))

    ok = [r for r in rows if r.status == "ok"]
    failed = [r for r in rows if r.status != "ok"]
    ok.sort(key=lambda r: r.metric_value, reverse=True)
    return ok + failed


# ------------------------------------------------------------------
# 统一入口
# ------------------------------------------------------------------
def race_models(df: pd.DataFrame, profile, candidate_keys: list,
                patient_id_col: str | None = None, n_splits: int = 5,
                sample_threshold: int = DEFAULT_SAMPLE_THRESHOLD,
                imbalance_strategy: str = "class_weight",
                top_n: int | None = None, params: dict | None = None) -> list:
    """按任务分派到对应赛马。返回按主指标排好序的排行榜（ok 在前，failed 在后）。

    params：手动模式单模型训练时的用户超参——仅当候选恰为 1 个模型时生效，
    保证排行榜指标与冠军重训所用参数一致（不这样做会出现报告指标与交付模型不符）。
    """
    if profile.task_type == TASK_CLUSTERING:
        return evaluate_clustering(df, profile, candidate_keys)
    if profile.task_type == TASK_SURVIVAL:
        return evaluate_survival(df, profile, candidate_keys, patient_id_col, n_splits)
    return _race_supervised(df, profile, candidate_keys, patient_id_col, n_splits,
                            sample_threshold, imbalance_strategy,
                            params if len(candidate_keys) == 1 else None)
