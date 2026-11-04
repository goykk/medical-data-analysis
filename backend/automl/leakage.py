# medical-data-analysis/backend/automl/leakage.py
"""模块二：泄露列检测与处置。

单特征对标签的判别力超过阈值即视为泄露（如目标列的重复副本、由结局反推出的列）。
- 二分类：单特征 AUC；
- 多分类：归一化互信息 NMI（AUC 在多分类下对一维得分无定义，NMI 是判别力的正确类比）；
- 回归：|Spearman| 相关系数；
- 生存：|C-index − 0.5| × 2。
检测在**画像之后、规则路由与赛马之前**执行；处置完成前不得进入赛马。
排除对象：目标列自身、随访时间/事件列、患者 ID 列、日期/文本/常量列。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

from automl.types import (LeakageFinding, TASK_CLASSIFICATION, TASK_CLUSTERING,
                          TASK_REGRESSION, TASK_SURVIVAL)

__all__ = ["detect_leakage", "apply_leakage_decision", "DEFAULT_LEAKAGE_THRESHOLD"]

log = logging.getLogger(__name__)

DEFAULT_LEAKAGE_THRESHOLD = 0.95


# ------------------------------------------------------------------
# 单列判别力打分
# ------------------------------------------------------------------
def _to_score_series(s: pd.Series):
    """数值列返回 float 序列；分类/文本列 factorize 成编码；日期列返回 None（跳过）。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return None
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    # 非数值：编码为整数（类别/文本）；全 NaN 或常量由调用方跳过。
    # factorize 把缺失值编码成 -1，必须还原成 NaN，否则缺失会被当成一个真实类别参与打分。
    codes = pd.factorize(s)[0].astype(float)
    return np.where(pd.isna(s).to_numpy(), np.nan, codes)


def _auc_binary(y, feat):
    mask = ~np.isnan(feat) & ~np.isnan(y)
    if mask.sum() < 2:
        return None
    yb = y[mask].astype(int)
    f = feat[mask]
    if len(np.unique(yb)) < 2 or len(np.unique(f)) < 2:
        return None
    auc = float(roc_auc_score(yb, f))
    return max(auc, 1.0 - auc)


def _nmi_classification(y, feat):
    """多分类判别力：归一化互信息（完美副本 NMI=1.0）。"""
    mask = ~np.isnan(feat) & ~np.isnan(y)
    if mask.sum() < 2:
        return None
    ys = y[mask]
    f = feat[mask]
    if len(np.unique(ys)) < 2 or len(np.unique(f)) < 2:
        return None
    if len(np.unique(f)) > 50:
        # 连续特征分箱，避免 NMI 因取值过多而虚高
        try:
            f = pd.qcut(pd.Series(f), q=20, duplicates="drop", labels=False).to_numpy()
        except Exception:
            f = pd.cut(pd.Series(f), bins=20, labels=False).to_numpy()
    # average_method="min"：泄露检测关心的是"特征是否决定标签"。
    # arithmetic 平均在基数不等时会系统性压低（4 水平特征完全决定二分类结局仅得 0.67），
    # min 归一（MI/min(H)）在"标签被完全决定"时恒为 1，与基数无关。
    return float(normalized_mutual_info_score(ys, f, average_method="min"))


def _score_survival(df, feat, time_col, event_col):
    from sksurv.metrics import concordance_index_censored
    ev = pd.to_numeric(df[event_col], errors="coerce").to_numpy(dtype=float)
    tt = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(feat) & ~np.isnan(ev) & ~np.isnan(tt)
    if mask.sum() < 3:
        return None
    c = concordance_index_censored(ev[mask].astype(bool), tt[mask], feat[mask])[0]
    return abs(float(c) - 0.5) * 2.0


# ------------------------------------------------------------------
# 主检测
# ------------------------------------------------------------------
def detect_leakage(df: pd.DataFrame, target: str | None = None,
                   task: str = TASK_CLASSIFICATION,
                   time_col: str | None = None, event_col: str | None = None,
                   patient_id_col: str | None = None,
                   threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
                   standard_refs: dict | None = None) -> list:
    """检测泄露列。聚类任务无标签，返回空列表。"""
    if df is None or len(df.columns) == 0 or len(df) == 0:
        return []
    if task == TASK_CLUSTERING:
        return []

    if task == TASK_SURVIVAL:
        if not time_col or not event_col:
            return []
        y = None
        y_raw = None
    else:
        if not target:
            return []
        if target not in df.columns:
            return []
        y_raw = df[target]
        if task == TASK_CLASSIFICATION:
            if pd.api.types.is_numeric_dtype(y_raw):
                # 数值编码标签直接用原值，避免 factorize 重排破坏与特征的对应
                y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
            else:
                s = pd.Series(pd.factorize(y_raw)[0], index=y_raw.index).astype(float)
                s[y_raw.isna()] = np.nan
                y = s.to_numpy(dtype=float)
        elif task == TASK_REGRESSION:
            y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        else:
            return []

    excluded = {target, time_col, event_col, patient_id_col}
    refs = standard_refs or {}

    findings = []
    skipped_errors = []
    for col in df.columns:
        if col in excluded or col is None:
            continue
        try:
            feat = _to_score_series(df[col])
            if feat is None:
                continue
            feat = np.where(np.isinf(feat), np.nan, feat)   # inf 按缺失处理，防下游崩潰

            if task == TASK_CLASSIFICATION:
                # 类别特征（factorize 编码）水平数 >2 时 AUC 对任意编码顺序无意义，
                # 完全决定结局的类别列可能因此漏网——此类列一律改用 NMI（顺序无关）
                feat_is_categorical = not pd.api.types.is_numeric_dtype(df[col]) \
                    and not pd.api.types.is_bool_dtype(df[col])
                feat_n_levels = len(np.unique(feat[~np.isnan(feat)]))
                # 结局水平数必须排除缺失值后再判断：nunique() 会把 NaN 当成一个独立取值，
                # 含缺失的二分类结局会被误判为多分类而错用 NMI
                n_y_levels = int(pd.Series(y).dropna().nunique())
                if n_y_levels <= 2 and not (feat_is_categorical and feat_n_levels > 2):
                    score = _auc_binary(y, feat)
                    metric = "auc"
                else:
                    score = _nmi_classification(y, feat)
                    metric = "nmi"
            elif task == TASK_REGRESSION:
                try:
                    from scipy.stats import spearmanr
                    mask = ~np.isnan(feat) & ~np.isnan(y)
                    if mask.sum() < 3 or len(np.unique(feat[mask])) < 2:
                        score = None
                    else:
                        score = abs(float(spearmanr(feat[mask], y[mask]).statistic))
                except Exception:
                    score = None
                metric = "spearman"
            else:  # survival
                score = _score_survival(df, feat, time_col, event_col)
                metric = "cindex"
        except Exception as exc:
            # 单列打分失败不得中断整体检测：记录后继续，绝不把整轮分析打成 500
            skipped_errors.append(f"{col}：{str(exc)[:80]}")
            continue

        if score is None or not np.isfinite(score):
            continue
        verdict = "leak" if score > threshold else "ok"
        note = f"{metric.upper()}={score:.3f} > threshold {threshold}" if verdict == "leak" else f"{metric.upper()}={score:.3f} ≤ threshold {threshold}"
        findings.append(LeakageFinding(
            column=col, ref=refs.get(col), score=round(float(score), 4),
            threshold=threshold, metric=metric, verdict=verdict, note=note))

    findings.sort(key=lambda f: -f.score)
    if skipped_errors:
        # 单列失败不打断整体检测，也不混入 findings 污染报告；记日志供排查
        log.warning("泄露检测跳过 %d 列（打分失败）：%s", len(skipped_errors),
                    "；".join(skipped_errors))
    return findings


def apply_leakage_decision(profile, findings: list, decision: str = "ask",
                           user_confirmed_drop: list | None = None) -> dict:
    """按用户决定处置泄露列。返回 {dropped, kept, note}。

    decision：
    - "drop"：全部剔除；
    - "keep"：全部保留（显式确认，计入报告风险）；
    - "ask"：用户给出 user_confirmed_drop 则按此剔除，未给出（跳过确认）默认全部剔除。
    """
    leak_cols = [f.column for f in findings if f.verdict == "leak"]
    if decision == "keep":
        return {"dropped": [], "kept": leak_cols,
                "note": "用户选择保留泄露列（已显式确认），相关指标可能虚高，已计入报告风险提示"}
    if decision == "drop":
        return {"dropped": leak_cols, "kept": [],
                "note": f"泄露列已全部剔除（{len(leak_cols)} 列）"}
    # ask
    if user_confirmed_drop is not None:
        drop_set = set(user_confirmed_drop)
        dropped = [c for c in leak_cols if c in drop_set]
        kept = [c for c in leak_cols if c not in drop_set]
        note = f"用户确认剔除 {len(dropped)} 列" + (f"，保留 {len(kept)} 列" if kept else "")
        return {"dropped": dropped, "kept": kept, "note": note}
    return {"dropped": leak_cols, "kept": [],
            "note": "用户跳过确认，泄露列默认全部剔除（此策略已在报告中显式说明）"}
