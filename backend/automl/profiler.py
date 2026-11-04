# medical-data-analysis/backend/automl/profiler.py
"""模块一：数据画像。

输出 DataProfile，供规则路由 / 赛马 / 冠军分析 / 报告消费。要点：
- 列类型识别复用 validators.infer_variable_types（含 ID / 常量 / 日期 / 文本判定）；
- 患者 ID 判定、生存结构判定**改由模块零标准名驱动**（role_hint），
  判定后仍做数值校验（事件列 0/1、随访时间非负、删失比例合法）；
- 任务推断：生存（时间+事件）> 聚类（无结局）> 分类/回归（结局取值形态）。
"""
from __future__ import annotations

import pandas as pd

from validators import infer_variable_types
from automl.errors import DataShapeError, SurvivalDataError
from automl.standardize import StandardizationResult, standardize_dataframe
from automl.types import (DataProfile, ColumnProfile, TASK_CLASSIFICATION,
                          TASK_CLUSTERING, TASK_REGRESSION, TASK_SURVIVAL)

__all__ = [
    "profile_dataframe", "infer_task_type", "detect_survival_pair",
    "validate_survival_inputs", "imbalance_ratio",
]

# 连续结局判定的取值水平上限：数值型且唯一值 ≤ 该值视为分类（编码分类/分级）
MAX_CLASS_LEVELS = 20


# ------------------------------------------------------------------
# 生存结构
# ------------------------------------------------------------------
def detect_survival_pair(df: pd.DataFrame,
                         standard_result: StandardizationResult | None = None) -> tuple[str | None, str | None]:
    """按标准名角色自动识别"随访时间 + 事件状态"列结构。

    返回 (time_col, event_col) 原始列名；任一缺失返回 (None, None)。
    多列命中同一角色时取第一列（在画像 warnings 中提示）。
    """
    std = standard_result or standardize_dataframe(df)
    times = [c.raw_name for c in std.columns
             if c.role_hint == "survival_time" and c.raw_name in df.columns]
    events = [c.raw_name for c in std.columns
              if c.role_hint == "survival_event" and c.raw_name in df.columns]
    return (times[0] if times else None), (events[0] if events else None)


def validate_survival_inputs(df: pd.DataFrame, time_col: str, event_col: str) -> dict:
    """数值校验：事件列 0/1、随访时间非负、删失比例 ∈ (0,1)。违规抛 SurvivalDataError。"""
    for col, label in ((time_col, "随访时间列"), (event_col, "事件列")):
        if col not in df.columns:
            raise SurvivalDataError(f"{label}「{col}」不存在",
                                    suggestion="请确认 time_col / event_col 传入的列名正确")

    t = pd.to_numeric(df[time_col], errors="coerce")
    if t.isna().any():
        raise SurvivalDataError(f"随访时间列「{time_col}」存在无法转换为数值的取值",
                                suggestion="请清理非数值的随访时间记录")
    if (t < 0).any():
        raise SurvivalDataError(f"随访时间列「{time_col}」存在负值",
                                suggestion="随访时间必须非负，请核对数据")

    ev = pd.to_numeric(df[event_col], errors="coerce")
    valid = set(ev.dropna().unique())
    if not valid.issubset({0, 1}):
        raise SurvivalDataError(
            f"事件列「{event_col}」取值必须为 0/1（0=删失，1=事件），当前取值 {sorted(valid)}",
            suggestion="请将事件状态统一为 0/1 编码后再分析")
    if ev.isna().any():
        raise SurvivalDataError(f"事件列「{event_col}」存在缺失",
                                suggestion="事件状态不允许缺失，请补齐或剔除该行")

    n_events = int((ev == 1).sum())
    n_censored = int((ev == 0).sum())
    total = n_events + n_censored
    censoring_rate = n_censored / total if total else 0.0
    if censoring_rate in (0.0, 1.0):
        raise SurvivalDataError(
            f"删失比例为 {censoring_rate:.0%}（事件数 {n_events}，删失数 {n_censored}），C-index 无意义",
            suggestion="生存分析需要同时包含事件与删失样本（建议删失比例 20%~50%）")
    return {"time_col": time_col, "event_col": event_col, "n_events": n_events,
            "n_censored": n_censored, "censoring_rate": round(censoring_rate, 4), "warnings": []}


# ------------------------------------------------------------------
# 任务推断
# ------------------------------------------------------------------
def _target_is_categorical(df: pd.DataFrame, target: str) -> bool:
    """结局是否为分类变量。

    不能只看水平数：把「≤20 个取值的连续测量值」（1 位小数的 BMI、住院天数）
    当分类会丢失信息，并把本该回归的问题做错。判定规则：
    - 非数值列（字符串/类别/布尔）→ 分类；
    - 数值列含小数 → 连续，走回归；
    - 整数型：水平数 ≤5（0/1、分级 1–4）→ 分类；超过 5 级时要求平均每个
      水平至少 10 例（n / n_levels ≥ 10）才视为分级标签，否则按连续变量回归。
    """
    s = df[target]
    if not pd.api.types.is_numeric_dtype(s):
        return True
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty or vals.nunique() < 2:
        return True
    if not bool(((vals - vals.round()).abs() <= 1e-9).all()):
        return False                      # 带小数的连续测量值
    n_levels = int(vals.nunique())
    if n_levels <= 5:
        return True
    return n_levels <= MAX_CLASS_LEVELS and len(vals) / n_levels >= 10


def infer_task_type(df: pd.DataFrame, target: str | None = None,
                    time_col: str | None = None, event_col: str | None = None,
                    standard_result: StandardizationResult | None = None) -> tuple[str, str]:
    """推断任务类型与理由。生存优先，其次无结局=聚类，再按结局取值形态分分类/回归。"""
    if time_col and event_col:
        return TASK_SURVIVAL, f"检测到随访时间列「{time_col}」与事件状态列「{event_col}」，按生存分析处理"

    if not target:
        return TASK_CLUSTERING, "未指定结局变量，按无标签聚类任务处理"

    if target not in df.columns:
        raise DataShapeError(f"结局变量「{target}」不在数据集中",
                             suggestion="请选择数据集中实际存在的列作为结局变量")

    n_unique = int(df[target].nunique(dropna=True))
    if n_unique < 2:
        raise DataShapeError(f"结局变量「{target}」只有单一取值，无法建模",
                             suggestion="请选择取值至少含两个水平的列作为结局变量")

    if _target_is_categorical(df, target):
        return TASK_CLASSIFICATION, f"结局变量「{target}」为分类变量（{n_unique} 个水平），按分类任务处理"
    return TASK_REGRESSION, f"结局变量「{target}」为连续数值，按回归任务处理"


# ------------------------------------------------------------------
# 类别不平衡
# ------------------------------------------------------------------
def imbalance_ratio(y) -> float | None:
    """分类任务：少数类占比；回归/其他返回 None。"""
    s = pd.Series(y).dropna()
    if s.empty or s.nunique() < 2:
        return None
    counts = s.value_counts()
    if len(counts) < 2:
        return None
    return round(float(counts.min() / counts.sum()), 4)


# ------------------------------------------------------------------
# 主画像
# ------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame,
                      standard_result: StandardizationResult | None = None,
                      target: str | None = None,
                      time_col: str | None = None,
                      event_col: str | None = None,
                      patient_id_col: str | None = None) -> DataProfile:
    """整表画像。standard_result 缺省时内部跑模块零标准化（标准名驱动角色判定）。"""
    if df is None or getattr(df, "columns", None) is None or len(df.columns) == 0:
        raise DataShapeError("数据集为空或没有列",
                             suggestion="请上传包含列名与数据的文件后再分析")
    if len(df) == 0:
        raise DataShapeError("数据集没有记录（0 行）", suggestion="请上传至少包含一行数据的文件")

    std = standard_result or standardize_dataframe(df)
    std_by_raw = {c.raw_name: c for c in std.columns}

    # 生存结构：显式入参优先，否则按标准名自动识别
    if not (time_col and event_col):
        auto_t, auto_e = detect_survival_pair(df, std)
        time_col = time_col or auto_t
        event_col = event_col or auto_e

    vinfos = {v["name"]: v for v in infer_variable_types(df)}

    columns: list = []
    warnings: list = []
    for name in df.columns:
        v = vinfos.get(name, {})
        itype = v.get("inferred_type")
        std_role = std_by_raw[name].role_hint if name in std_by_raw else None

        # 角色判定：标准名角色提示优先，取值类型兜底
        is_id = (std_role == "id") or (itype == "id") or (name == patient_id_col)
        if is_id:
            role, excluded, reason = "id", True, "患者标识列（仅作分组键，不进特征矩阵）"
        elif std_role == "survival_time" or name == time_col:
            role, excluded, reason = "survival_time", True, "随访时间列"
        elif std_role == "survival_event" or name == event_col:
            role, excluded, reason = "survival_event", True, "事件状态列"
        elif itype == "constant":
            role, excluded, reason = "constant", True, v.get("exclude_reason") or "常量列"
        elif itype == "datetime":
            role, excluded, reason = "datetime", True, v.get("exclude_reason") or "日期/时间列"
        elif itype == "text":
            role, excluded, reason = "text", True, v.get("exclude_reason") or "自由文本列"
        elif itype in ("numeric", "categorical"):
            role, excluded, reason = itype, False, None
        else:
            role, excluded, reason = itype or "numeric", False, None

        columns.append(ColumnProfile(
            name=name, role=role, inferred_type=itype or "unknown",
            n_unique=int(v.get("n_unique", df[name].nunique(dropna=True))),
            missing_rate=float(v.get("missing_rate", 0.0) or 0.0),
            is_id_like=is_id, excluded=excluded, exclude_reason=reason,
            note=std_by_raw[name].note if name in std_by_raw else None,
        ))

    # 患者 ID：显式 > 标准名 id > 取值特征 id
    if patient_id_col is None:
        id_cands = [c.name for c in columns if c.role == "id"]
        if len(id_cands) > 1:
            warnings.append(f"检测到多个疑似患者标识列 {id_cands}，已取第一个「{id_cands[0]}」作为分组键，"
                            "如需指定其他列请显式传入 patient_id_col")
        patient_id_col = id_cands[0] if id_cands else None

    n_features_used = sum(1 for c in columns
                          if c.role in ("numeric", "categorical")
                          and c.name not in (target, time_col, event_col))
    is_high_dim = n_features_used > len(df)

    task_type, task_reason = infer_task_type(df, target, time_col, event_col, std)

    censoring_rate = None
    if task_type == TASK_SURVIVAL and time_col and event_col:
        sv = validate_survival_inputs(df, time_col, event_col)
        censoring_rate = sv["censoring_rate"]
        warnings.extend(sv.get("warnings", []))

    if len(df) == 1:
        warnings.append("数据集仅 1 行，无法建模")
    if is_high_dim:
        warnings.append(f"高维小样本：特征数 {n_features_used} > 样本数 {len(df)}，"
                        "将走 Lasso/弹性网络/线性 SVM 路线并跳过树模型集成")
    if n_features_used == 0 and task_type not in (TASK_CLUSTERING,):
        warnings.append("没有可用的数值/分类特征列，请检查列类型判定")

    # 数值结局的「分类 vs 回归」存在天然歧义：6–20 个整数水平既可能是多分类标签，
    # 也可能是疼痛评分这类序数量表。推断只能给默认值，必须把选择权明确告知用户。
    if task_type == TASK_CLASSIFICATION and target and target in df.columns \
            and pd.api.types.is_numeric_dtype(df[target]):
        vals = pd.to_numeric(df[target], errors="coerce").dropna()
        if not vals.empty and bool(((vals - vals.round()).abs() <= 1e-9).all()):
            n_levels = int(vals.nunique())
            if 6 <= n_levels <= MAX_CLASS_LEVELS:
                warnings.append(
                    f"结局变量「{target}」有 {n_levels} 个整数水平，已按多分类处理；"
                    "若它是量表/评分等序数变量（如疼痛评分 0–10），请改按回归分析以保留分数高低的信息")

    # 标准引用（raw_name -> ref，报告双标签与下游取数用）
    standard_refs = {c.raw_name: c.ref for c in std.columns}

    imbalance = None
    if task_type == TASK_CLASSIFICATION and target:
        imbalance = imbalance_ratio(df[target])

    return DataProfile(
        n_rows=len(df), n_cols=len(df.columns), columns=columns,
        task_type=task_type, task_reason=task_reason, target=target,
        time_col=time_col, event_col=event_col, patient_id_col=patient_id_col,
        censoring_rate=censoring_rate, imbalance_ratio=imbalance,
        is_high_dim_small_n=is_high_dim, n_features_used=n_features_used,
        standard_refs=standard_refs, warnings=warnings,
    )
