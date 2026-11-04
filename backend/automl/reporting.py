# medical-data-analysis/backend/automl/reporting.py
"""模块六：结构化报告组装 + JSON 安全化。

报告 JSON 固定携带免责声明字段；能回答"系统为什么选这个模型"
（数据画像 -> 选型逻辑 -> 排行榜 -> 冠军详解 -> 局限性）。
"""
from __future__ import annotations

import math

import numpy as np

from automl.types import TASK_CLASSIFICATION, TASK_REGRESSION

__all__ = ["DISCLAIMER", "build_report", "json_safe"]

DISCLAIMER = "本结果仅供研究参考与辅助决策，不能替代临床医生判断。"


def json_safe(obj):
    """递归把 NaN/Inf 转为 None，numpy 标量转原生类型。

    starlette JSONResponse 用 allow_nan=False，响应体含 NaN/Inf 会直接 500
    （本项目已在 /api/ml/train 多分类上踩过此坑），报告必须做安全化。
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        return json_safe(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


def _limitations(profile, leakage_result, sampling_note, champion):
    lims = list(profile.warnings)
    if leakage_result:
        dropped = leakage_result.get("dropped", [])
        if dropped:
            lims.append(f"泄露列 {len(dropped)} 个已按策略处理：{'、'.join(dropped)}")
        if leakage_result.get("note"):
            lims.append(leakage_result["note"])
    if sampling_note:
        lims.append(sampling_note)
    if champion and champion.shap and champion.shap.get("warning"):
        lims.append(champion.shap["warning"])
    if champion:
        lims.extend(champion.notes)
    lims.append("模型表现基于当前数据集，外推到其他人群/机构前需外部验证。")
    return lims


def build_report(profile, standardization=None, leakage=None, leakage_result=None,
                 routing=None, leaderboard=None, champion=None,
                 sampling_note=None) -> dict:
    """组装完整结构化报告。所有入参均可选，缺省置空/默认，保证 JSON 可序列化。"""
    leaderboard = leaderboard or []
    report = {
        "profile": profile.to_dict() if profile else {},
        "standardization": (standardization.to_dict() if standardization else None),
        "leakage": {
            "findings": [f.__dict__ for f in (leakage or [])],
            "decision": leakage_result,
        },
        "routing": routing or {},
        "leaderboard": [r.to_dict() for r in leaderboard],
        "champion": champion.to_dict() if champion else None,
        "limitations": _limitations(profile, leakage_result, sampling_note, champion),
        "disclaimer": DISCLAIMER,
        "reproducibility": {
            "random_state": getattr(__import__("config"), "RANDOM_SEED", None),
            "note": "所有随机过程使用统一 random_state，结果可复现",
        },
    }
    return json_safe(report)
