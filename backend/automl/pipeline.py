# medical-data-analysis/backend/automl/pipeline.py
"""模块七：模式一（自动选最优模型）与模式二（手动选模型）编排。

模式一五步：标准化(模块零) -> 画像 -> 泄露检测与处置 -> 规则路由 -> 赛马 -> 冠军分析 -> 报告。
模式二与模式一共用同一套训练/评估/报告流水线，仅跳过路由（指定单一模型）。
"""
from __future__ import annotations

from automl.champion import analyze_champion
from automl.leakage import (apply_leakage_decision, detect_leakage,
                            DEFAULT_LEAKAGE_THRESHOLD)
from automl.models_registry import validate_model_data_fit
from automl.profiler import profile_dataframe
from automl.racer import (evaluate_clustering, evaluate_survival, race_models,
                          DEFAULT_SAMPLE_THRESHOLD)
from automl.reporting import build_report
from automl.router import route_models
from automl.standardize import standardize_dataframe, save_user_mapping
from automl.types import TASK_CLUSTERING, TASK_SURVIVAL

__all__ = ["run_auto", "run_manual", "prepare", "build_pending_confirmations"]

DEFAULT_POLICY = {
    "skip_mapping": "低置信映射按原名作未知列处理（保留为普通特征）",
    "skip_leakage": "泄露列默认剔除",
    "report_note": "跳过确认时采用默认策略，并在报告局限性中显式说明",
}


def build_pending_confirmations(standardization, findings: list) -> list:
    """列名映射待确认项与泄露列警告合并为一次交互。"""
    pending = []
    for c in standardization.columns:
        if c.status == "pending":
            pending.append({"kind": "mapping", "raw_name": c.raw_name,
                            "candidates": c.candidates, "reason": c.confirm_reason})
    for f in findings:
        if f.verdict == "leak":
            pending.append({"kind": "leakage", "column": f.column, "score": f.score,
                            "metric": f.metric, "suggestion": "建议剔除"})
    return pending


def prepare(df, schema=None, target: str | None = None, time_col: str | None = None,
            event_col: str | None = None, patient_id_col: str | None = None,
            leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD) -> dict:
    """一次交互拿齐：标准化结果 + 画像 + 泄露警告 + 路由预览 + 待确认项。"""
    std = standardize_dataframe(df)
    profile = profile_dataframe(df, std, target, time_col, event_col, patient_id_col)
    findings = detect_leakage(df, target, profile.task_type, profile.time_col,
                              profile.event_col, profile.patient_id_col,
                              leakage_threshold, profile.standard_refs)
    routing = route_models(profile)
    return {
        "standardization": std.to_dict(),
        "profile": profile.to_dict(),
        "leakage": [f.__dict__ for f in findings],
        "routing": routing,
        "pending_confirmations": build_pending_confirmations(std, findings),
        "default_policy": DEFAULT_POLICY,
    }


def run_auto(df, schema=None, target: str | None = None, time_col: str | None = None,
             event_col: str | None = None, patient_id_col: str | None = None,
             leakage_decision: str = "ask", confirmed_drop: list | None = None,
             mapping_decisions: dict | None = None, with_shap: bool = True,
             leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
             sample_threshold: int = DEFAULT_SAMPLE_THRESHOLD) -> dict:
    """自动选最优模型：五步流水线，返回结构化报告。"""
    # 用户确认过的列名映射写入用户词典（持久生效，不污染全局词典）
    if mapping_decisions:
        for raw, std_name in mapping_decisions.items():
            if std_name:
                save_user_mapping(raw, std_name)

    std = standardize_dataframe(df)
    profile = profile_dataframe(df, std, target, time_col, event_col, patient_id_col)
    findings = detect_leakage(df, target, profile.task_type, profile.time_col,
                              profile.event_col, profile.patient_id_col,
                              leakage_threshold, profile.standard_refs)
    leak_res = apply_leakage_decision(profile, findings, leakage_decision, confirmed_drop)

    # 处置完成后才进入赛马：剔除泄露列
    df2 = df.drop(columns=leak_res["dropped"]) if leak_res["dropped"] else df

    std2 = standardize_dataframe(df2)
    profile2 = profile_dataframe(df2, std2, target, time_col, event_col, patient_id_col)
    routing = route_models(profile2)
    leaderboard = race_models(df2, profile2, routing["candidate_keys"],
                              patient_id_col, sample_threshold=sample_threshold)

    winner = next((r for r in leaderboard if r.status == "ok"), None)
    champion = None
    if winner and profile2.task_type != TASK_CLUSTERING:
        champion = analyze_champion(df2, profile2, winner.model_key,
                                    metric_name=winner.metric_name,
                                    metric_value=winner.metric_value,
                                    with_shap=with_shap)

    sampling_note = next((n for r in leaderboard for n in r.notes if "抽样赛马" in n), None)
    return build_report(profile2, std2, findings, leak_res, routing, leaderboard,
                        champion, sampling_note)


def run_manual(df, schema=None, model_key: str | None = None,
               target: str | None = None, time_col: str | None = None,
               event_col: str | None = None, patient_id_col: str | None = None,
               params: dict | None = None, with_shap: bool = True) -> dict:
    """手动选模型：指定 model_key，与自动模式共用赛马/冠军/报告流水线。

    泄露检测与自动模式一致：手动模式没有逐列确认交互，按既定默认策略
    （泄露列默认剔除）处置，并在报告 limitations 中显式说明——
    绝不让含泄露列的模型带着虚高指标悄无声息地交付。
    """
    if not model_key:
        from automl.errors import AutoMLError
        raise AutoMLError("手动模式必须指定 model_key", suggestion="请传入要训练的模型 key")

    std = standardize_dataframe(df)
    profile = profile_dataframe(df, std, target, time_col, event_col, patient_id_col)

    findings = detect_leakage(df, target, profile.task_type, profile.time_col,
                              profile.event_col, profile.patient_id_col,
                              DEFAULT_LEAKAGE_THRESHOLD, profile.standard_refs)
    leak_res = apply_leakage_decision(profile, findings, "ask", None)
    df2 = df.drop(columns=leak_res["dropped"]) if leak_res["dropped"] else df
    if leak_res["dropped"]:
        # 剔除泄露列后特征集合变化，需重建标准化与画像
        std = standardize_dataframe(df2)
        profile = profile_dataframe(df2, std, target, time_col, event_col, patient_id_col)
    validate_model_data_fit(model_key, profile.task_type)

    if profile.task_type == TASK_CLUSTERING:
        leaderboard = evaluate_clustering(df2, profile, [model_key])
        champion = None
    elif profile.task_type == TASK_SURVIVAL:
        leaderboard = evaluate_survival(df2, profile, [model_key], patient_id_col)
        winner = next((r for r in leaderboard if r.status == "ok"), None)
        champion = analyze_champion(df2, profile, model_key, params, with_shap,
                                    metric_name="cindex",
                                    metric_value=winner.metric_value if winner else None) if winner else None
    else:
        leaderboard = race_models(df2, profile, [model_key], patient_id_col, params=params)
        winner = next((r for r in leaderboard if r.status == "ok"), None)
        champion = analyze_champion(df2, profile, model_key, params, with_shap,
                                    metric_name=winner.metric_name if winner else None,
                                    metric_value=winner.metric_value if winner else None) if winner else None

    return build_report(profile, std, findings, leak_res, route_models(profile),
                        leaderboard, champion)
