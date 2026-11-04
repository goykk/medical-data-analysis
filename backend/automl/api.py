# medical-data-analysis/backend/automl/api.py
"""FastAPI 路由（/api/automl/*）：挂载自动建模引擎。

通过 create_router(get_dataset) 注入数据集访问函数，避免与 app.py 形成循环 import。
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from automl import pipeline
from automl.leakage import DEFAULT_LEAKAGE_THRESHOLD, apply_leakage_decision, detect_leakage
from automl.models_registry import available_models
from automl.profiler import profile_dataframe
from automl.standardize import load_user_lexicon, save_user_mapping, standardize_dataframe

__all__ = ["create_router"]


class _DatasetRequest(BaseModel):
    dataset_id: str
    target: str | None = None
    time_col: str | None = None
    event_col: str | None = None
    patient_id_col: str | None = None


class AnalyzeRequest(_DatasetRequest):
    leakage_decision: str = "ask"          # ask | drop | keep
    confirmed_drop: list = Field(default_factory=list)
    mapping_decisions: dict = Field(default_factory=dict)
    with_shap: bool = True
    leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD


class ManualTrainRequest(_DatasetRequest):
    model_key: str
    params: dict | None = None
    with_shap: bool = True


class ConfirmRequest(_DatasetRequest):
    mapping_decisions: dict = Field(default_factory=dict)   # {原始名: 标准名或 null}
    leakage_decision: dict = Field(default_factory=dict)    # {action, columns}
    save_user_mappings: bool = True


def create_router(get_dataset):
    router = APIRouter(prefix="/api/automl", tags=["automl"])

    def _df(dataset_id: str):
        return get_dataset(dataset_id)["df"]

    @router.get("/models")
    def models(task: str | None = None):
        return {"models": available_models(task=task)}

    @router.post("/standardize")
    def standardize(req: _DatasetRequest):
        std = standardize_dataframe(_df(req.dataset_id))
        return {"dataset_id": req.dataset_id, "standardization": std.to_dict()}

    @router.post("/prepare")
    def prepare(req: _DatasetRequest):
        return pipeline.prepare(_df(req.dataset_id), None, req.target, req.time_col,
                                req.event_col, req.patient_id_col)

    @router.post("/confirm")
    def confirm(req: ConfirmRequest):
        df = _df(req.dataset_id)
        if req.save_user_mappings:
            for raw, std_name in req.mapping_decisions.items():
                if std_name:
                    save_user_mapping(raw, std_name)
        # 重新跑标准化 + 泄露检测，按用户决定生成最终映射
        std = standardize_dataframe(df)
        profile = profile_dataframe(df, std, req.target, req.time_col, req.event_col,
                                    req.patient_id_col)
        findings = detect_leakage(df, req.target, profile.task_type, profile.time_col,
                                  profile.event_col, profile.patient_id_col)
        action = req.leakage_decision.get("action", "drop")
        cols = req.leakage_decision.get("columns")
        leak_res = apply_leakage_decision(profile, findings, action, cols)
        return {"final_mapping": std.mapping_table, "dropped_leakage": leak_res["dropped"],
                "kept_leakage": leak_res["kept"], "note": leak_res["note"]}

    @router.post("/analyze")
    def analyze(req: AnalyzeRequest):
        # 阈值钳制在有效区间：传 999 之类极端值会静默关闭泄露检测
        threshold = min(max(float(req.leakage_threshold), 0.5), 0.999)
        return pipeline.run_auto(
            _df(req.dataset_id), None, req.target, req.time_col, req.event_col,
            req.patient_id_col, req.leakage_decision, req.confirmed_drop or None,
            req.mapping_decisions or None, req.with_shap, threshold)

    @router.post("/train")
    def train(req: ManualTrainRequest):
        return pipeline.run_manual(
            _df(req.dataset_id), None, req.model_key, req.target, req.time_col,
            req.event_col, req.patient_id_col, req.params, req.with_shap)

    @router.get("/lexicon/user")
    def user_lexicon():
        return {"user_mappings": load_user_lexicon()}

    return router
