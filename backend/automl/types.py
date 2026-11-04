# medical-data-analysis/backend/automl/types.py
"""自动建模引擎的公共数据结构（dataclass）。

各模块的输入输出一律用这里的结构，避免"返回 dict、靠猜 key"的散乱约定。
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "TASK_CLASSIFICATION", "TASK_REGRESSION", "TASK_CLUSTERING", "TASK_SURVIVAL",
    "ColumnProfile", "DataProfile", "LeakageFinding", "ModelSpec", "RaceRow",
    "ChampionDetail",
]

TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
TASK_CLUSTERING = "clustering"
TASK_SURVIVAL = "survival"

# 列角色（role）：numeric / categorical / datetime / text / id / constant /
#                 survival_time / survival_event
FEATURE_ROLES = ("numeric", "categorical")


@dataclass
class ColumnProfile:
    """单列的画像。role 由"取值特征 + 模块零标准名角色提示"共同判定。"""

    name: str                # 原始列名
    role: str                # 见 FEATURE_ROLES 与生存/id 角色
    inferred_type: str       # 取值层面的类型（来自 validators.infer_variable_types）
    n_unique: int
    missing_rate: float      # 0..1
    is_id_like: bool
    excluded: bool           # 是否不参与特征建模
    exclude_reason: str | None = None
    note: str | None = None


@dataclass
class LeakageFinding:
    """单列泄露检测结果。score 越接近 1 越可能泄露。"""

    column: str          # 原始列名
    ref: str | None      # 标准引用（报告双标签用，可空）
    score: float         # 判别力：AUC / |Spearman| / |C-index-0.5|*2
    threshold: float
    metric: str          # auc | spearman | cindex
    verdict: str         # leak | ok
    note: str = ""


@dataclass
class ModelSpec:
    """模型注册条目：元数据 + 默认超参 + SHAP 策略 + 依赖。"""

    key: str
    label: str
    family: str                  # linear | tree | svm | linear_sparse | cluster | mixture | survival
    task_types: tuple            # 适用任务（可多个）
    params: dict = field(default_factory=dict)         # 构造器默认超参
    param_docs: list = field(default_factory=list)     # [{name,label,default,desc}]
    supports_shap: str = "none"  # tree | linear | none
    shap_note: str = ""
    requires: str | None = None  # 依赖包 import 名，缺失则 available=False
    notes: str = ""

    def to_dict(self, available: bool | None = None, reason: str = "") -> dict:
        d = {
            "key": self.key, "label": self.label, "family": self.family,
            "task_types": list(self.task_types), "params": self.params,
            "param_docs": self.param_docs, "supports_shap": self.supports_shap,
            "shap_note": self.shap_note, "requires": self.requires, "notes": self.notes,
        }
        if available is not None:
            d["available"] = available
            d["reason"] = reason
        return d


@dataclass
class RaceRow:
    """赛马排行榜一行。单个模型训练失败不中断整次分析，记录失败原因后剔除。"""

    model_key: str
    label: str
    metric_name: str           # 排名主指标（auc/rmse/silhouette/cindex）
    metric_value: float
    ci_low: float | None = None
    ci_high: float | None = None
    fit_seconds: float = 0.0
    status: str = "ok"         # ok | failed
    error_message: str | None = None
    metrics: dict = field(default_factory=dict)      # 次要指标（auprc/accuracy/mae/r2/ch_index...）
    higher_is_better: bool = True
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key, "label": self.label,
            "metric_name": self.metric_name, "metric_value": self.metric_value,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "fit_seconds": round(float(self.fit_seconds), 3), "status": self.status,
            "error_message": self.error_message, "metrics": self.metrics,
            "higher_is_better": self.higher_is_better, "notes": self.notes,
        }


@dataclass
class ChampionDetail:
    """冠军模型深度分析：全量重训 + 特征重要性 + SHAP + 中文解读。"""

    model_key: str
    label: str
    params: dict
    metric_name: str | None
    metric_value: float | None
    importance_top: list           # [{variable, importance, raw}]
    shap: dict | None              # {method, warning, n_sample, features}
    interpretation: str
    refit_seconds: float
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key, "label": self.label, "params": self.params,
            "metric_name": self.metric_name, "metric_value": self.metric_value,
            "importance_top15": self.importance_top, "shap": self.shap,
            "interpretation": self.interpretation,
            "refit_seconds": round(float(self.refit_seconds), 3), "notes": self.notes,
        }


@dataclass
class DataProfile:
    """数据集画像，供规则路由 / 赛马 / 冠军分析 / 报告消费。"""

    n_rows: int
    n_cols: int
    columns: list            # list[ColumnProfile]
    task_type: str           # classification|regression|clustering|survival
    task_reason: str         # 为什么这样判（报告"选型逻辑"用）
    target: str | None
    time_col: str | None
    event_col: str | None
    patient_id_col: str | None   # GroupKFold 分组键（仅作分组，不进特征）
    censoring_rate: float | None
    imbalance_ratio: float | None
    is_high_dim_small_n: bool    # 特征数 > 样本数
    n_features_used: int
    standard_refs: dict = field(default_factory=dict)   # 原始列名 -> 标准引用（含来源后缀）
    warnings: list = field(default_factory=list)

    def feature_columns(self) -> list:
        """可入模特征列（排除 target/time/event/id/常量/文本/日期）。"""
        return [c.name for c in self.columns
                if c.role in FEATURE_ROLES and c.name not in (self.target, self.time_col, self.event_col)]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "n_cols": self.n_cols,
            "task_type": self.task_type, "task_reason": self.task_reason,
            "target": self.target, "time_col": self.time_col, "event_col": self.event_col,
            "patient_id_col": self.patient_id_col,
            "censoring_rate": self.censoring_rate, "imbalance_ratio": self.imbalance_ratio,
            "is_high_dim_small_n": self.is_high_dim_small_n,
            "n_features_used": self.n_features_used,
            "columns": [c.__dict__ for c in self.columns],
            "warnings": self.warnings,
        }
