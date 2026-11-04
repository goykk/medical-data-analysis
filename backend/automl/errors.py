# medical-data-analysis/backend/automl/errors.py
"""自动建模引擎统一异常体系。

约定与 ml_engine.MLError 一致：
- message 为面向用户的明确中文原因（禁止底层库原始报错直接外溢）；
- suggestion 为可执行的修正建议；
- detail 附带结构化信息供前端展示。
"""

__all__ = ["AutoMLError", "DataShapeError", "SurvivalDataError", "ModelFitError"]


class AutoMLError(Exception):
    """建模无法继续时抛出。"""

    def __init__(self, message: str, suggestion: str = "", detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.detail = detail or {}


class DataShapeError(AutoMLError):
    """数据形态不满足最低要求：空表 / 无列 / 空列名 / 无特征 / 目标列缺失 / 全 NaN / 标签单一取值。"""


class SurvivalDataError(AutoMLError):
    """生存数据结构非法：事件列非 0/1、随访时间为负、删失比例为 0 或 1 等。"""


class ModelFitError(AutoMLError):
    """单个模型训练失败：仅在赛马内部被捕获并记录，不中断整次分析。"""
