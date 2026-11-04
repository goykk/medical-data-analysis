# medical-data-analysis/backend/automl/__init__.py
"""智能建模分析引擎（模块零 + 五步流水线）。

模块边界：
- standardize  模块零：列名智能识别与标准化（画像之前）；
- profiler     数据画像；router 规则路由；racer 赛马；champion 冠军分析；
- reporting    结构化报告；pipeline 编排；models_registry 模型清单；
- survival     生存数据适配；errors / types 公共底座。

与现有 ml_engine.py 的关系：并行命名空间 /api/automl/*，互不影响。
"""

__version__ = "0.1.0"
