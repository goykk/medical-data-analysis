# medical-data-analysis/backend/automl/router.py
"""模块三：规则路由。

按数据形态把候选模型池缩小到合适子集，输出「数据形态 -> 候选模型」规则表，
供报告"选型逻辑"与赛马阶段消费。规则逐条对应需求：
- 表格分类：逻辑回归 / 随机森林 / XGBoost / LightGBM / CatBoost；
- 表格回归：Ridge / Lasso / 随机森林 / XGBoost / LightGBM / CatBoost；
- 高维小样本：Lasso / 弹性网络 / 线性 SVM（跳过树模型集成并说明原因）；
- 无标签聚类：KMeans / 层次聚类 / DBSCAN / GMM（自动选 k）；
- 生存分析：Cox 比例风险 / 随机生存森林。
"""
from __future__ import annotations

from automl.types import (DataProfile, TASK_CLASSIFICATION, TASK_CLUSTERING,
                          TASK_REGRESSION, TASK_SURVIVAL)

__all__ = ["route_models", "ROUTING_RULES"]

ROUTING_RULES = [
    {
        "task": TASK_CLASSIFICATION,
        "condition": "表格分类（特征数 ≤ 样本数）",
        "candidates": ["logistic", "random_forest", "xgboost", "lightgbm", "catboost"],
        "reason": "结构型表格数据，线性可解释模型 + 三大梯度提升树同台比较",
    },
    {
        "task": TASK_REGRESSION,
        "condition": "表格回归（特征数 ≤ 样本数）",
        "candidates": ["ridge", "lasso", "random_forest", "xgboost", "lightgbm", "catboost"],
        "reason": "连续结局，线性 + 树模型同台比较",
    },
    {
        "task": (TASK_CLASSIFICATION, TASK_REGRESSION),
        "condition": "高维小样本（特征数 > 样本数）",
        "candidates": ["lasso", "elastic_net", "linear_svm"],
        "reason": "特征数超过样本数，树模型集成缺乏稀疏性假设、极易过拟合，"
                 "改用带正则的稀疏线性族（Lasso / 弹性网络 / 线性 SVM）",
    },
    {
        "task": TASK_CLUSTERING,
        "condition": "无标签数据",
        "candidates": ["kmeans", "agglomerative", "dbscan", "gmm"],
        "reason": "无标签聚类：KMeans / 层次 / GMM 自动选 k，DBSCAN 仅全量评估",
    },
    {
        "task": TASK_SURVIVAL,
        "condition": "含随访时间 + 事件状态列",
        "candidates": ["cox_ph", "random_survival_forest"],
        "reason": "生存分析：Cox 比例风险 + 随机生存森林",
    },
]


def _rule_for(profile: DataProfile) -> dict:
    task = profile.task_type
    if task == TASK_SURVIVAL:
        return ROUTING_RULES[4]
    if task == TASK_CLUSTERING:
        return ROUTING_RULES[3]
    if profile.is_high_dim_small_n:
        return ROUTING_RULES[2]
    if task == TASK_CLASSIFICATION:
        return ROUTING_RULES[0]
    return ROUTING_RULES[1]


def route_models(profile: DataProfile) -> dict:
    """输出 {task, candidate_keys, rule_table, excluded}。"""
    rule = _rule_for(profile)
    candidates = list(rule["candidates"])

    # 高维小样本：树模型集成被跳过，明确列出并说明原因
    excluded = []
    if profile.is_high_dim_small_n and profile.task_type in (TASK_CLASSIFICATION, TASK_REGRESSION):
        tree_models = ["random_forest", "xgboost", "lightgbm", "catboost"]
        excluded = [{"key": k, "reason": "高维小样本下树模型集成易过拟合、无稀疏性假设，已跳过"}
                    for k in tree_models]

    return {
        "task": profile.task_type,
        "candidate_keys": candidates,
        "rule_table": [{
            "task": rule["task"] if isinstance(rule["task"], tuple) else [rule["task"]],
            "condition": rule["condition"],
            "candidates": rule["candidates"],
            "reason": rule["reason"],
        }],
        "excluded": excluded,
    }
