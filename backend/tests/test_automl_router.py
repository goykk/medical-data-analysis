# medical-data-analysis/backend/tests/test_automl_router.py
"""模块三（模型清单 + 规则路由）测试。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl.errors import AutoMLError
from automl.models_registry import (available_models, build_estimator,
                                    sanitize_params, validate_model_data_fit, MODELS)
from automl.profiler import profile_dataframe
from automl.router import route_models
from automl.types import (TASK_CLASSIFICATION, TASK_CLUSTERING, TASK_REGRESSION,
                          TASK_SURVIVAL)


def _profile(n_rows, n_feat, target="结局", task_binary=True):
    rng = np.random.default_rng(0)
    cols = {f"g{i}": rng.normal(size=n_rows) for i in range(n_feat)}
    cols[target] = rng.integers(0, 2, n_rows) if task_binary else rng.normal(size=n_rows)
    return profile_dataframe(pd.DataFrame(cols), target=target)


# ------------------------------------------------------------------
# 规则路由
# ------------------------------------------------------------------
def test_route_classification_normal():
    p = _profile(100, 4)
    r = route_models(p)
    assert r["task"] == TASK_CLASSIFICATION
    assert r["candidate_keys"] == ["logistic", "random_forest", "xgboost", "lightgbm", "catboost"]
    assert r["excluded"] == []


def test_route_regression_normal():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"g{i}": rng.normal(size=80) for i in range(4)})
    df["收缩压"] = rng.normal(130, 15, 80)
    p = profile_dataframe(df, target="收缩压")
    r = route_models(p)
    assert r["task"] == TASK_REGRESSION
    assert r["candidate_keys"] == ["ridge", "lasso", "random_forest", "xgboost", "lightgbm", "catboost"]


def test_route_high_dim_small_n():
    p = _profile(8, 12)      # 特征 12 > 样本 8
    r = route_models(p)
    assert r["candidate_keys"] == ["lasso", "elastic_net", "linear_svm"]
    excluded_keys = {e["key"] for e in r["excluded"]}
    assert excluded_keys == {"random_forest", "xgboost", "lightgbm", "catboost"}
    assert all("过拟合" in e["reason"] for e in r["excluded"])


def test_route_clustering():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"g{i}": rng.normal(size=50) for i in range(4)})
    p = profile_dataframe(df)   # 无 target
    r = route_models(p)
    assert r["task"] == TASK_CLUSTERING
    assert r["candidate_keys"] == ["kmeans", "agglomerative", "dbscan", "gmm"]


def test_route_survival():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({"随访时间": rng.exponential(300, n), "事件": rng.binomial(1, 0.6, n),
                       "年龄": rng.normal(60, 10, n)})
    p = profile_dataframe(df)
    r = route_models(p)
    assert r["task"] == TASK_SURVIVAL
    assert r["candidate_keys"] == ["cox_ph", "random_survival_forest"]


# ------------------------------------------------------------------
# 模型清单
# ------------------------------------------------------------------
def test_available_models_lists_all():
    all_models = available_models()
    keys = {m["key"] for m in all_models}
    assert keys == set(MODELS.keys())
    # 已安装依赖的模型应可用
    by_key = {m["key"]: m for m in all_models}
    assert by_key["xgboost"]["available"] is True
    assert by_key["cox_ph"]["available"] is True
    assert by_key["logistic"]["available"] is True


def test_available_models_filter_by_task():
    cls = available_models(task=TASK_CLASSIFICATION)
    assert all(TASK_CLASSIFICATION in m["task_types"] for m in cls)
    assert all(m["key"] != "cox_ph" for m in cls)


# ------------------------------------------------------------------
# 构造与校验
# ------------------------------------------------------------------
def test_build_logistic():
    from sklearn.linear_model import LogisticRegression
    m = build_estimator("logistic", TASK_CLASSIFICATION)
    assert isinstance(m, LogisticRegression)
    assert m.max_iter == 2000


def test_build_logistic_balanced_class_weight():
    m = build_estimator("logistic", TASK_CLASSIFICATION, imbalance="balanced")
    assert m.class_weight == "balanced"


def test_build_xgboost_single_thread():
    m = build_estimator("xgboost", TASK_CLASSIFICATION)
    # xgboost 3.x 的参数名是 nthread（存储于 get_params），n_jobs 是实例属性
    assert m.get_params().get("nthread") == 1 or getattr(m, "n_jobs", None) == 1


def test_build_survival_models():
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    assert isinstance(build_estimator("cox_ph", TASK_SURVIVAL), CoxPHSurvivalAnalysis)
    rsf = build_estimator("random_survival_forest", TASK_SURVIVAL)
    assert isinstance(rsf, RandomSurvivalForest)
    assert rsf.n_jobs == 1


def test_build_clustering_models():
    from sklearn.cluster import KMeans, DBSCAN
    assert isinstance(build_estimator("kmeans", TASK_CLUSTERING), KMeans)
    assert isinstance(build_estimator("dbscan", TASK_CLUSTERING), DBSCAN)


def test_build_lasso_differs_by_task():
    from sklearn.linear_model import Lasso, LogisticRegression
    assert isinstance(build_estimator("lasso", TASK_REGRESSION), Lasso)
    assert isinstance(build_estimator("lasso", TASK_CLASSIFICATION), LogisticRegression)


def test_build_unknown_key_raises():
    with pytest.raises(AutoMLError):
        build_estimator("not_a_model", TASK_CLASSIFICATION)


def test_validate_mismatch_raises():
    with pytest.raises(AutoMLError) as e:
        validate_model_data_fit("cox_ph", TASK_CLASSIFICATION)
    assert "生存" in str(e.value) or "不适用" in str(e.value)


def test_validate_ok():
    validate_model_data_fit("logistic", TASK_CLASSIFICATION)  # 不抛异常


def test_sanitize_params_whitelist_and_clamp():
    """用户超参：白名单外键丢弃、安全参数锁定、数值钳制。"""
    out = sanitize_params("random_forest", {
        "n_estimators": 10**9,        # 超限 → 钳到 5000
        "n_jobs": -1,                 # 安全参数 → 拒绝覆盖
        "random_state": 0,            # 安全参数 → 拒绝覆盖
        "min_samples_leaf": 5,        # 合法 → 保留
        "evil_key": "x",              # 未声明 → 丢弃
    })
    assert out["n_estimators"] == 5000
    assert out["min_samples_leaf"] == 5
    assert "n_jobs" not in out and "random_state" not in out and "evil_key" not in out


def test_sanitize_params_passthrough_strings():
    # l1_ratio 在 elastic_net 声明参数内，原样通过；penalty 未声明，被丢弃
    out = sanitize_params("elastic_net", {"l1_ratio": 0.7, "penalty": "l2"})
    assert out == {"l1_ratio": 0.7}


def test_build_estimator_applies_sanitized_params():
    est = build_estimator("random_forest", TASK_CLASSIFICATION,
                          params={"n_estimators": 10**9, "n_jobs": -1})
    assert est.n_estimators == 5000
    assert est.n_jobs == 1            # 注册表默认未被用户覆盖
