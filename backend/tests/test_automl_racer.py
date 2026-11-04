# medical-data-analysis/backend/tests/test_automl_racer.py
"""模块四（交叉验证赛马）测试：分类/回归/多分类/聚类/生存 + 患者级划分 + 失败不中断。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl.profiler import profile_dataframe
from automl.racer import (build_cv, build_preprocessor, fit_pipeline,
                          imbalance_weight_mode, make_sampler, race_models,
                          _configure_imbalance)
from automl.types import TASK_CLASSIFICATION, TASK_REGRESSION


def _cls_df(n=120, n_feat=4, weights=None, seed=0):
    from sklearn.datasets import make_classification
    if weights is None:
        weights = [0.5, 0.5]
    X, y = make_classification(n_samples=n, n_features=n_feat, n_informative=3,
                               n_redundant=0, weights=list(weights), random_state=seed)
    df = pd.DataFrame(X, columns=[f"g{i}" for i in range(n_feat)])
    df["结局"] = y
    return df


def _surv_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"年龄": rng.normal(60, 10, n).round(0),
                      "血红蛋白": rng.normal(130, 15, n).round(1)})
    lin = 0.05 * (X["年龄"] - 60) - 0.01 * (X["血红蛋白"] - 130)
    t = rng.weibull(1.5, n) * np.exp(-lin) * 200
    c = rng.exponential(300, n)
    ev = (t <= c).astype(int)
    X["随访时间"] = np.where(ev == 1, t, c)
    X["事件"] = ev
    return X


def test_race_classification_binary():
    df = _cls_df()
    p = profile_dataframe(df, target="结局")
    rows = race_models(df, p, ["logistic", "random_forest"], n_splits=4)
    assert len(rows) == 2
    assert all(r.status == "ok" for r in rows)
    assert rows[0].metric_name == "auc"
    assert 0.0 <= rows[0].metric_value <= 1.0
    assert rows[0].ci_low is not None and rows[0].ci_high is not None
    assert "auprc" in rows[0].metrics and "accuracy" in rows[0].metrics
    # 按 AUC 降序
    assert rows[0].metric_value >= rows[1].metric_value


def test_race_multiclass_has_macro_auprc():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=150, n_features=4, n_informative=3,
                               n_redundant=0, n_classes=3, n_clusters_per_class=1,
                               random_state=1)
    df = pd.DataFrame(X, columns=["g0", "g1", "g2", "g3"])
    df["分级"] = y
    p = profile_dataframe(df, target="分级")
    rows = race_models(df, p, ["logistic"], n_splits=3)
    assert rows[0].status == "ok"
    assert rows[0].metric_name == "auc"
    assert "auprc" in rows[0].metrics


def test_race_regression():
    rng = np.random.default_rng(2)
    n = 120
    X = rng.normal(size=(n, 4))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.5, n)
    df = pd.DataFrame(X, columns=["g0", "g1", "g2", "g3"])
    df["y"] = y
    p = profile_dataframe(df, target="y")
    rows = race_models(df, p, ["ridge", "random_forest"], n_splits=4)
    assert rows[0].metric_name == "rmse"
    assert rows[0].higher_is_better is False
    assert "mae" in rows[0].metrics and "r2" in rows[0].metrics
    assert rows[0].metric_value <= rows[1].metric_value  # RMSE 升序


def test_single_model_failure_does_not_abort(monkeypatch):
    import automl.racer as racer
    df = _cls_df()
    p = profile_dataframe(df, target="结局")
    orig = racer.build_estimator

    def fake(key, task, params=None, imbalance=None):
        if key == "random_forest":
            raise RuntimeError("模拟训练失败")
        return orig(key, task, params, imbalance)

    monkeypatch.setattr(racer, "build_estimator", fake)
    rows = race_models(df, p, ["logistic", "random_forest"], n_splits=3)
    ok = [r for r in rows if r.status == "ok"]
    failed = [r for r in rows if r.status == "failed"]
    assert len(ok) == 1 and len(failed) == 1
    assert failed[0].model_key == "random_forest"
    assert "模拟训练失败" in failed[0].error_message


def test_group_kfold_keeps_patient_in_one_fold():
    # 同一患者多行：分组划分后任一患者不得同时出现在训练与验证折
    rng = np.random.default_rng(0)
    n_patients = 20
    rows = []
    for pid in range(n_patients):
        for _ in range(3):
            rows.append({"pid": f"P{pid:02d}", "x": rng.normal(), "y": rng.integers(0, 2)})
    df = pd.DataFrame(rows)
    p = profile_dataframe(df, target="y")
    cv = build_cv(p, n_splits=5, patient_id_col="pid")
    from sklearn.model_selection import GroupKFold
    assert isinstance(cv, GroupKFold)
    X = df[["x"]]
    y = df["y"].to_numpy()
    groups = df["pid"].to_numpy()
    for tr, te in cv.split(X, y, groups=groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_imbalance_class_weight_auto():
    from sklearn.linear_model import LogisticRegression
    df = _cls_df(weights=(0.92, 0.08), seed=5)
    p = profile_dataframe(df, target="结局")
    assert p.imbalance_ratio < 0.1
    from automl.models_registry import build_estimator
    est = build_estimator("logistic", TASK_CLASSIFICATION)
    est, note = _configure_imbalance(est, "logistic", p)
    assert isinstance(est, LogisticRegression) and est.class_weight == "balanced"
    assert note and "类权重" in note


def test_xgboost_scale_pos_weight_only_for_binary():
    """scale_pos_weight 只在二分类有定义：多分类下设置会被 XGBoost 拒绝，导致模型训练失败。"""
    pytest.importorskip("xgboost")
    from types import SimpleNamespace
    from automl.models_registry import build_estimator

    profile = SimpleNamespace(task_type=TASK_CLASSIFICATION, imbalance_ratio=0.04)

    binary_est = build_estimator("xgboost", TASK_CLASSIFICATION)
    binary_est, _ = _configure_imbalance(binary_est, "xgboost", profile, n_classes=2)
    assert binary_est.get_params()["scale_pos_weight"] == pytest.approx(24.0)

    multi_est = build_estimator("xgboost", TASK_CLASSIFICATION)
    multi_est, note = _configure_imbalance(multi_est, "xgboost", profile, n_classes=3)
    assert multi_est.get_params()["scale_pos_weight"] is None   # 保持默认，未被设置
    assert note is None


def test_imbalance_weight_mode_only_for_xgboost_multiclass():
    """XGBoost 多分类无 class_weight、scale_pos_weight 也无效，只能走逐样本权重。"""
    from types import SimpleNamespace
    imbalanced = SimpleNamespace(task_type=TASK_CLASSIFICATION, imbalance_ratio=0.05)
    balanced = SimpleNamespace(task_type=TASK_CLASSIFICATION, imbalance_ratio=0.5)
    regression = SimpleNamespace(task_type=TASK_REGRESSION, imbalance_ratio=None)

    assert imbalance_weight_mode("xgboost", imbalanced, 3) is True
    assert imbalance_weight_mode("xgboost", imbalanced, 2) is False     # 二分类走 scale_pos_weight
    assert imbalance_weight_mode("logistic", imbalanced, 3) is False
    assert imbalance_weight_mode("xgboost", balanced, 3) is False
    assert imbalance_weight_mode("xgboost", regression, None) is False
    assert imbalance_weight_mode("xgboost", imbalanced, 3, sampler=object()) is False


def test_fit_pipeline_routes_balanced_sample_weight():
    pytest.importorskip("xgboost")
    from sklearn.pipeline import Pipeline
    from automl.models_registry import build_estimator

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = np.array([0] * 50 + [1] * 7 + [2] * 3)
    est = build_estimator("xgboost", TASK_CLASSIFICATION,
                          {"n_estimators": 20, "max_depth": 3})
    pipe = Pipeline([("prep", build_preprocessor(["a", "b"], [])), ("model", est)])
    fit_pipeline(pipe, X, y, weighted=True)
    assert list(pipe.named_steps["model"].classes_) == [0, 1, 2]
    assert pipe.predict(X).shape == (len(X),)


def test_race_records_per_fold_confusion_matrices():
    """赛马必须留下逐折混淆矩阵：医疗场景需要看假阴性/假阳性，而非只有 AUC。"""
    df = _cls_df(weights=(0.6, 0.4), seed=11)
    p = profile_dataframe(df, target="结局")
    rows = race_models(df, p, ["logistic"], n_splits=4)
    m = rows[0].metrics
    assert len(m["confusion_matrices"]) == 4
    assert all(len(cm) == 2 and len(cm[0]) == 2 for cm in m["confusion_matrices"])
    mean_manual = np.mean(m["confusion_matrices"], axis=0)
    assert np.allclose(m["confusion_matrix_mean"], np.round(mean_manual, 2), atol=1e-6)


def test_race_multiclass_confusion_matrix_is_square():
    rng = np.random.default_rng(12)
    n = 180
    X = rng.normal(size=(n, 3))
    y = np.where(X[:, 0] + X[:, 1] > 1, 2, np.where(X[:, 0] - X[:, 1] > 0, 1, 0))
    df = pd.DataFrame(X, columns=["g0", "g1", "g2"])
    df["分级"] = y
    p = profile_dataframe(df, target="分级")
    rows = race_models(df, p, ["logistic"], n_splits=3)
    m = rows[0].metrics
    assert len(m["confusion_matrix_mean"]) == 3
    assert all(len(row) == 3 for row in m["confusion_matrix_mean"])
    # 每折矩阵总和 = 该折验证集样本数（三折合计约等于全样本）
    per_fold_total = sum(sum(sum(r) for r in cm) for cm in m["confusion_matrices"])
    assert per_fold_total == n


def test_make_sampler_returns_smote_when_imbalanced():
    df = _cls_df(weights=(0.95, 0.05), seed=6)
    p = profile_dataframe(df, target="结局")
    assert make_sampler(p, strategy="smote") is not None
    assert make_sampler(p, strategy="none") is None
    # 均衡数据不采样
    df2 = _cls_df(weights=(0.5, 0.5))
    assert make_sampler(profile_dataframe(df2, target="结局"), strategy="smote") is None


def test_race_with_large_sample_subsampling():
    df = _cls_df(n=120, weights=(0.5, 0.5), seed=7)
    p = profile_dataframe(df, target="结局")
    rows = race_models(df, p, ["logistic"], n_splits=3, sample_threshold=60)
    assert any("抽样赛马" in n for n in rows[0].notes)
    assert "分层抽样" in "".join(rows[0].notes)


def test_race_all_missing_column_dropped():
    df = _cls_df()
    df["全空列"] = np.nan
    p = profile_dataframe(df, target="结局")
    # 全缺失列由画像阶段自动排除并记录原因（infer_variable_types → excluded + 全列缺失）
    col = next(c for c in p.columns if c.name == "全空列")
    assert col.excluded and col.exclude_reason
    rows = race_models(df, p, ["logistic"], n_splits=3)
    assert rows[0].status == "ok"


def test_drop_all_missing_helper():
    from automl.racer import _drop_all_missing
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [np.nan, np.nan, np.nan],
                       "c": ["x", "y", "z"]})
    out, dropped, num, cat = _drop_all_missing(df, ["a"], ["c"])
    assert dropped == ["b"]
    assert "b" not in out.columns
    assert "a" in num and "c" in cat


def test_race_survival():
    df = _surv_df()
    p = profile_dataframe(df)   # 标准名识别出 随访时间+事件
    rows = race_models(df, p, ["cox_ph", "random_survival_forest"], n_splits=3)
    assert all(r.status == "ok" for r in rows)
    assert rows[0].metric_name == "cindex"
    assert 0.5 <= rows[0].metric_value <= 1.0


def test_race_clustering():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=60, centers=3, n_features=3, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    p = profile_dataframe(df)   # 无 target → 聚类
    rows = race_models(df, p, ["kmeans", "dbscan"], )
    assert rows[0].metric_name == "silhouette"
    assert all(r.status == "ok" for r in rows)
    dbscan = next(r for r in rows if r.model_key == "dbscan")
    assert any("无新样本预测" in n for n in dbscan.notes)


def test_numeric_coded_categorical_column():
    """数值编码类别列（int64）不得让预处理崩溃。

    回归用例：曾对 int64 类别列用字符串「缺失」填充，sklearn 1.8 抛
    ValueError: fill_value='缺失' cannot be cast to dtype('int64')。
    """
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({
        "年龄": rng.normal(60, 10, n).round(0),
        "性别": rng.integers(0, 2, n),        # 0/1 数值编码类别
        "分级": rng.integers(1, 4, n),        # 三分类数值编码
        "结局": rng.integers(0, 2, n),
    })
    p = profile_dataframe(df, target="结局")
    role = next(c for c in p.columns if c.name == "性别").role
    assert role == "categorical", f"前置条件：性别应为 categorical，实际 {role}"
    rows = race_models(df, p, ["logistic"], n_splits=3)
    assert rows[0].status == "ok", rows[0].error_message


def test_cors_allows_both_loopback_forms():
    """127.0.0.1 与 localhost 必须都被允许，否则 POST 预检被拒、
    表现为「页面能打开但请求全部失败」。"""
    from config import CORS_ORIGINS
    assert "http://127.0.0.1:3001" in CORS_ORIGINS
    assert "http://localhost:3001" in CORS_ORIGINS
