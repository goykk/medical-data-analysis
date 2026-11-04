# medical-data-analysis/backend/tests/test_automl_champion_reporting.py
"""模块五/六（冠军分析 + 报告）测试。"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import automl.champion as champion_mod
from automl.champion import analyze_champion, interpret
from automl.profiler import profile_dataframe
from automl.racer import race_models
from automl.reporting import DISCLAIMER, build_report, json_safe
from automl.types import (TASK_CLASSIFICATION, TASK_REGRESSION, TASK_SURVIVAL,
                          ChampionDetail)


def _cls_df(n=150, n_feat=6, seed=0):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=n_feat, n_informative=4,
                               n_redundant=0, random_state=seed)
    df = pd.DataFrame(X, columns=[f"g{i}" for i in range(n_feat)])
    df["结局"] = y
    return df


def test_importance_values_tolerates_raising_property():
    """模型属性主动抛异常时不能冒泡（sksurv RandomSurvivalForest 的
    feature_importances_ 是 NotImplementedError property，hasattr 无法拦截）。

    回归：旧实现 hasattr(model, "feature_importances_") 触发异常，
    /api/automl/analyze 在生存分析冠军解释阶段 500。
    """
    class RaisingModel:
        @property
        def feature_importances_(self):
            raise NotImplementedError()

        @property
        def coef_(self):
            raise NotImplementedError()

    model = RaisingModel()
    assert champion_mod._attr_or_none(model, "feature_importances_") is None
    assert champion_mod._importance_values(model) is None
    assert champion_mod._importance_top(model, ["x1", "x2"]) == []


def test_attr_or_none_rethrows_unexpected_model_failure():
    """只有 AttributeError/NotImplementedError 才降级；真实故障必须冒泡。"""
    class BrokenModel:
        @property
        def feature_importances_(self):
            raise RuntimeError("model state corrupted")

    with pytest.raises(RuntimeError, match="model state corrupted"):
        champion_mod._attr_or_none(BrokenModel(), "feature_importances_")


def test_champion_logistic_importance_and_shap():
    df = _cls_df()
    p = profile_dataframe(df, target="结局")
    c = analyze_champion(df, p, "logistic", metric_name="auc", metric_value=0.9)
    assert c.model_key == "logistic"
    assert c.importance_top and len(c.importance_top) <= 15
    assert c.importance_top[0]["importance"] > 0
    assert c.shap is not None
    assert c.shap["method"] == "LinearExplainer"
    assert c.shap["features"], "SHAP 特征为空"
    assert "不能替代临床医生" in c.interpretation


def test_champion_tree_shap():
    df = _cls_df()
    p = profile_dataframe(df, target="结局")
    c = analyze_champion(df, p, "random_forest", metric_name="auc", metric_value=0.88)
    assert c.shap["method"] == "TreeExplainer"
    # 独热哑变量按原始特征聚合：feature 名都应是原始列名
    orig = set(p.feature_columns())
    assert set(f["variable"] for f in c.shap["features"]).issubset(orig)


def test_champion_categorical_onehot_aggregated():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "g0": rng.normal(size=n), "g1": rng.normal(size=n),
        "分组": rng.choice(["A", "B", "C"], n),
        "结局": rng.integers(0, 2, n),
    })
    p = profile_dataframe(df, target="结局")
    c = analyze_champion(df, p, "random_forest", top_k=15)
    # 类别列"分组"应聚合为一个特征，而不是拆成 A/B/C
    shp_vars = [f["variable"] for f in c.shap["features"]]
    assert "分组" in shp_vars
    assert not any(v.startswith("分组_") or v.startswith("分组A") for v in shp_vars)


def test_champion_regression():
    rng = np.random.default_rng(1)
    n = 120
    X = rng.normal(size=(n, 4))
    y = 2 * X[:, 0] + rng.normal(0, 0.5, n)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["y"] = y
    p = profile_dataframe(df, target="y")
    c = analyze_champion(df, p, "ridge", metric_name="rmse", metric_value=1.5)
    assert c.importance_top
    assert c.shap["method"] == "LinearExplainer"


def test_champion_survival():
    rng = np.random.default_rng(2)
    n = 100
    X = pd.DataFrame({"年龄": rng.normal(60, 10, n), "血红蛋白": rng.normal(130, 15, n)})
    lin = 0.05 * (X["年龄"] - 60)
    t = rng.weibull(1.5, n) * np.exp(-lin) * 200
    c = rng.exponential(300, n)
    ev = (t <= c).astype(int)
    X["随访时间"] = np.where(ev == 1, t, c)
    X["事件"] = ev
    p = profile_dataframe(X)
    ch = analyze_champion(X, p, "cox_ph", metric_name="cindex", metric_value=0.7)
    assert ch.model_key == "cox_ph"
    assert ch.importance_top
    assert ch.shap["method"] == "coefficients"
    assert "不做 SHAP" in ch.shap["warning"]


def test_interpret_contains_key_features():
    c = ChampionDetail("logistic", "逻辑回归", {}, "auc", 0.9,
                       [{"variable": "g0", "importance": 0.5, "raw": 1.0}],
                       {"method": "LinearExplainer", "warning": None, "n_sample": 10,
                        "features": [{"variable": "g0", "mean_abs_shap": 0.3}]},
                       "", 0.1)
    p = profile_dataframe(_cls_df(), target="结局")
    txt = interpret(c, p)
    assert "逻辑回归" in txt and "g0" in txt


# ------------------------------------------------------------------
# 报告
# ------------------------------------------------------------------
def test_report_disclaimer_and_structure():
    df = _cls_df()
    p = profile_dataframe(df, target="结局")
    rows = race_models(df, p, ["logistic"], n_splits=3)
    c = analyze_champion(df, p, "logistic", metric_name="auc",
                         metric_value=rows[0].metric_value)
    rep = build_report(profile=p, leaderboard=rows, champion=c,
                       leakage_result={"dropped": [], "kept": [], "note": ""})
    assert rep["disclaimer"] == DISCLAIMER
    assert rep["profile"]["task_type"] == TASK_CLASSIFICATION
    assert rep["leaderboard"][0]["model_key"] == "logistic"
    assert rep["champion"]["importance_top15"]
    assert "reproducibility" in rep and rep["reproducibility"]["random_state"] is not None
    assert any("外部验证" in l for l in rep["limitations"])


def test_json_safe_handles_nan_inf():
    obj = {"a": float("nan"), "b": float("inf"), "c": [1.0, float("-inf")],
           "d": {"x": np.nan}, "e": np.int64(3), "f": np.float32(2.5), "g": np.bool_(True)}
    out = json_safe(obj)
    assert out["a"] is None and out["b"] is None
    assert out["c"] == [1.0, None]
    assert out["d"]["x"] is None
    assert out["e"] == 3 and isinstance(out["e"], int)
    assert out["f"] == 2.5 and out["g"] is True


def test_json_safe_output_is_json_serializable():
    import json
    df = _cls_df()
    p = profile_dataframe(df, target="结局")
    rows = race_models(df, p, ["logistic"], n_splits=3)
    c = analyze_champion(df, p, "logistic", metric_name="auc", metric_value=0.9)
    rep = build_report(profile=p, leaderboard=rows, champion=c)
    json.dumps(rep, ensure_ascii=False)   # 不得抛异常（NaN/Inf 已安全化）


def test_run_manual_detects_and_drops_leakage():
    """手动模式此前完全跳过泄露检测；修复后应检测并按默认策略剔除泄露列，
    报告 leakage.findings/decision 均非空，且冠军特征不含泄露列。"""
    from automl.pipeline import run_manual
    df = _cls_df()
    df["结局副本"] = df["结局"]          # 明显泄露列
    rep = run_manual(df, model_key="logistic", target="结局", with_shap=False)
    assert rep["leakage"]["findings"], "手动模式未做泄露检测"
    decision = rep["leakage"]["decision"]
    assert decision and "结局副本" in decision["dropped"]
    if rep["champion"]:
        feats = {f["variable"] for f in rep["champion"]["importance_top15"]}
        assert "结局副本" not in feats
