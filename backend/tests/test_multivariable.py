# medical-data-analysis/backend/tests/test_multivariable.py
"""多因素分析（Logistic 回归 / 多元线性回归）与单变量筛选的单元测试。

正确性验证策略：用已知真实系数模拟数据，检验模型能否在容许误差内还原真实效应；
共线性、EPV、完全分离等边界情形验证警示与异常是否被正确触发。
"""

import numpy as np
import pandas as pd
import pytest

import multivariable
from config import MIN_EPV, VIF_THRESHOLD


def _sim_logistic(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.integers(0, 2, n)
    logit = -0.5 + 1.0 * x1 + 0.7 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _sim_linear(n=800, seed=6):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 3.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_pick_event_level_prefers_positive():
    """二分类结局：恰好一个"阳性类"水平时以它为事件，避免码点排序把 阴性 当事件。"""
    pick = multivariable._pick_event_level
    assert pick(["阴性", "阳性"]) == ("阴性", "阳性")
    assert pick(["0", "1"]) == ("0", "1")
    assert pick(["B", "M"]) == ("B", "M")          # malignant 为事件
    assert pick(["否", "是"]) == ("否", "是")
    # 无阳性类命中：保持排序行为（参照=前者，事件=后者），行为确定不变
    assert pick(["男", "女"]) == ("男", "女")
    assert pick(["甲", "乙"]) == ("甲", "乙")


def _term(res, name):
    return next(t for t in res["terms"] if t["term"] == name)


# ------------------------------------------------------------------
# Logistic 回归
# ------------------------------------------------------------------

def test_logistic_recovers_true_or():
    df = _sim_logistic()
    res = multivariable.run_multivariable(df, "y", ["x1", "x2"], [],
                                          {"x1": "numeric", "x2": "numeric", "y": "categorical"})
    assert res["task"] == "logistic"
    assert res["n_events"] == int(df["y"].sum())
    or1 = _term(res, "x1")["estimate"]
    or2 = _term(res, "x2")["estimate"]
    assert or1 == pytest.approx(np.exp(1.0), rel=0.15)   # 真实 OR = e^1
    assert or2 == pytest.approx(np.exp(0.7), rel=0.20)   # 真实 OR = e^0.7


def test_logistic_confidence_interval_brackets_estimate():
    df = _sim_logistic()
    res = multivariable.run_multivariable(df, "y", ["x1"], [],
                                          {"x1": "numeric", "y": "categorical"})
    row = _term(res, "x1")
    assert row["ci"][0] < row["estimate"] < row["ci"][1]
    assert row["estimate_label"] == "OR"


def test_logistic_outcome_reference_and_event_level():
    df = _sim_logistic()
    res = multivariable.run_multivariable(df, "y", ["x1"], [],
                                          {"x1": "numeric", "y": "categorical"})
    assert res["outcome_info"]["levels"] == ["0", "1"]
    assert res["outcome_info"]["reference"] == "0"
    assert res["outcome_info"]["event_level"] == "1"


def test_categorical_predictor_dummy_coding_reference():
    rng = np.random.default_rng(9)
    n = 600
    grade = rng.choice(["I", "II", "III"], n)
    effect = {"I": 0.0, "II": 0.8, "III": 1.6}
    logit = np.array([effect[g] for g in grade]) + rng.normal(0, 0.5, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({"grade": grade, "y": y})
    res = multivariable.run_multivariable(df, "y", ["grade"], [],
                                          {"grade": "categorical", "y": "categorical"})
    coding = res["coding"][0]
    assert coding["kind"] == "categorical"
    assert coding["reference"] == "I"                       # 参照组为排序后首个水平
    assert coding["columns"] == ["grade_II", "grade_III"]
    terms = {t["term"]: t for t in res["terms"]}
    assert terms["grade_II"]["level"] == "II"
    assert terms["grade_II"]["reference"] == "I"


def test_single_predictor_crude_equals_adjusted():
    """只有一个目标自变量时，单变量模型与多因素模型等价，校正前后应完全一致。"""
    df = _sim_logistic()
    res = multivariable.run_multivariable(df, "y", ["x1"], [],
                                          {"x1": "numeric", "y": "categorical"})
    assert len(res["terms"]) == 1
    row = res["terms"][0]
    assert row["crude_estimate"] == pytest.approx(row["estimate"], rel=1e-6)
    assert row["crude_p"] == pytest.approx(row["p"], rel=1e-6)
    assert res["confounding"] == []


def test_confounding_detected_when_adjustment_changes_estimate():
    """x 与结局无直接关联，但通过 z 混杂；校正后 x 的效应应被识别为改变。"""
    rng = np.random.default_rng(11)
    n = 4000
    z = rng.normal(0, 1, n)
    x = z + rng.normal(0, 1, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.0 * z))))
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    res = multivariable.run_multivariable(df, "y", ["x"], ["z"],
                                          {"x": "numeric", "z": "numeric", "y": "categorical"})
    assert any(c["variable"] == "x" for c in res["confounding"])


def test_logistic_requires_binary_outcome():
    df = pd.DataFrame({"y": ["a", "b", "c"] * 20, "x": np.random.default_rng(1).normal(size=60)})
    with pytest.raises(multivariable.ModelingError) as exc:
        multivariable.run_multivariable(df, "y", ["x"], [], {"x": "numeric", "y": "categorical"})
    assert "二分类结局" in str(exc.value)


# ------------------------------------------------------------------
# 多元线性回归
# ------------------------------------------------------------------

def test_logistic_matches_independent_statsmodels_reference():
    """与独立的 statsmodels 参照实现逐参数比对，证明本模块实现无偏差。"""
    import statsmodels.api as sm

    df = _sim_logistic()
    res = multivariable.run_multivariable(df, "y", ["x1", "x2"], [],
                                          {"x1": "numeric", "x2": "numeric", "y": "categorical"})
    X = sm.add_constant(df[["x1", "x2"]].astype(float))
    ref = sm.Logit(df["y"].astype(int), X).fit(disp=0)
    ref_ci = ref.conf_int(alpha=0.05)
    for term in res["terms"]:
        name = term["term"]
        assert term["estimate"] == pytest.approx(float(np.exp(ref.params[name])), rel=1e-4)
        assert term["p"] == pytest.approx(float(ref.pvalues[name]), abs=1e-5)
        assert term["ci"][0] == pytest.approx(float(np.exp(ref_ci.loc[name, 0])), rel=1e-4)
        assert term["ci"][1] == pytest.approx(float(np.exp(ref_ci.loc[name, 1])), rel=1e-4)


def test_linear_matches_independent_statsmodels_reference():
    import statsmodels.api as sm

    df = _sim_linear()
    res = multivariable.run_multivariable(df, "y", ["x1", "x2"], [],
                                          {"x1": "numeric", "x2": "numeric", "y": "numeric"})
    X = sm.add_constant(df[["x1", "x2"]].astype(float))
    ref = sm.OLS(df["y"].astype(float), X).fit()
    for term in res["terms"]:
        name = term["term"]
        assert term["estimate"] == pytest.approx(float(ref.params[name]), abs=1e-4)
        assert term["p"] == pytest.approx(float(ref.pvalues[name]), abs=1e-6)


def test_linear_recovers_true_coefficients():
    df = _sim_linear()
    res = multivariable.run_multivariable(df, "y", ["x1", "x2"], [],
                                          {"x1": "numeric", "x2": "numeric", "y": "numeric"})
    assert res["task"] == "linear"
    assert _term(res, "x1")["estimate"] == pytest.approx(2.0, abs=0.1)
    assert _term(res, "x2")["estimate"] == pytest.approx(-1.5, abs=0.1)
    assert res["model_summary"]["r_squared"] > 0.9


# ------------------------------------------------------------------
# 边界与风险提示
# ------------------------------------------------------------------

def test_vif_warning_on_collinearity():
    rng = np.random.default_rng(21)
    n = 500
    x1 = rng.normal(0, 1, n)
    x3 = x1 + rng.normal(0, 0.001, n)          # 与 x1 几乎完全共线
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1))))
    df = pd.DataFrame({"x1": x1, "x3": x3, "y": y})
    res = multivariable.run_multivariable(df, "y", ["x1", "x3"], [],
                                          {"x1": "numeric", "x3": "numeric", "y": "categorical"})
    high = [v for v in res["vif"] if v["vif"] and v["vif"] > VIF_THRESHOLD]
    assert high, "高度共线变量应产生 VIF>10"
    assert any("VIF" in w for w in res["warnings"])


def test_vif_matches_manual_definition_with_intercept():
    """VIF = 1/(1-R²_j)，R²_j 来自"该变量对其余变量 + 截距"的辅助回归。

    CODE_REVIEW 曾建议从设计矩阵里删掉 const 再算 VIF；实测 statsmodels 的
    `variance_inflation_factor` 在两种入参下结果一致，且与含截距的手工 R² 完全相同。
    真正会算错的是**无截距**辅助回归（VIF 偏小），因此保留含 const 的调用方式。
    """
    import statsmodels.api as sm
    rng = np.random.default_rng(31)
    n = 300
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(0, 0.25, n)
    x3 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3,
                       "y": (rng.random(n) < 0.3).astype(int)})
    res = multivariable.run_multivariable(df, "y", ["x1", "x2", "x3"], [],
                                          {"x1": "numeric", "x2": "numeric",
                                           "x3": "numeric", "y": "categorical"})
    vif = {v["term"]: v["vif"] for v in res["vif"]}
    r2 = sm.OLS(df["x1"], sm.add_constant(df[["x2", "x3"]])).fit().rsquared
    assert vif["x1"] == pytest.approx(1.0 / (1.0 - r2), rel=0.02)
    assert vif["x3"] == pytest.approx(1.0, abs=0.05)      # 与其他变量无关


def test_epv_warning_on_rare_events():
    rng = np.random.default_rng(22)
    n = 80
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    logit = -3.0 + 0.5 * x + 0.3 * z + 0.2 * w
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({"x": x, "z": z, "w": w, "y": y})
    res = multivariable.run_multivariable(df, "y", ["x", "z", "w"], [],
                                          {k: "numeric" for k in ["x", "z", "w"]} | {"y": "categorical"})
    assert res["epv"] is not None and res["epv"] < MIN_EPV
    assert any("EPV" in w for w in res["warnings"])


def test_perfect_separation_handled_gracefully():
    rng = np.random.default_rng(23)
    x = rng.normal(0, 1, 60)
    y = (x > 0).astype(int)
    df = pd.DataFrame({"x": x, "y": y})
    try:
        res = multivariable.run_multivariable(df, "y", ["x"], [], {"x": "numeric", "y": "categorical"})
    except multivariable.ModelingError as exc:
        assert "完全分离" in str(exc.value)
        return
    # 若 statsmodels 未抛错，则必须给出未收敛警示，且不得返回发散的效应估计
    assert any("未收敛" in w for w in res["warnings"]), "完全分离情形下必须给出警示而非静默返回"
    assert res["terms"][0]["estimate"] is None, "发散的 OR 不应直接返回给用户"


def test_perfect_collinearity_reports_actionable_error():
    """完全共线变量必须在拟合前被识别，并给出可操作的变量名。

    回归：旧实现把 statsmodels 的 'Singular matrix' 原样抛给用户，无法判断该删哪个变量。
    """
    rng = np.random.default_rng(24)
    n = 120
    x1 = rng.normal(0, 1, n)
    x2 = 2.0 * x1                                   # 与 x1 完全共线
    y = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1)))
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    with pytest.raises(multivariable.ModelingError) as exc:
        multivariable.run_multivariable(
            df, "y", ["x1", "x2"], [],
            {"x1": "numeric", "x2": "numeric", "y": "categorical"})
    msg = str(exc.value)
    assert "完全共线" in msg and "x1" in msg and "x2" in msg
    assert "Singular matrix" not in msg


def test_constant_predictor_reports_actionable_error():
    """常量自变量必须被明确指出，而不是在拟合阶段报奇异矩阵。"""
    rng = np.random.default_rng(25)
    n = 120
    x1 = rng.normal(0, 1, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1)))
    df = pd.DataFrame({"x1": x1, "const_x": 5.0, "y": y})
    with pytest.raises(multivariable.ModelingError) as exc:
        multivariable.run_multivariable(
            df, "y", ["x1", "const_x"], [],
            {"x1": "numeric", "const_x": "numeric", "y": "categorical"})
    msg = str(exc.value)
    assert "常量变量" in msg and "const_x" in msg
    assert "Singular matrix" not in msg


def test_design_rank_check_is_scale_invariant():
    """量纲差异极大的两个独立变量不得被误判为完全共线。

    matrix_rank 的默认容差是绝对尺度相关的；未归一化时，1e-12 量级的
    独立变量可能被判为 0 方差/共线，导致合法模型被拒绝。
    """
    rng = np.random.default_rng(26)
    n = 200
    X = pd.DataFrame({
        "big_scale": rng.normal(0, 1, n),
        "tiny_scale": rng.normal(0, 1e-12, n),
    })
    coding = [{"variable": c, "columns": [c]} for c in X.columns]
    multivariable._check_design_rank(X, coding)   # 不应抛 ModelingError


def test_design_columns_not_less_than_samples_rejected():
    """设计列数 ≥ 样本量时应给出“参数多于观测”的明确原因，而不是共线。"""
    rng = np.random.default_rng(27)
    n = 20
    cols = {f"x{i}": rng.normal(0, 1, n) for i in range(25)}
    df = pd.DataFrame({**cols, "y": rng.integers(0, 2, n)})
    types = {c: "numeric" for c in cols} | {"y": "categorical"}
    with pytest.raises(multivariable.ModelingError) as exc:
        multivariable.run_multivariable(df, "y", list(cols), [], types)
    msg = str(exc.value)
    assert "不少于完整病例样本量" in msg
    assert "完全共线" not in msg


def test_missing_predictor_raises():
    df = _sim_logistic()
    with pytest.raises(multivariable.ModelingError):
        multivariable.run_multivariable(df, "y", [], [], {"y": "categorical"})
    with pytest.raises(multivariable.ModelingError):
        multivariable.run_multivariable(df, "y", ["not_exist"], [], {"y": "categorical"})


# ------------------------------------------------------------------
# 单变量筛选与 FDR
# ------------------------------------------------------------------

def test_screen_univariate_returns_fdr_and_ranks(medical_df):
    import validators
    schema = validators.infer_variable_types(medical_df)
    types = {v["name"]: v["confirmed_type"] for v in schema if not v["excluded"]}
    candidates = [v["name"] for v in schema if not v["excluded"] and v["name"] != "结局"]
    res = multivariable.screen_univariate(medical_df, "结局", candidates, types)
    assert res["n_tests"] > 0
    assert res["fdr_method"] == "Benjamini-Hochberg"
    p_fdrs = [r["p_fdr"] for r in res["rows"]]
    assert p_fdrs == sorted(p_fdrs)
    for row in res["rows"]:
        assert row["p_fdr"] >= row["p"] - 1e-9      # 校正后 p 值不小于原始 p 值
    glucose = next(r for r in res["rows"] if r["variable"] == "空腹血糖")
    assert glucose["significant_fdr"] is True       # 结局由血糖生成，应被识别为显著


def test_screen_univariate_note_warns_about_feature_selection(medical_df):
    import validators
    schema = validators.infer_variable_types(medical_df)
    types = {v["name"]: v["confirmed_type"] for v in schema if not v["excluded"]}
    res = multivariable.screen_univariate(medical_df, "结局", ["年龄", "空腹血糖"], types)
    assert "交叉验证" in res["note"]
