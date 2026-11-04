# medical-data-analysis/backend/tests/test_stat_integrity.py
"""统计完整性回归测试：p 值精度、FDR 正确性与结局变量护栏。

背景（2026-10 WDBC 事故：用户报告"表 1 / 表 2 全部算错"）：两处缺陷叠加——
1. p 值在进入 BH-FDR 之前被 round(·, 3) 截断：真实 p 在 1e-6~1e-78 量级的特征
   全部塌缩成 0.0 并列，FDR 校正与重要性排序失去区分度，重要性分级与真实
   区分力完全颠倒，且平台永远显示不出 1.57e-6 这类数值；
2. 分组/结局变量无类型护栏：连续变量被当成 N 个"单样本组"跑 Kruskal-Wallis，
   η² 必然趋近 1、p 值退化为随机数，整张筛选表静默输出垃圾结果。

本文件逐条锁死修复行为（WDBC scipy 直算基线：radius_worst p=1.136e-78，
30 特征 BH 后 27 个显著，不显著为 smoothness_se / texture_se / fractal_dimension_mean）。
"""

import io
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats as sps

import analyzer
import multivariable
import app as app_module
from config import ALPHA, MIN_AVG_GROUP_SIZE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DIR = os.path.join(ROOT, "sample_data")


@pytest.fixture
def client():
    return TestClient(app_module.app)


def _upload(client, df, filename):
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    r = client.post("/api/dataset/upload",
                    files={"file": (filename, buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


# ==================================================================
# 一、p 值精度：检验 → FDR 全链路保持完整精度
# ==================================================================

def test_fdr_correct_preserves_tiny_p():
    """1e-78 量级的 p 在 BH 校正后仍不得塌缩成 0.0（修复前 round(·,6) 必塌缩）。"""
    pvals = [1.57e-6, 1.136e-78, 0.5]
    res = analyzer.fdr_correct(pvals)
    assert res["p"][1] == pytest.approx(1.136e-78, rel=1e-12)
    assert res["p_fdr"][1] > 0, "极小 p 校正后必须仍大于 0"
    assert res["p_fdr"][1] <= 1.136e-76, "校正后 p 与原 p 同量级（BH×n_tests）"
    assert res["significant"][1] is True
    assert res["p_fdr"][2] >= res["p"][2] - 1e-12, "校正只会把 p 改大"


def test_fdr_correct_handles_non_finite_p_values():
    """None/NaN/Inf 按 p=1 计入 BH 分母，但不返回 p_fdr/显著性。

    回归：相关矩阵遇到常量列会产生 NaN p；旧实现把 None 直接交给
    multipletests，触发 TypeError（'/api/analysis/correlation' 500）。
    """
    from statsmodels.stats.multitest import multipletests

    pvals = [0.01, None, float("nan"), float("inf"), 0.5]
    res = analyzer.fdr_correct(pvals)
    assert len(res["p"]) == len(res["p_fdr"]) == len(res["significant"]) == len(pvals)
    assert [res["p"][i] for i in (1, 2, 3)] == [None, None, None]
    assert [res["p_fdr"][i] for i in (1, 2, 3)] == [None, None, None]
    assert [res["significant"][i] for i in (1, 2, 3)] == [False, False, False]
    # 无定义检验按 p=1 参与校正：m=5，而不是只用 2 个有效 p
    _, corrected, _, _ = multipletests([0.01, 1.0, 1.0, 1.0, 0.5],
                                        alpha=0.05, method="fdr_bh")
    assert res["p_fdr"][0] == pytest.approx(corrected[0], rel=1e-12)
    assert res["p_fdr"][4] == pytest.approx(corrected[4], rel=1e-12)


def test_fdr_invalid_tests_increase_conservativeness():
    """无定义检验按 p=1 计入分母后，有效检验的校正 p 应更大（更保守）。"""
    with_invalid = analyzer.fdr_correct([0.01, None, 0.5])
    valid_only = analyzer.fdr_correct([0.01, 0.5])
    assert with_invalid["p_fdr"][0] > valid_only["p_fdr"][0]
    assert with_invalid["p_fdr"][2] > valid_only["p_fdr"][1]


def test_compare_groups_p_full_precision_matches_scipy():
    """组间比较返回的 p 必须与 scipy 直算值一致（修复前 round(·,3) 必不一致）。"""
    rng = np.random.default_rng(12)
    a, b = rng.exponential(1.0, 45), rng.exponential(3.0, 45)
    df = pd.DataFrame({"group": ["A"] * 45 + ["B"] * 45, "val": np.r_[a, b]})
    res = analyzer.compare_groups(df, "val", "group", "numeric")
    _, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    assert res["test"]["p"] == pytest.approx(p, rel=1e-12), \
        "p 值必须保留完整精度，任何 round 截断都会让该断言失败"


def test_compare_groups_extreme_p_not_collapsed():
    """完全分离数据的极小 p（1e-60 量级）不得被压成 0.0。"""
    rng = np.random.default_rng(13)
    a = rng.normal(0, 1, 200)
    b = a + 10.0
    df = pd.DataFrame({"group": ["A"] * 200 + ["B"] * 200, "val": np.r_[a, b]})
    res = analyzer.compare_groups(df, "val", "group", "numeric")
    assert res["test"]["p"] is not None
    assert 0 < res["test"]["p"] < 1e-30


def test_normality_p_full_precision():
    rng = np.random.default_rng(3)
    arr = rng.normal(0, 1, 200)
    res = analyzer.normality_test(arr)
    _, p = sps.shapiro(arr)
    assert res["p"] == pytest.approx(p, rel=1e-12)


def test_correlation_matrix_full_precision_p():
    """相关矩阵的 p 与 p_fdr 保持完整精度；FDR 基于未舍入值。

    数据强度校准为 r≈0.89、p≈7e-104：足够小以验证未被 round(·,6) 截断，
    又不至于让 scipy 数值下溢成真实的 0.0。
    """
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 300)
    y = 2.0 * x + rng.normal(0, 1.0, 300)
    res = analyzer.correlation_matrix(pd.DataFrame({"x": x, "y": y}), ["x", "y"])
    fn = sps.pearsonr if res["method"][0][1] == "pearson" else sps.spearmanr
    _, p = fn(x, y)
    assert res["p"][0][1] == pytest.approx(p, rel=1e-12)
    assert 0 < res["p"][0][1] < 1e-10, "强相关的 p 应远小于 0.001 且未被截断"
    assert res["p_fdr"][0][1] >= res["p"][0][1] - 1e-15


def test_multivariable_term_p_full_precision():
    """多因素回归的项级 p 保持完整精度（修复前 round(·,6) 会把 1e-30 压成 0.0）。"""
    import statsmodels.api as sm
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(0, 1, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(2.0 * x))))
    df = pd.DataFrame({"x": x, "y": y})
    res = multivariable.run_multivariable(df, "y", ["x"], [],
                                         {"x": "numeric", "y": "categorical"})
    ref = sm.Logit(df["y"].astype(int), sm.add_constant(df[["x"]].astype(float))).fit(disp=0)
    term = res["terms"][0]
    assert 0 < term["p"] < 1e-6
    assert term["p"] == pytest.approx(float(ref.pvalues["x"]), rel=1e-9)
    assert term["crude_p"] == pytest.approx(float(ref.pvalues["x"]), rel=1e-9)


def test_hosmer_lemeshow_p_full_precision():
    """HL 检验 p 保持完整精度（前端依据 p>0.05 判断校准好坏，round 会在边界翻转结论）。"""
    from ml_engine import hosmer_lemeshow
    rng = np.random.default_rng(2024)
    n = 200
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.3 + 1.4 * x)))
    y = rng.binomial(1, p)
    res = hosmer_lemeshow(y, p, groups=10)
    expected = float(sps.chi2.sf(res["statistic"], res["dof"]))
    # 统计量本身保留 4 位舍入，这里用宽松容差核对 p 未被 round(·,4) 截断
    assert res["p"] == pytest.approx(expected, rel=0.02)
    assert res["p"] > 0


# ==================================================================
# 二、结局（分组）变量护栏：错误输入必须明确拒绝，不得静默输出垃圾
# ==================================================================

def test_compare_groups_rejects_singleton_groups():
    """连续变量被误用作分组（每组≈1 个样本）：valid=False 且说明原因（修复前会算出 η²≈1 的垃圾）。"""
    rng = np.random.default_rng(15)
    n = 120
    g = rng.normal(0, 1, n)          # 120 个近似唯一的"组"，平均每组 1 例
    df = pd.DataFrame({"group": g, "val": rng.normal(0, 1, n)})
    res = analyzer.compare_groups(df, "val", "group", "numeric")
    assert res["valid"] is False
    assert "分组变量" in res["note"] and "平均每组" in res["note"]


def test_compare_groups_allows_multi_center_design():
    """正例：30 家医院 × 每中心 12 例的多中心设计是正当用法，护栏不得误伤。"""
    rng = np.random.default_rng(15)
    centers = np.repeat([f"中心{i:02d}" for i in range(30)], 12)
    val = rng.normal(10, 2, len(centers))
    df = pd.DataFrame({"center": centers, "val": val})
    res = analyzer.compare_groups(df, "val", "center", "numeric")
    assert res["valid"] is True
    assert len(res["group_levels"]) == 30


def test_screen_rejects_numeric_outcome():
    """数值型结局 → ModelingError（接口层转 422），绝不静默计算。"""
    rng = np.random.default_rng(16)
    n = 100
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "x": rng.normal(0, 1, n)})
    with pytest.raises(multivariable.ModelingError) as exc:
        multivariable.screen_univariate(df, "y", ["x"], {"y": "numeric", "x": "numeric"})
    assert "数值型" in str(exc.value)


def test_screen_rejects_unconfirmed_outcome_type():
    """结局类型缺失（如被排除的字段）→ 明确报错。"""
    rng = np.random.default_rng(17)
    df = pd.DataFrame({"y": rng.normal(0, 1, 50), "x": rng.normal(0, 1, 50)})
    with pytest.raises(multivariable.ModelingError) as exc:
        multivariable.screen_univariate(df, "y", ["x"], {"x": "numeric"})
    assert "分类变量" in str(exc.value)


def test_screen_rejects_singleton_level_categorical_outcome():
    """分类结局但每组≈1 例（连续变量误用）→ 明确报错，绝不静默输出垃圾。"""
    rng = np.random.default_rng(18)
    n = 60
    g = np.array([f"L{i}" for i in range(n)])   # 60 个水平 × 每组 1 例
    df = pd.DataFrame({"g": g, "x": rng.normal(0, 1, n)})
    with pytest.raises(multivariable.ModelingError) as exc:
        multivariable.screen_univariate(df, "g", ["x"], {"g": "categorical", "x": "numeric"})
    assert "平均每组" in str(exc.value)


def test_screen_allows_multi_center_outcome():
    """正例：25 级分类结局 × 每组 12 例（多中心）可正常筛选，护栏不得误伤。"""
    rng = np.random.default_rng(18)
    g = np.repeat([f"中心{i}" for i in range(25)], 12)
    # 组间均值逐级递增，保证 Kruskal-Wallis 显著（否则只是"能跑"而非"结果正确"）
    x = np.tile(np.arange(25, dtype=float) * 2.0, 12) + rng.normal(0, 1.0, len(g))
    df = pd.DataFrame({"g": g, "x": x})
    res = multivariable.screen_univariate(df, "g", ["x"], {"g": "categorical", "x": "numeric"})
    assert res["n_tests"] == 1
    assert res["rows"][0]["significant_fdr"] is True


def test_screen_rejects_missing_outcome():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(multivariable.ModelingError) as exc:
        multivariable.screen_univariate(df, "nope", ["x"], {"x": "numeric"})
    assert "不存在" in str(exc.value)


def test_screen_accepts_multilevel_categorical_outcome():
    """正例：3 水平分类结局合法（Kruskal-Wallis 是正当用法），护栏不得误伤。"""
    rng = np.random.default_rng(19)
    n = 150
    g = rng.choice(["A", "B", "C"], n)
    val = np.where(g == "A", 10.0, np.where(g == "B", 15.0, 20.0)) + rng.normal(0, 2, n)
    df = pd.DataFrame({"g": g, "val": val, "noise": rng.normal(0, 1, n)})
    res = multivariable.screen_univariate(df, "g", ["val", "noise"],
                                          {"g": "categorical", "val": "numeric",
                                           "noise": "numeric"})
    assert res["n_tests"] == 2
    val_row = next(r for r in res["rows"] if r["variable"] == "val")
    assert val_row["significant_fdr"] is True
    assert val_row["p_fdr"] < ALPHA


def test_multivariable_dedupes_duplicate_predictors():
    """同一预测变量重复传入时去重，设计矩阵不得出现重复列。"""
    rng = np.random.default_rng(21)
    n = 120
    df = pd.DataFrame({
        "y": rng.integers(0, 2, n),
        "x": rng.normal(0, 1, n),
        "z": rng.normal(0, 1, n),
    })
    res = multivariable.run_multivariable(df, "y", ["x", "x", "z"], [],
                                          {"y": "categorical", "x": "numeric", "z": "numeric"},
                                          task="logistic")
    assert res["predictors"] == ["x", "z"]
    assert {t["term"] for t in res["terms"]} == {"x", "z"}


def test_multivariable_raises_on_dummy_name_collision():
    """分类变量的类别名与另一变量名冲突 → 明确报错而非静默出错。"""
    rng = np.random.default_rng(22)
    n = 120
    grp = np.where(rng.random(n) < 0.5, "x", "w")
    df = pd.DataFrame({
        "y": rng.integers(0, 2, n),
        "grp": grp,                    # 哑变量编码后产生列「grp_x」
        "grp_x": rng.normal(0, 1, n),  # 与哑变量列重名
    })
    with pytest.raises(multivariable.ModelingError) as exc:
        multivariable.run_multivariable(df, "y", ["grp", "grp_x"], [],
                                        {"y": "categorical", "grp": "categorical",
                                         "grp_x": "numeric"}, task="logistic")
    assert "重名" in str(exc.value)


# ==================================================================
# 三、WDBC 真实基线：与 scipy 直算完全一致（事故数据的逐项锁定）
# ==================================================================

@pytest.fixture(scope="module")
def wdbc_df():
    path = os.path.join(SAMPLE_DIR, "wdbc.csv")
    if not os.path.exists(path):
        pytest.skip("sample_data/wdbc.csv 不存在")
    return pd.read_csv(path)


def test_screen_wdbc_full_precision_ranking_and_baseline(wdbc_df):
    """对仓库 WDBC 全 30 特征筛选：p 值精度、排序、显著集与 scipy 直算基线一致。

    修复前的行为：p 全部被 round 成 0.0 并列 → 重要性分级与真实区分力颠倒；
    且若误选连续结局会输出 η²≈0.99 的垃圾行。
    """
    import validators
    schema = validators.infer_variable_types(wdbc_df)
    types = {v["name"]: v["confirmed_type"] for v in schema if not v.get("excluded")}
    feats = [c for c in wdbc_df.columns if c not in ("id", "diagnosis")]
    assert len(feats) == 30

    res = multivariable.screen_univariate(wdbc_df, "diagnosis", feats, types)
    rows = res["rows"]
    assert res["n_tests"] == 30 and res["errors"] == []

    # p 保持完整精度：全部严格大于 0（修复前会被 round 塌缩成 0.0 并列）
    assert all(r["p"] > 0 for r in rows)

    # radius_worst：用户手工核对过的锚点值 1.13×10⁻⁷⁸
    rw = next(r for r in rows if r["variable"] == "radius_worst")
    assert rw["p"] == pytest.approx(1.1356300904893913e-78, rel=1e-6)
    assert rw["significant_fdr"] is True

    # BH 校正后 p 不小于原始 p；行按 p_fdr 升序排列
    for r in rows:
        assert r["p_fdr"] >= r["p"] - 1e-300
    assert [r["p_fdr"] for r in rows] == sorted(r["p_fdr"] for r in rows)

    # 显著集与 scipy 直算基线一致：27/30，不显著为 smoothness_se / texture_se / fractal_dimension_mean
    nonsig = {r["variable"] for r in rows if not r["significant_fdr"]}
    assert len(nonsig) == 3
    assert nonsig == {"error_smoothness", "fractal_dimension_mean", "error_texture"}

    # 与独立 scipy Mann-Whitney 路径逐特征比对（修复前后差异的最终裁决）
    b = wdbc_df.loc[wdbc_df["diagnosis"] == "B", "radius_worst"]
    m = wdbc_df.loc[wdbc_df["diagnosis"] == "M", "radius_worst"]
    _, p_ref = sps.mannwhitneyu(b, m, alternative="two-sided")
    assert rw["p"] == pytest.approx(float(p_ref), rel=1e-9)


# ==================================================================
# 四、接口层：护栏以 422 明确报错，响应体携带面向用户的原因
# ==================================================================

def test_screen_endpoint_rejects_numeric_outcome(client):
    """HTTP 层：数值型结局 → 422 + 明确 message（修复前静默返回垃圾表）。"""
    rng = np.random.default_rng(20)
    n = 100
    df = pd.DataFrame({"年龄": rng.normal(55, 10, n).round(1),
                       "x1": rng.normal(0, 1, n).round(2),
                       "分组": rng.choice(["A", "B"], n)})
    did = _upload(client, df, "numeric_outcome.csv")
    r = client.post("/api/analysis/screen",
                    json={"dataset_id": did, "outcome": "年龄",
                          "candidates": ["x1", "分组"]})
    assert r.status_code == 422, r.text
    body = r.json()
    assert {"code", "message", "detail"} <= set(body)
    assert "数值型" in body["message"]


def test_univariate_endpoint_rejects_numeric_group(client):
    """HTTP 层：数值型分组 → 422 + 明确 message（修复前算出 η²≈0.99 的废表）。"""
    rng = np.random.default_rng(21)
    n = 100
    df = pd.DataFrame({"年龄": rng.normal(55, 10, n).round(1),
                       "x1": rng.normal(0, 1, n).round(2),
                       "分组": rng.choice(["A", "B"], n)})
    did = _upload(client, df, "numeric_group.csv")
    r = client.post("/api/analysis/univariate",
                    json={"dataset_id": did, "target": "x1", "group": "年龄",
                          "target_type": "numeric"})
    assert r.status_code == 422, r.text
    assert "数值型" in r.json()["message"]


def test_univariate_endpoint_accepts_categorical_group(client):
    """正例：分类分组不受护栏影响，照常 200。"""
    rng = np.random.default_rng(22)
    n = 100
    df = pd.DataFrame({"x1": rng.normal(0, 1, n).round(2),
                       "分组": rng.choice(["A", "B"], n)})
    did = _upload(client, df, "cat_group.csv")
    r = client.post("/api/analysis/univariate",
                    json={"dataset_id": did, "target": "x1", "group": "分组",
                          "target_type": "numeric"})
    assert r.status_code == 200, r.text
    assert r.json()["valid"] is True


def test_screen_endpoint_wdbc_tiny_p_survives_json(client, wdbc_df):
    """HTTP 全链路：1e-78 量级的 p 在 JSON 响应体中无损往返（序列化不崩溃、不丢精度）。"""
    did = _upload(client, wdbc_df, "wdbc_integrity.csv")
    r = client.post(f"/api/dataset/{did}/confirm", json={"types": {}})
    assert r.status_code == 200, r.text
    feats = [c for c in wdbc_df.columns if c not in ("id", "diagnosis")]
    r = client.post("/api/analysis/screen",
                    json={"dataset_id": did, "outcome": "diagnosis", "candidates": feats})
    assert r.status_code == 200, r.text
    rows = r.json()["rows"]
    rw = next(x for x in rows if x["variable"] == "radius_worst")
    assert rw["p"] == pytest.approx(1.1356300904893913e-78, rel=1e-6)
    assert rw["significant_fdr"] is True
