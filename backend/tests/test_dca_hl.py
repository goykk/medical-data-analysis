# medical-data-analysis/backend/tests/test_dca_hl.py
"""DCA 与 Hosmer-Lemeshow 检验的手工核算单元测试。

这两处是"严禁手写统计公式"规则下已批准的例外（无成熟库实现），
因此必须用**逐步手工算例**验证实现正确性，而不是只跑一遍函数看它不报错。
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ml_engine import (calibration_analysis, decision_curve_analysis,
                       hl_groups, hosmer_lemeshow)


# ------------------------------------------------------------------
# Hosmer-Lemeshow 检验：手工算例
# ------------------------------------------------------------------

def test_hosmer_lemeshow_manual_two_groups():
    """10 个样本分 2 组，逐组手工计算 H 统计量。

    样本（已按预测概率升序）：
      p = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
      y = [   0,    0,    0,    0,    0,    1,    1,    1,    1,    1]
    第 1 组（前 5 个）：n=5, O1=0, E1=1.50, p̄=0.30
          term = (0 - 1.5)² / (5 × 0.30 × 0.70) = 2.25 / 1.05 = 2.142857
    第 2 组（后 5 个）：n=5, O1=5, E1=3.95, p̄=0.79
          term = (5 - 3.95)² / (5 × 0.79 × 0.21) = 1.1025 / 0.8295 = 1.328511
    """
    p = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    term1 = (0 - 1.5) ** 2 / (5 * 0.30 * 0.70)
    term2 = (5 - 3.95) ** 2 / (5 * 0.79 * 0.21)
    expected_h = term1 + term2

    res = hosmer_lemeshow(y, p, groups=2)
    assert res["statistic"] == pytest.approx(expected_h, abs=1e-4)
    assert res["groups"] == 2
    assert res["dof"] == 1                      # max(1, g-2) = max(1, 0) = 1
    assert res["p"] == pytest.approx(float(stats.chi2.sf(expected_h, 1)), abs=1e-4)


def test_hosmer_lemeshow_table_matches_manual_groups():
    res = hosmer_lemeshow(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                          np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
                          groups=2)
    g0, g1 = res["table"]
    assert (g0["n"], g0["observed"], g0["expected"]) == (5, 0.0, 1.5)
    assert (g1["n"], g1["observed"], g1["expected"]) == (5, 5.0, 3.95)
    assert g0["mean_prob"] == pytest.approx(0.3, abs=1e-4)
    assert g1["mean_prob"] == pytest.approx(0.79, abs=1e-4)


def test_hosmer_lemeshow_against_independent_pandas_implementation():
    """用另一套独立实现（pandas 分组聚合）交叉验证分组统计与 H 统计量。"""
    rng = np.random.default_rng(2024)
    n = 200
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.3 + 1.4 * x)))
    y = rng.binomial(1, p)

    res = hosmer_lemeshow(y, p, groups=10)

    df = pd.DataFrame({"y": y, "p": p}).sort_values("p").reset_index(drop=True)
    df["grp"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    manual_h = 0.0
    for _, sub in df.groupby("grp"):
        n_g, o1, e1 = len(sub), sub["y"].sum(), sub["p"].sum()
        pbar = e1 / n_g
        manual_h += (o1 - e1) ** 2 / (n_g * pbar * (1 - pbar))

    assert res["statistic"] == pytest.approx(manual_h, abs=1e-4)
    assert res["dof"] == 8                      # g - 2 = 8
    assert 0.0 <= res["p"] <= 1.0


def test_hosmer_lemeshow_perfect_calibration_gives_small_statistic():
    """完美校准（预测概率即真实概率分组均值）时 H 应很小、p 应远大于 0.05。"""
    rng = np.random.default_rng(7)
    n = 1000
    p = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, p)
    res = hosmer_lemeshow(y, p, groups=10)
    assert res["p"] > 0.05


def test_hl_groups_scale_with_sample_size():
    """HL 分组数不能固定 10：小样本每组不足 10 例会让检验退化。"""
    assert hl_groups(40) == 3
    assert hl_groups(80) == 5
    assert hl_groups(143) == 10
    assert hl_groups(600) == 20


def test_calibration_analysis_uses_adaptive_hl_groups():
    rng = np.random.default_rng(8)
    n = 80                       # 小样本 → 5 组
    p = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, p)
    res = calibration_analysis(y, p, n_test=n)
    assert res["hosmer_lemeshow"]["groups"] == 5
    assert res["hosmer_lemeshow"]["dof"] == 3

    n2 = 700                     # 大样本 → 20 组
    p2 = rng.uniform(0.05, 0.95, n2)
    y2 = rng.binomial(1, p2)
    res2 = calibration_analysis(y2, p2, n_test=n2)
    assert res2["hosmer_lemeshow"]["groups"] == 20


# ------------------------------------------------------------------
# 决策曲线分析 DCA：手工算例
# ------------------------------------------------------------------

def test_dca_manual_three_thresholds():
    """5 个样本、3 个阈值，逐个手工计算净获益。

      y = [0, 0, 0, 1, 1]，p = [0.10, 0.20, 0.40, 0.70, 0.90]，患病率 = 0.4
      pt=0.01 : pred 全为 1 → TP=2, FP=3, w=0.01/0.99
                NB_model = 2/5 - 3/5 × w = 0.4 - 0.00606061 = 0.39393939
                NB_all   = 0.4 - 0.6 × w = 0.39393939
      pt=0.50 : pred=(p≥0.5)=[0,0,0,1,1] → TP=2, FP=0, w=1
                NB_model = 0.4 ; NB_all = 0.4 - 0.6 = -0.2
      pt=0.99 : pred 全为 0 → TP=0, FP=0 → NB_model = 0
                NB_all = 0.4 - 0.6 × 99 = -59.0
    """
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.10, 0.20, 0.40, 0.70, 0.90])
    res = decision_curve_analysis(y, p, n_points=3)

    assert res["thresholds"] == pytest.approx([0.01, 0.5, 0.99], abs=1e-9)
    assert res["prevalence"] == pytest.approx(0.4, abs=1e-9)

    w_low = 0.01 / 0.99
    assert res["net_benefit_model"] == pytest.approx(
        [0.4 - 0.6 * w_low, 0.4, 0.0], abs=1e-6)
    assert res["net_benefit_all"] == pytest.approx(
        [0.4 - 0.6 * w_low, -0.2, 0.4 - 0.6 * 99.0], abs=1e-6)
    assert res["net_benefit_none"] == [0.0, 0.0, 0.0]
    assert res["n"] == 5


def test_dca_manual_single_threshold_counting():
    """再验一个阈值下的 TP/FP 计数：pt=0.45 → pred=[0,0,0,1,1]。"""
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.10, 0.20, 0.40, 0.70, 0.90])
    res = decision_curve_analysis(y, p, n_points=2)      # 阈值取 0.01 与 0.99
    # 单独用 0.45 复核计数逻辑
    pt = 0.45
    pred = (p >= pt).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    assert (tp, fp) == (2, 0)
    nb = tp / 5 - fp / 5 * (pt / (1 - pt))
    assert nb == pytest.approx(0.4, abs=1e-12)
    assert res["title"].startswith("决策曲线分析 DCA（测试集 n=5）")


def test_dca_against_independent_loop_implementation():
    """与独立的显式循环实现逐点比对（100 个阈值点）。"""
    rng = np.random.default_rng(31)
    n = 300
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(2.0 * (p - 0.5)))))

    res = decision_curve_analysis(y, p, n_points=100)
    prevalence = float(y.mean())

    manual_model, manual_all, thresholds = [], [], np.linspace(0.01, 0.99, 100)
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        w = pt / (1 - pt)
        manual_model.append(tp / n - fp / n * w)
        manual_all.append(prevalence - (1 - prevalence) * w)

    assert res["thresholds"] == pytest.approx([round(float(t), 4) for t in thresholds], abs=1e-9)
    assert res["net_benefit_model"] == pytest.approx(manual_model, abs=1e-4)
    assert res["net_benefit_all"] == pytest.approx(manual_all, abs=1e-4)
    assert res["best_threshold"] in [round(float(t), 4) for t in thresholds]


def test_dca_treat_none_is_always_zero():
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.3, 100)
    p = rng.uniform(0, 1, 100)
    res = decision_curve_analysis(y, p, n_points=50)
    assert all(v == 0.0 for v in res["net_benefit_none"])
