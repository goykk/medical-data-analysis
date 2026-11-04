# medical-data-analysis/backend/tests/test_automl_leakage.py
"""模块二（泄露列检测）测试：分类/回归/生存三种度量 + 处置决策 + 排除对象。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl.leakage import apply_leakage_decision, detect_leakage
from automl.types import TASK_CLASSIFICATION, TASK_REGRESSION, TASK_SURVIVAL


def test_detects_perfect_copy_binary():
    """目标列的重复副本 -> AUC=1.0 -> leak。"""
    rng = np.random.default_rng(0)
    n = 100
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"结局": y, "结局副本": y, "噪声": rng.normal(size=n)})
    finds = detect_leakage(df, target="结局", task=TASK_CLASSIFICATION)
    leaks = {f.column: f for f in finds if f.verdict == "leak"}
    assert "结局副本" in leaks
    assert leaks["结局副本"].score >= 0.99
    assert leaks["结局副本"].metric == "auc"


def test_random_feature_not_leak():
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"结局": y, "噪声": rng.normal(size=n)})
    finds = detect_leakage(df, target="结局", task=TASK_CLASSIFICATION)
    assert all(f.verdict == "ok" for f in finds)


def test_multiclass_perfect_copy_leak():
    """三分类：one-vs-rest 宏平均 AUC 也应检出泄露。"""
    rng = np.random.default_rng(2)
    n = 150
    y = rng.integers(0, 3, n)
    df = pd.DataFrame({"分级": y, "分级副本": y})
    finds = detect_leakage(df, target="分级", task=TASK_CLASSIFICATION)
    leaks = [f for f in finds if f.verdict == "leak"]
    assert any(f.column == "分级副本" and f.score >= 0.99 for f in leaks)


def test_excludes_target_id_time_event():
    rng = np.random.default_rng(3)
    n = 100
    ev = rng.binomial(1, 0.6, n)
    tt = rng.exponential(200, n)
    df = pd.DataFrame({
        "住院号": [f"P{i:03d}" for i in range(n)],
        "随访时间": tt, "事件": ev,
        "结局": rng.integers(0, 2, n),
    })
    # 分类任务：排除住院号；时间/事件列为对象列也需不被误判
    finds = detect_leakage(df, target="结局", task=TASK_CLASSIFICATION,
                           time_col="随访时间", event_col="事件", patient_id_col="住院号")
    checked = {f.column for f in finds}
    assert "住院号" not in checked
    assert "随访时间" not in checked and "事件" not in checked
    assert "结局" not in checked


def test_regression_spearman_leak():
    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": 3.0 * x + rng.normal(0, 1e-6, n), "噪声": rng.normal(size=n)})
    finds = detect_leakage(df, target="y", task=TASK_REGRESSION)
    leaks = {f.column: f for f in finds if f.verdict == "leak"}
    assert "x" in leaks and leaks["x"].metric == "spearman"
    assert leaks["x"].score > 0.99


def test_survival_cindex_leak():
    """feature = -时间 -> 风险单调 -> C-index=1 -> leak。"""
    rng = np.random.default_rng(5)
    n = 150
    tt = rng.exponential(300, n)
    ev = rng.binomial(1, 0.6, n)
    risk = -tt
    df = pd.DataFrame({"随访时间": tt, "事件": ev, "反时间风险": risk})
    finds = detect_leakage(df, task=TASK_SURVIVAL, time_col="随访时间", event_col="事件")
    leaks = {f.column: f for f in finds if f.verdict == "leak"}
    assert "反时间风险" in leaks and leaks["反时间风险"].metric == "cindex"
    assert leaks["反时间风险"].score > 0.95


def test_clustering_returns_empty():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert detect_leakage(df, task="clustering") == []


def test_binary_with_multilevel_categorical_uses_nmi():
    """二分类结局 + 多水平类别特征（如 ICD 码）若完全决定结局，
    此前因 factorize 编码顺序问题 AUC 可能 ≈0.5；修复后改用 NMI，应准确检出泄露。"""
    rng = np.random.default_rng(7)
    n = 100
    codes = ["A", "B", "C", "D"]
    # 类别列完全决定结局（A/B→0，C/D→1）
    cat = rng.choice(codes, n)
    y = np.array([0 if c in ("A", "B") else 1 for c in cat])
    noise = rng.normal(size=n)
    df = pd.DataFrame({"结局": y, "科室编码": cat, "噪声": noise})
    finds = detect_leakage(df, target="结局", task=TASK_CLASSIFICATION)
    leaks = {f.column: f for f in finds if f.verdict == "leak"}
    assert "科室编码" in leaks
    assert leaks["科室编码"].metric == "nmi"
    assert leaks["科室编码"].score >= 0.99


def test_inf_column_does_not_crash_detection():
    """单列含 inf 值：inf 按缺失处理，检测不中断、打分有限、其余列不受影响。"""
    rng = np.random.default_rng(8)
    n = 50
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "结局": y,
        "正常": rng.normal(size=n),
        "含无穷": np.where(rng.random(n) < 0.3, np.inf, rng.normal(size=n)),
    })
    finds = detect_leakage(df, target="结局", task=TASK_CLASSIFICATION)
    cols = {f.column for f in finds}
    assert "正常" in cols                       # 其余列正常打分
    assert all(np.isfinite(f.score) for f in finds)   # 不产生 NaN/Inf 得分


def test_binary_target_with_missing_still_uses_auc():
    """含缺失的二分类结局：nunique() 把 NaN 当成一个水平会误走 NMI，
    排除缺失后仍应走 AUC 并检出完美副本。"""
    rng = np.random.default_rng(9)
    n = 120
    y = rng.integers(0, 2, n).astype(float)
    y[:15] = np.nan
    df = pd.DataFrame({"结局": y, "结局副本": y.copy(), "噪声": rng.normal(size=n)})
    finds = detect_leakage(df, target="结局", task=TASK_CLASSIFICATION)
    leak = next(f for f in finds if f.column == "结局副本")
    assert leak.metric == "auc"
    assert leak.score >= 0.99


def test_missing_categorical_value_is_nan_not_a_level():
    """类别列的缺失值必须还原为 NaN，不能被 factorize 的 -1 当成真实类别参与打分。"""
    from automl.leakage import _to_score_series
    arr = _to_score_series(pd.Series(["a", "b", None, "a", None]))
    assert np.isnan(arr[2]) and np.isnan(arr[4])
    assert not np.isnan(arr[0]) and not np.isnan(arr[1])


def test_apply_decision_drop_keep_ask():
    findings = []
    # 构造两个泄露 finding（直接构造对象）
    from automl.types import LeakageFinding
    findings = [LeakageFinding(column="c1", ref=None, score=0.99, threshold=0.95,
                               metric="auc", verdict="leak"),
                LeakageFinding(column="c2", ref=None, score=0.5, threshold=0.95,
                               metric="auc", verdict="ok")]

    assert apply_leakage_decision(None, findings, "drop")["dropped"] == ["c1"]
    assert apply_leakage_decision(None, findings, "keep")["kept"] == ["c1"]
    # ask + 用户确认只剔除 c1
    r = apply_leakage_decision(None, findings, "ask", user_confirmed_drop=["c1"])
    assert r["dropped"] == ["c1"] and r["kept"] == []
    # ask + 跳过确认 -> 默认全部剔除
    r = apply_leakage_decision(None, findings, "ask")
    assert r["dropped"] == ["c1"] and "跳过确认" in r["note"]
