# medical-data-analysis/backend/tests/test_automl_profiler.py
"""模块一（数据画像）测试：任务推断、生存结构、不平衡度、高维小样本、ID 判定。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl.errors import DataShapeError, SurvivalDataError
from automl.profiler import (detect_survival_pair, imbalance_ratio,
                             infer_task_type, profile_dataframe,
                             validate_survival_inputs)
from automl.standardize import standardize_dataframe
from automl.types import (TASK_CLASSIFICATION, TASK_CLUSTERING, TASK_REGRESSION,
                          TASK_SURVIVAL)


def _surv_df(n=200, seed=0, censoring=True):
    rng = np.random.default_rng(seed)
    t = rng.exponential(300, n).round(1)
    ev = rng.binomial(1, 0.7, n)
    if not censoring:
        ev = np.ones(n, dtype=int)
    return pd.DataFrame({"随访时间": t, "事件": ev,
                         "年龄": rng.normal(60, 10, n).round(0),
                         "血红蛋白": rng.normal(130, 15, n).round(1)})


def test_task_classification_binary():
    df = pd.DataFrame({"白细胞": [3, 5, 8, 9], "死亡": [0, 1, 0, 1]})
    t, reason = infer_task_type(df, "死亡")
    assert t == TASK_CLASSIFICATION and "分类" in reason


def test_task_regression_continuous():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"收缩压": rng.normal(130, 15, 60).round(0),
                       "年龄": rng.normal(60, 10, 60).round(0)})
    t, reason = infer_task_type(df, "收缩压")
    assert t == TASK_REGRESSION and "回归" in reason


def test_numeric_outcome_with_decimals_is_regression():
    """1 位小数的数值结局（如 BMI）即使水平数 ≤20 也是连续变量，不能当分类。"""
    rng = np.random.default_rng(1)
    levels = np.array([18.5, 21.2, 24.8, 27.5, 30.1])
    df = pd.DataFrame({"结局": rng.choice(levels, 120),
                       "x": rng.normal(size=120)})
    assert df["结局"].nunique() <= 20          # 前提：水平数确实少
    t, _ = infer_task_type(df, "结局")
    assert t == TASK_REGRESSION


def test_numeric_integer_outcome_few_levels_is_classification():
    """0/1/2/3 这类整数分级结局仍按分类处理。"""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"分级": rng.integers(0, 4, 120),
                       "x": rng.normal(size=120)})
    t, _ = infer_task_type(df, "分级")
    assert t == TASK_CLASSIFICATION


def test_numeric_integer_outcome_many_levels_is_regression():
    """整数但水平多且每级样本稀少（0–19 分级、平均每级 <10 例）→ 按连续变量回归。"""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"计数": rng.integers(0, 20, 120),
                       "x": rng.normal(size=120)})
    t, _ = infer_task_type(df, "计数")
    assert t == TASK_REGRESSION


def test_ordinal_scale_outcome_gets_explicit_warning():
    """6–20 个整数水平的结局存在分类/回归歧义，必须显式提醒用户可选择回归。"""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"疼痛评分": rng.integers(0, 11, 220),
                       "年龄": rng.normal(60, 10, 220).round(0)})
    p = profile_dataframe(df, target="疼痛评分")
    assert p.task_type == TASK_CLASSIFICATION
    assert any("序数" in w or "量表" in w for w in p.warnings)


def test_task_clustering_no_target():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t, _ = infer_task_type(df, None)
    assert t == TASK_CLUSTERING


def test_task_survival_detected_via_standard_names():
    df = _surv_df()
    t, e = detect_survival_pair(df, standardize_dataframe(df))
    assert t == "随访时间" and e == "事件"
    task, reason = infer_task_type(df, "死亡", t, e, standardize_dataframe(df))
    assert task == TASK_SURVIVAL and "生存分析" in reason


def test_task_target_missing_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(DataShapeError):
        infer_task_type(df, "不存在的列")


def test_task_single_value_target_raises():
    df = pd.DataFrame({"a": [1, 2, 3], "结局": [1, 1, 1]})
    with pytest.raises(DataShapeError):
        infer_task_type(df, "结局")


def test_validate_survival_ok():
    df = _surv_df(censoring=True)
    r = validate_survival_inputs(df, "随访时间", "事件")
    assert 0.0 < r["censoring_rate"] < 1.0
    assert r["n_events"] > 0 and r["n_censored"] > 0


def test_validate_survival_event_not_binary_raises():
    df = _surv_df()
    df["事件"] = [1, 2, 0, 1][:len(df)] if len(df) <= 4 else np.where(np.arange(len(df)) < 5, 2, df["事件"])
    with pytest.raises(SurvivalDataError) as e:
        validate_survival_inputs(df, "随访时间", "事件")
    assert "0/1" in str(e.value)


def test_validate_survival_negative_time_raises():
    df = _surv_df()
    df.loc[0, "随访时间"] = -3
    with pytest.raises(SurvivalDataError):
        validate_survival_inputs(df, "随访时间", "事件")


def test_validate_survival_no_censoring_raises():
    df = _surv_df(censoring=False)   # 全事件，删失比例 0
    with pytest.raises(SurvivalDataError) as e:
        validate_survival_inputs(df, "随访时间", "事件")
    assert "C-index" in str(e.value)


def test_imbalance_ratio():
    y = [0] * 90 + [1] * 10
    assert imbalance_ratio(y) == 0.1
    assert imbalance_ratio([1, 1, 1]) is None           # 单一取值 → None
    assert imbalance_ratio([]) is None                   # 空 → None


def test_profile_high_dim_small_n():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"g{i}": rng.normal(size=8) for i in range(12)})
    df["结局"] = [0, 1] * 4
    p = profile_dataframe(df, target="结局")
    assert p.is_high_dim_small_n is True
    assert p.n_features_used == 12
    assert any("高维小样本" in w for w in p.warnings)


def test_profile_id_detection_by_standard_name():
    df = pd.DataFrame({"住院号": [f"P{i:04d}" for i in range(30)],
                       "年龄": np.random.default_rng(0).normal(60, 10, 30).round(0),
                       "结局": [0, 1] * 15})
    p = profile_dataframe(df, target="结局")
    assert p.patient_id_col == "住院号"
    col = next(c for c in p.columns if c.name == "住院号")
    assert col.is_id_like and col.excluded and col.role == "id"
    # ID 列不进入特征
    assert "住院号" not in p.feature_columns()


def test_profile_survival_end_to_end():
    df = _surv_df()
    p = profile_dataframe(df)   # 无显式 time/event，靠标准名识别
    assert p.task_type == TASK_SURVIVAL
    assert p.time_col == "随访时间" and p.event_col == "事件"
    assert p.censoring_rate is not None and 0.0 < p.censoring_rate < 1.0


def test_profile_empty_df_raises():
    with pytest.raises(DataShapeError):
        profile_dataframe(pd.DataFrame())
    with pytest.raises(DataShapeError):
        profile_dataframe(pd.DataFrame(columns=["a"]))


def test_profile_missing_rate_and_roles():
    df = pd.DataFrame({"年龄": [1, 2, None, 4, 5], "分组": ["a", "b", "a", "b", "a"]})
    p = profile_dataframe(df)
    age = next(c for c in p.columns if c.name == "年龄")
    assert age.role == "numeric" and age.missing_rate == 0.2
    grp = next(c for c in p.columns if c.name == "分组")
    assert grp.role == "categorical"
