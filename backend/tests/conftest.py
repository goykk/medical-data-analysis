# medical-data-analysis/backend/tests/conftest.py
"""pytest 共用夹具：生成确定性模拟医疗数据（脱敏），供各测试模块复用。"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_medical_df(n: int = 300, seed: int = 42, with_issues: bool = True) -> pd.DataFrame:
    """生成模拟医疗数据集（含缺失值、重复行、超参考区间异常值、ID 列与常量列）。"""
    rng = np.random.default_rng(seed)
    age = rng.normal(55, 10, n).round(0)
    gender = rng.integers(0, 2, n)
    bmi = rng.normal(24, 3, n).round(1)
    sbp = rng.normal(130, 15, n).round(0)
    glucose = rng.normal(6.0, 1.5, n).round(2)
    smoking = rng.integers(0, 2, n)
    group = rng.choice(["对照组", "试验组"], n)

    logit = (-0.4 + 0.9 * (glucose - 6.0) + 0.03 * (age - 55)
             + 0.8 * smoking + 0.5 * (group == "试验组"))
    prob = 1.0 / (1.0 + np.exp(-logit))
    outcome = rng.binomial(1, prob)

    df = pd.DataFrame({
        "患者ID": [f"P{i:04d}" for i in range(n)],
        "年龄": age,
        "性别": gender,
        "体质指数": bmi,
        "收缩压": sbp,
        "空腹血糖": glucose,
        "吸烟": smoking,
        "分组": group,
        "结局": outcome,
        "中心": "A",
    })

    if with_issues:
        idx = rng.choice(n, size=15, replace=False)
        df.loc[idx, "空腹血糖"] = np.nan
        df.loc[0, "空腹血糖"] = 18.6          # 超出医学参考区间的异常值（重度高血糖）
        df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # 2 条完全重复行
    return df


def make_non_medical_df(n: int = 50, seed: int = 7) -> pd.DataFrame:
    """生成明显的非医疗数据集，用于校验器负例测试。"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "商品编号": [f"S{i:03d}" for i in range(n)],
        "销售额": rng.normal(1000, 200, n).round(2),
        "门店": rng.choice(["门店甲", "门店乙", "门店丙"], n),
        "库存量": rng.integers(0, 500, n),
        "季度": rng.choice(["Q1", "Q2", "Q3", "Q4"], n),
    })


@pytest.fixture
def medical_df():
    return make_medical_df()


@pytest.fixture
def non_medical_df():
    return make_non_medical_df()
