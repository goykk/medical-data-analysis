# medical-data-analysis/backend/tests/e2e_smoke_automl.py
"""端到端冒烟测试：三类验收数据集跑通「标准化→画像→路由→赛马→冠军→报告」全流程。

列名一律使用原始风格命名（中文简称/英文缩写/中英混合/带单位/内部编码/生存结构），
严禁使用标准名——验证模块零的标准化能力贯穿全流程。

运行：pytest tests/e2e_smoke_automl.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl.pipeline import run_auto, run_manual
from automl.reporting import DISCLAIMER
from automl.types import TASK_CLASSIFICATION, TASK_SURVIVAL


def _binary_raw_df(n=150, seed=0):
    from sklearn.datasets import make_classification
    rng = np.random.default_rng(seed)
    X, y = make_classification(n_samples=n, n_features=6, n_informative=4,
                               n_redundant=0, random_state=seed)
    df = pd.DataFrame(X, columns=["WBC", "HGB血红蛋白", "血糖(mmol/L)", "血小板",
                                  "Cr", "ALT"])
    df["分组"] = rng.choice(["对照组", "试验组"], n)
    df["结局"] = y
    return df


def _high_dim_df(n=40, p=50, seed=1):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=p, n_informative=10,
                               n_redundant=0, random_state=seed)
    df = pd.DataFrame(X, columns=[f"JYZD{i:03d}" for i in range(p)])   # 医院内部编码风格
    df["结局"] = y
    return df


def _survival_df(n=160, seed=2):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"年龄": rng.normal(60, 10, n).round(0),
                      "WBC": rng.normal(7, 2, n).round(1),
                      "血红蛋白": rng.normal(130, 15, n).round(1)})
    lin = 0.04 * (X["年龄"] - 60) - 0.15 * (X["WBC"] - 7)
    t = rng.weibull(1.5, n) * np.exp(-lin) * 250
    c = rng.exponential(350, n)
    ev = (t <= c).astype(int)
    X["随访时间"] = np.where(ev == 1, t, c).round(1)
    X["是否死亡"] = ev
    return X


def test_smoke_binary_classification():
    df = _binary_raw_df()
    rep = run_auto(df, target="结局")
    assert rep["disclaimer"] == DISCLAIMER
    assert rep["profile"]["task_type"] == TASK_CLASSIFICATION
    assert rep["leaderboard"] and rep["leaderboard"][0]["status"] == "ok"
    assert rep["champion"] and rep["champion"]["importance_top15"]
    assert rep["champion"]["shap"]["features"]
    # 原始名命中标准名（模块零贯穿全流程）
    std = {c["raw_name"]: c["standard_name"] for c in rep["standardization"]["columns"]}
    assert std["WBC"] == "白细胞计数"
    assert std["HGB血红蛋白"] == "血红蛋白"
    assert std["血糖(mmol/L)"] == "血糖"


def test_smoke_high_dim_small_n():
    df = _high_dim_df()
    rep = run_auto(df, target="结局")
    assert rep["profile"]["is_high_dim_small_n"] is True
    assert set(rep["routing"]["candidate_keys"]) == {"lasso", "elastic_net", "linear_svm"}
    assert rep["leaderboard"] and rep["leaderboard"][0]["status"] == "ok"
    assert rep["champion"]


def test_smoke_survival():
    df = _survival_df()
    rep = run_auto(df)   # 不传 time/event，靠标准名自动识别
    assert rep["profile"]["task_type"] == TASK_SURVIVAL
    assert rep["profile"]["time_col"] == "随访时间"
    assert rep["profile"]["event_col"] == "是否死亡"
    assert 0.0 < rep["profile"]["censoring_rate"] < 1.0
    assert rep["leaderboard"][0]["metric_name"] == "cindex"
    assert rep["champion"]["model_key"] in ("cox_ph", "random_survival_forest")


def test_smoke_manual_mode():
    df = _binary_raw_df()
    rep = run_manual(df, model_key="logistic", target="结局")
    assert rep["champion"]["model_key"] == "logistic"
    assert rep["disclaimer"] == DISCLAIMER
    assert rep["leaderboard"][0]["status"] == "ok"


# ------------------------------------------------------------------
# 可直接以脚本运行：python tests/e2e_smoke_automl.py
# ------------------------------------------------------------------
def _run_all():
    scenarios = [
        ("普通二分类表格（原始风格列名）", test_smoke_binary_classification),
        ("高维小样本（内部编码列名）", test_smoke_high_dim_small_n),
        ("生存数据（随访时间+是否死亡）", test_smoke_survival),
        ("手动模式（指定 logistic）", test_smoke_manual_mode),
    ]
    failed = 0
    for name, fn in scenarios:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:200]}")
    print(f"\n冒烟测试：{len(scenarios) - failed}/{len(scenarios)} 通过")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
