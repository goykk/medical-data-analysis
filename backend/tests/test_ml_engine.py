# medical-data-analysis/backend/tests/test_ml_engine.py
"""机器学习模块测试：WDBC 全流程、防数据泄漏、可复现、三件套、SHAP、新样本预测。"""

import os

import numpy as np
import pandas as pd
import pytest

import ml_engine
from config import MIN_N_FOR_MODELING, RANDOM_SEED

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WDBC_PATH = os.path.join(ROOT, "sample_data", "wdbc.csv")
MEDICAL_PATH = os.path.join(ROOT, "sample_data", "medical_sample.csv")

WDBC_FEATURES = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                 "smoothness_mean", "compactness_mean", "concavity_mean",
                 "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"]


@pytest.fixture(scope="module")
def wdbc():
    if not os.path.exists(WDBC_PATH):
        pytest.skip("缺少 sample_data/wdbc.csv，请先运行 sample_data/generate_sample_data.py")
    return pd.read_csv(WDBC_PATH)


@pytest.fixture(scope="module")
def wdbc_result(wdbc):
    return ml_engine.train_and_evaluate(wdbc, WDBC_FEATURES, "diagnosis", "classification",
                                        variable_types={f: "numeric" for f in WDBC_FEATURES})


# ------------------------------------------------------------------
# WDBC 全流程（基准测试）
# ------------------------------------------------------------------

def test_wdbc_classification_runs_and_performs(wdbc_result):
    assert wdbc_result.get("skipped") is not True
    assert wdbc_result["n_total"] == 569
    assert wdbc_result["n_train"] + wdbc_result["n_test"] == 569
    best = next(m for m in wdbc_result["models"] if m["name"] == wdbc_result["best_model"]["name"])
    assert best["test_metrics"]["roc_auc"] > 0.90, "WDBC 上 AUC 应显著优于随机"
    assert best["test_metrics"]["accuracy"] > 0.85
    assert best["cv_mean"] > 0.90


def test_two_candidate_models_are_compared(wdbc_result):
    names = [m["name"] for m in wdbc_result["models"]]
    assert set(names) == {"logistic", "random_forest"}
    for m in wdbc_result["models"]:
        assert "±" in m["cv_text"]            # 交叉验证报告均值±标准差
    assert wdbc_result["best_model"]["reason"]


def test_roc_calibration_dca_all_present_and_labelled(wdbc_result):
    """评估三件套必须齐全，且全部基于测试集（标题注明 n=测试集样本量）。"""
    n_test = wdbc_result["n_test"]
    roc, cal, dca = wdbc_result["roc"], wdbc_result["calibration"], wdbc_result["dca"]
    assert roc["auc"] > 0.90
    assert len(roc["fpr"]) == len(roc["tpr"]) == len(roc["thresholds"])
    assert 0 <= roc["youden_threshold"] <= 1
    assert roc["title"] == f"ROC 曲线（测试集 n={n_test}）"

    assert len(cal["prob_true"]) == len(cal["prob_pred"])
    assert cal["title"] == f"校准曲线（测试集 n={n_test}）"
    assert "statistic" in cal["hosmer_lemeshow"] and "p" in cal["hosmer_lemeshow"]

    assert len(dca["net_benefit_model"]) == len(dca["thresholds"])
    assert all(v == 0.0 for v in dca["net_benefit_none"])
    assert dca["title"] == f"决策曲线分析 DCA（测试集 n={n_test}）"
    # WDBC 训练集仅 427 例：按校准方法选择规则应走 sigmoid（Platt），
    # isotonic 只在 n≥1000 时启用（小样本等渗回归会退化成 0/1 台阶）
    # 校准复用 CV 的同一批折（OOF 概率），不再额外拟合 CalibratedClassifierCV
    assert wdbc_result["calibration_model"] == "OOF校准（sigmoid, 复用 5 折 CV）"


def test_confusion_matrix_and_per_class_recall(wdbc_result):
    best = next(m for m in wdbc_result["models"] if m["name"] == wdbc_result["best_model"]["name"])
    tm = best["test_metrics"]
    cm = tm["confusion_matrix"]
    assert sum(sum(r) for r in cm) == wdbc_result["n_test"]
    assert len(tm["per_class"]) == 2                    # B / M 两类的召回率都要给
    for cls, v in tm["per_class"].items():
        assert 0 <= v["recall"] <= 1
    assert tm["sensitivity"] is not None and tm["specificity"] is not None


def test_class_weight_enabled_for_imbalance(wdbc_result):
    """类别不平衡时启用 class_weight。"""
    pipe = wdbc_result["_fitted"]["pipeline"]
    model = pipe.named_steps["model"]
    if hasattr(model, "class_weight"):
        assert model.class_weight in ("balanced", "balanced_subsample")


def test_no_data_leakage_scaler_fitted_on_train_only(wdbc, wdbc_result):
    """标准化参数必须只在训练集上拟合，不得使用全量数据。"""
    pipe = wdbc_result["_fitted"]["pipeline"]
    X_train = wdbc_result["_fitted"]["X_train"]
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
    train_means = X_train[WDBC_FEATURES].astype(float).mean().to_numpy()
    full_means = wdbc[WDBC_FEATURES].astype(float).mean().to_numpy()
    assert np.allclose(scaler.mean_, train_means, atol=1e-6)
    assert not np.allclose(scaler.mean_, full_means, atol=1e-3), "标准化使用了全量数据，存在数据泄漏"


def test_random_seed_fixed_and_reproducible(wdbc):
    r1 = ml_engine.train_and_evaluate(wdbc, WDBC_FEATURES, "diagnosis", "classification",
                                      variable_types={f: "numeric" for f in WDBC_FEATURES},
                                      with_shap=False)
    r2 = ml_engine.train_and_evaluate(wdbc, WDBC_FEATURES, "diagnosis", "classification",
                                      variable_types={f: "numeric" for f in WDBC_FEATURES},
                                      with_shap=False)
    assert r1["random_seed"] == RANDOM_SEED
    a1 = next(m["test_metrics"]["roc_auc"] for m in r1["models"] if m["name"] == r1["best_model"]["name"])
    a2 = next(m["test_metrics"]["roc_auc"] for m in r2["models"] if m["name"] == r2["best_model"]["name"])
    assert a1 == a2, "固定随机种子下结果必须可复现"


# ------------------------------------------------------------------
# SHAP
# ------------------------------------------------------------------

def test_shap_uses_allowed_explainer_and_aggregates_dummies(wdbc_result):
    shap_res = wdbc_result["shap"]
    assert "error" not in shap_res, shap_res.get("error")
    assert shap_res["method"] in ("TreeExplainer", "LinearExplainer")
    names = [f["variable"] for f in shap_res["features"]]
    assert set(names) == set(WDBC_FEATURES)             # 聚合后应还原为原始变量
    assert shap_res["n_sample"] <= 1000                 # 蜂群图最多采样 1000 点
    for f in shap_res["features"]:
        assert len(f["values"]) == shap_res["n_sample"]
    top = shap_res["features"][0]["variable"]
    assert top in WDBC_FEATURES


def test_shap_respects_max_points(wdbc, wdbc_result):
    """样本量超 1000 时后端采样 1000 点（此处全量不足 1000，验证不超过上限）。"""
    assert wdbc_result["shap"]["n_sample"] <= min(1000, wdbc_result["n_test"])


def test_feature_importance_aggregated_to_original_variables(wdbc_result):
    names = [f["variable"] for f in wdbc_result["feature_importance"]]
    assert set(names) == set(WDBC_FEATURES)
    assert abs(sum(f["importance"] for f in wdbc_result["feature_importance"]) - 1.0) < 1e-6


def test_categorical_dummy_aggregation():
    """分类特征独热后，重要性与 SHAP 均应按原始变量聚合。"""
    rng = np.random.default_rng(3)
    n = 260
    df = pd.DataFrame({
        "x_num": rng.normal(0, 1, n),
        "grade": rng.choice(["I", "II", "III"], n),
    })
    logit = 1.2 * df["x_num"] + np.where(df["grade"] == "III", 1.5, np.where(df["grade"] == "II", 0.6, 0))
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    res = ml_engine.train_and_evaluate(df, ["x_num", "grade"], "y", "classification",
                                       variable_types={"x_num": "numeric", "grade": "categorical"})
    imp = {f["variable"] for f in res["feature_importance"]}
    assert imp == {"x_num", "grade"}, imp
    shap_names = {f["variable"] for f in res["shap"]["features"]}
    assert shap_names == {"x_num", "grade"}, shap_names


# ------------------------------------------------------------------
# 回归任务
# ------------------------------------------------------------------

def test_regression_task_with_known_relationship():
    """舒张压由收缩压线性生成，回归模型应能学出该关系。"""
    if not os.path.exists(MEDICAL_PATH):
        pytest.skip("缺少 sample_data/medical_sample.csv")
    df = pd.read_csv(MEDICAL_PATH, encoding="gbk")
    res = ml_engine.train_and_evaluate(df, ["收缩压", "年龄", "体质指数"], "舒张压", "regression",
                                       variable_types={"收缩压": "numeric", "年龄": "numeric",
                                                       "体质指数": "numeric"})
    assert res["task"] == "regression"
    best = next(m for m in res["models"] if m["name"] == res["best_model"]["name"])
    assert best["test_metrics"]["r2"] > 0.5
    assert "rmse" in best["test_metrics"]
    assert "roc" not in res            # 回归任务不输出分类三件套


# ------------------------------------------------------------------
# 样本量红线
# ------------------------------------------------------------------

def test_small_sample_is_skipped():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=20), "y": rng.integers(0, 2, 20)})
    res = ml_engine.train_and_evaluate(df, ["x"], "y", "classification",
                                       variable_types={"x": "numeric"})
    assert res["skipped"] is True
    assert "严重不足" in res["reason"]


def test_moderate_sample_warns_but_still_models():
    rng = np.random.default_rng(2)
    n = 45                                    # < MIN_N_FOR_MODELING，触发警示但不至跳过
    x = rng.normal(size=n)
    y = (x + rng.normal(0, 0.5, n) > 0).astype(int)
    df = pd.DataFrame({"x": x, "y": y})
    res = ml_engine.train_and_evaluate(df, ["x"], "y", "classification",
                                       variable_types={"x": "numeric"}, with_shap=False)
    assert res.get("skipped") is not True
    assert any("样本量" in w or "EPV" in w for w in res["warnings"])
    assert MIN_N_FOR_MODELING == 50


# ------------------------------------------------------------------
# 模型持久化与新样本预测
# ------------------------------------------------------------------

@pytest.fixture
def trained_model(wdbc):
    dataset_id = "test_ds_wdbc"
    res = ml_engine.train_and_evaluate(wdbc, WDBC_FEATURES, "diagnosis", "classification",
                                       variable_types={f: "numeric" for f in WDBC_FEATURES},
                                       with_shap=False)
    ml_engine.save_model(dataset_id, res)
    yield dataset_id
    ml_engine.invalidate_model(dataset_id)


def test_predict_new_sample_returns_probability_and_shap(trained_model, wdbc):
    sample = {f: float(wdbc[f].iloc[0]) for f in WDBC_FEATURES}
    out = ml_engine.predict_new_sample(trained_model, sample)
    assert out["predicted_class"] in ("B", "M")
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-3
    assert out["positive_probability"] is not None
    assert out["shap_force_plot"], "SHAP 力图应为非空 base64 字符串"
    import base64
    assert base64.b64decode(out["shap_force_plot"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert "不构成临床诊断依据" in out["disclaimer"]


def test_predict_is_reproducible(trained_model, wdbc):
    sample = {f: float(wdbc[f].iloc[5]) for f in WDBC_FEATURES}
    a = ml_engine.predict_new_sample(trained_model, sample)
    b = ml_engine.predict_new_sample(trained_model, sample)
    assert a["predicted_class"] == b["predicted_class"]
    assert a["probabilities"] == b["probabilities"]


def test_extrapolation_warning(trained_model, wdbc):
    bundle = ml_engine.load_model(trained_model)
    lo, hi = bundle["numeric_ranges"]["radius_mean"]
    sample = {f: float(wdbc[f].iloc[0]) for f in WDBC_FEATURES}
    sample["radius_mean"] = hi * 3 + 100          # 明显超出训练集范围
    out = ml_engine.predict_new_sample(trained_model, sample)
    assert any("超出训练集范围" in w for w in out["warnings"])


def test_unseen_category_warning(trained_model, wdbc):
    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame({"g": rng.choice(["甲", "乙"], n), "x": rng.normal(size=n)})
    df["y"] = (df["x"] + rng.normal(0, 0.5, n) > 0).astype(int)
    res = ml_engine.train_and_evaluate(df, ["g", "x"], "y", "classification",
                                       variable_types={"g": "categorical", "x": "numeric"},
                                       with_shap=False)
    ml_engine.save_model("test_ds_cat", res)
    try:
        out = ml_engine.predict_new_sample("test_ds_cat", {"g": "丙", "x": 0.0})
        assert any("未在训练集中出现过" in w for w in out["warnings"])
    finally:
        ml_engine.invalidate_model("test_ds_cat")


def test_missing_model_gives_clear_error():
    with pytest.raises(ml_engine.MLError) as exc:
        ml_engine.predict_new_sample("not_trained_ds", {"x": 1.0})
    assert "尚未完成建模" in str(exc.value)


def test_invalidate_model_makes_it_unusable(trained_model):
    assert ml_engine.load_model(trained_model) is not None
    ml_engine.invalidate_model(trained_model)
    assert ml_engine.load_model(trained_model) is None


def test_new_training_overwrites_old_model(wdbc):
    ds = "test_ds_overwrite"
    r1 = ml_engine.train_and_evaluate(wdbc, WDBC_FEATURES[:3], "diagnosis", "classification",
                                      variable_types={f: "numeric" for f in WDBC_FEATURES[:3]},
                                      with_shap=False)
    ml_engine.save_model(ds, r1)
    r2 = ml_engine.train_and_evaluate(wdbc, WDBC_FEATURES, "diagnosis", "classification",
                                      variable_types={f: "numeric" for f in WDBC_FEATURES},
                                      with_shap=False)
    ml_engine.save_model(ds, r2)
    try:
        assert ml_engine.load_model(ds)["features"] == WDBC_FEATURES
    finally:
        ml_engine.invalidate_model(ds)


def test_modeling_never_forks_worker_processes():
    """建模一律串行：服务端进程内用 joblib/loky 派生子进程会导致 worker 被异常终止（500）。

    事故现象：一看板就"无法连接后端"——/api/ml/train 因 loky worker 崩溃直接 500，
    且崩溃期间主进程被牵制，前端心跳超时判定为"后端离线"。
    """
    assert ml_engine.N_JOBS == 1
    for task in ("classification", "regression"):
        for _label, est in ml_engine._candidate_models(task).values():
            # 未设 n_jobs 的估计器默认即串行（None），同视为单进程
            assert (getattr(est, "n_jobs", 1) or 1) == 1


# ------------------------------------------------------------------
# 缺失值不再整行剔除
# ------------------------------------------------------------------

def test_feature_missing_values_are_imputed_not_dropped(wdbc):
    """特征缺失必须由 Pipeline 内 imputer 处理，不能在划分前整行 dropna。

    此前 dropna() 在建模前删掉含缺失的行，使 imputer 永远拿不到缺失值（形同失效），
    同时在小样本上造成明显的样本损失与选择性偏倚。
    """
    df = wdbc.copy()
    feature = WDBC_FEATURES[0]
    df.loc[df.index[:20], feature] = np.nan
    res = ml_engine.train_and_evaluate(df, WDBC_FEATURES, "diagnosis", "classification",
                                       variable_types={f: "numeric" for f in WDBC_FEATURES},
                                       with_shap=False)
    assert res["n_total"] == 569, "特征缺失不应导致整行被剔除"
    assert res["n_train"] + res["n_test"] == 569
    prep = res["_fitted"]["pipeline"].named_steps["prep"]
    imputer = prep.named_transformers_["num"].named_steps["imputer"]
    assert not np.isnan(np.asarray(imputer.statistics_, dtype=float)).any()


def test_object_numeric_feature_with_junk_does_not_fail():
    """确认页把 object 列设为 numeric 且含脏值（'NaNN'）时，不得抛 ValueError（500）。"""
    rng = np.random.default_rng(9)
    n = 80
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": [f"{v:.2f}" for v in x],
                       "y": (x > 0).astype(int)})
    df.loc[df.index[5], "x"] = "NaNN"
    res = ml_engine.train_and_evaluate(df, ["x"], "y", "classification",
                                       variable_types={"x": "numeric"}, with_shap=False)
    assert res.get("skipped") is not True
    assert res["n_total"] == n
    imputer = res["_fitted"]["pipeline"].named_steps["prep"] \
        .named_transformers_["num"].named_steps["imputer"]
    assert not np.isnan(np.asarray(imputer.statistics_, dtype=float)).any()


def test_target_missing_rows_are_dropped(wdbc):
    """结局缺失无法填补，必须剔除；其余行保留。"""
    df = wdbc.copy()
    df.loc[df.index[:10], "diagnosis"] = None
    res = ml_engine.train_and_evaluate(df, WDBC_FEATURES, "diagnosis", "classification",
                                       variable_types={f: "numeric" for f in WDBC_FEATURES},
                                       with_shap=False)
    assert res["n_total"] == 559


# ------------------------------------------------------------------
# 多分类评估三件套（one-vs-rest）
# ------------------------------------------------------------------

def test_multiclass_produces_ovr_three_piece():
    rng = np.random.default_rng(0)
    n = 240
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.where(x1 + x2 > 1.0, 2, np.where(x1 - x2 > 0, 1, 0))
    df = pd.DataFrame({"x1": x1, "x2": x2, "grade": y})
    res = ml_engine.train_and_evaluate(df, ["x1", "x2"], "grade", "classification",
                                       variable_types={"x1": "numeric", "x2": "numeric"},
                                       with_shap=False)

    roc, cal, dca = res["roc"], res["calibration"], res["dca"]
    assert roc["multi"] is True and len(roc["classes"]) == 3
    assert all("auc" in c and "youden_threshold" in c for c in roc["classes"])
    assert roc["macro_auc"] == round(float(np.mean([c["auc"] for c in roc["classes"]])), 4)
    assert roc["auc"] == roc["macro_auc"]

    assert cal["multi"] is True and len(cal["classes"]) == 3
    assert all(len(c["prob_true"]) == len(c["prob_pred"]) for c in cal["classes"])

    assert dca["multi"] is True and len(dca["classes"]) == 3
    assert len(dca["thresholds"]) == len(dca["net_benefit_model"])
    assert dca["best_net_benefit"] is not None
    # 小样本（n=240）走 sigmoid 校准，且每类曲线齐全
    assert res["calibration_model"].startswith("OOF校准（sigmoid")


# ------------------------------------------------------------------
# 规模自适应（P0：大样本自动降折/减树/抽样）
# ------------------------------------------------------------------

def _toy_binary(n=240, seed=3):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": (x1 + 0.5 * x2 > 0).astype(int)})


def test_large_data_switches_to_fast_mode(monkeypatch):
    """训练集超过阈值时应切换快速模式：降折 + 减树，并在 warnings 中明确告知。"""
    monkeypatch.setattr(ml_engine, "FAST_MODE_ROWS", 10)
    monkeypatch.setattr(ml_engine, "FAST_CV_FOLDS", 3)
    monkeypatch.setattr(ml_engine, "FAST_RF_TREES", 20)
    df = _toy_binary(240)
    res = ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                       variable_types={"x1": "numeric", "x2": "numeric"},
                                       with_shap=False)
    assert res["speed"]["fast_mode"] is True
    assert res["speed"]["cv_folds"] == 3
    assert res["speed"]["estimator"] == "hist_gb"
    assert any("快速训练模式" in w for w in res["warnings"])
    names = [m["name"] for m in res["models"]]
    assert "hist_gb" in names and "random_forest" not in names   # 大样本用 HGB 替换 RF
    rf = next(e for _l, e in ml_engine._candidate_models("classification", 20).values()
              if isinstance(e, ml_engine.RandomForestClassifier))
    assert rf.n_estimators == 20


def test_small_data_keeps_standard_mode():
    df = _toy_binary(240)
    res = ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                       variable_types={"x1": "numeric", "x2": "numeric"},
                                       with_shap=False)
    assert res["speed"]["fast_mode"] is False
    assert res["speed"]["cv_folds"] == ml_engine.CV_FOLDS
    assert res["speed"]["rf_trees"] == ml_engine.RF_TREES


# ------------------------------------------------------------------
# 训练结果缓存（P0：同一数据集版本/同一口径重复训练直接命中）
# ------------------------------------------------------------------

def test_train_result_cache_hit_and_isolation():
    df = _toy_binary(200)
    ml_engine.clear_cache()
    key = "test_ds_cache:1"
    vt = {"x1": "numeric", "x2": "numeric"}
    r1 = ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                      variable_types=vt, with_shap=False, cache_key=key)
    r2 = ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                      variable_types=vt, with_shap=False, cache_key=key)
    assert r1["from_cache"] is False and r2["from_cache"] is True
    assert r1["n_train"] == r2["n_train"] and r1["models"] == r2["models"]

    # 换数据集版本（清洗/确认后 rev+1）不得命中旧结果
    r3 = ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                      variable_types=vt, with_shap=False,
                                      cache_key="test_ds_cache:2")
    assert r3["from_cache"] is False

    # 换特征集合也不得命中
    r4 = ml_engine.train_and_evaluate(df, ["x1"], "y", "classification",
                                      variable_types=vt, with_shap=False, cache_key=key)
    assert r4["from_cache"] is False

    assert ml_engine.clear_cache("test_ds_cache") >= 2
    assert ml_engine.clear_cache("test_ds_cache") == 0


def test_cache_is_lru_bounded():
    df = _toy_binary(120, seed=5)
    ml_engine.clear_cache()
    n = ml_engine.TRAIN_CACHE_SIZE
    for i in range(n + 2):
        ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                     variable_types={"x1": "numeric", "x2": "numeric"},
                                     with_shap=False, cache_key=f"test_ds_lru:{i}")
    assert len(ml_engine._TRAIN_CACHE) <= n
    ml_engine.clear_cache()


def test_cached_payload_can_be_saved_twice():
    """缓存命中的 payload 仍要能保存模型：save_model 不得 pop 掉 `_fitted`。"""
    ds = "test_ds_cache_save"
    df = _toy_binary(160, seed=6)
    key = f"{ds}:1"
    vt = {"x1": "numeric", "x2": "numeric"}
    ml_engine.clear_cache()
    r1 = ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                      variable_types=vt, with_shap=False, cache_key=key)
    ml_engine.save_model(ds, r1)
    r2 = ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                      variable_types=vt, with_shap=False, cache_key=key)
    assert r2["from_cache"] is True
    ml_engine.save_model(ds, r2)              # 命中缓存后再保存，不得抛「模型未训练」
    try:
        assert ml_engine.load_model(ds)["features"] == ["x1", "x2"]
    finally:
        ml_engine.invalidate_model(ds)
        ml_engine.clear_cache()


# ------------------------------------------------------------------
# P1：结构性泄露自动剔除 + 宽表降维
# ------------------------------------------------------------------

def test_structural_leak_column_is_auto_dropped():
    """与结局几乎等价的列（单变量 AUC≈1）必须自动剔除，避免给出虚高指标。"""
    df = _toy_binary(200, seed=7)
    df["y_copy"] = df["y"]                    # 结构性泄露：结局副本
    res = ml_engine.train_and_evaluate(df, ["x1", "x2", "y_copy"], "y", "classification",
                                       variable_types={"x1": "numeric", "x2": "numeric",
                                                       "y_copy": "numeric"},
                                       with_shap=False)
    assert res.get("skipped") is not True
    assert res["leak_columns"] == ["y_copy"]
    assert "y_copy" not in res["features"]
    assert any("泄露" in w for w in res["warnings"])


def test_all_features_leaky_returns_skipped_not_error():
    """全部特征都是泄露列时应返回 skipped（可解释的跳过），而不是抛异常当系统错误。"""
    df = _toy_binary(120, seed=11)
    df["copy_a"] = df["y"]
    df["copy_b"] = df["y"]
    res = ml_engine.train_and_evaluate(df, ["copy_a", "copy_b"], "y", "classification",
                                       variable_types={"copy_a": "numeric", "copy_b": "numeric"},
                                       with_shap=False)
    assert res["skipped"] is True
    assert "泄露" in res["reason"]
    assert res["skip_detail"]["leak_columns"]


def test_wide_table_is_screened_to_top_k(monkeypatch):
    """宽表（>阈值）应在训练集上做方差/F检验/共线性筛选，只保留 top-k。"""
    rng = np.random.default_rng(8)
    n = 240
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = {f"noise_{i}": rng.normal(size=n) for i in range(120)}
    df = pd.DataFrame({"x1": x1, "x2": x2, **noise})
    df["y"] = (x1 + 0.5 * x2 > 0).astype(int)
    monkeypatch.setattr(ml_engine, "SCREEN_TRIGGER_FEATURES", 50)
    monkeypatch.setattr(ml_engine, "SCREEN_TOP_K", 10)
    feats = ["x1", "x2"] + list(noise)
    res = ml_engine.train_and_evaluate(df, feats, "y", "classification",
                                       variable_types={f: "numeric" for f in feats},
                                       with_shap=False)
    assert res["feature_screening"]["triggered"] is True
    assert res["feature_screening"]["original"] == len(feats)
    assert len(res["features"]) <= 10
    assert res["feature_screening"]["dropped_count"] >= len(feats) - 10
    assert set(res["features"]) >= {"x1", "x2"}
    assert any("宽表自动筛选" in w for w in res["warnings"])


def _wide_binary_medical(n=240, n_feat=120, seed=21):
    """构造一张“全是 0/1 列”的宽表：cat_7 与结局强相关，其余为噪声。"""
    rng = np.random.default_rng(seed)
    data = {f"cat_{i}": rng.integers(0, 2, n) for i in range(n_feat)}
    key = data[f"cat_{min(7, n_feat - 1)}"]
    y = np.where(key == 1, rng.random(n) < 0.8, rng.random(n) < 0.2).astype(int)
    df = pd.DataFrame(data)
    df["y"] = y
    return df, [f"cat_{i}" for i in range(n_feat)]


def test_wide_binary_categorical_columns_are_screened(monkeypatch):
    """全是 0/1 的类别列也必须参与宽表筛选（旧逻辑会原样保留上千列）。"""
    df, feats = _wide_binary_medical()
    monkeypatch.setattr(ml_engine, "SCREEN_TRIGGER_FEATURES", 50)
    monkeypatch.setattr(ml_engine, "SCREEN_TOP_K", 10)
    res = ml_engine.train_and_evaluate(df, feats, "y", "classification",
                                       variable_types={f: "categorical" for f in feats},
                                       with_shap=False)
    assert res["feature_screening"]["triggered"] is True
    assert len(res["features"]) <= 10
    assert "cat_7" in res["features"]
    assert res["feature_screening"]["selected"] <= 10


def test_wide_binary_columns_are_used_as_numeric(monkeypatch):
    """宽表里的 0/1 列在建模时按数值处理，避免上千列独热编码。"""
    df, feats = _wide_binary_medical(n=160, n_feat=60, seed=22)
    monkeypatch.setattr(ml_engine, "SCREEN_TRIGGER_FEATURES", 50)
    res = ml_engine.train_and_evaluate(df, feats, "y", "classification",
                                       variable_types={f: "categorical" for f in feats},
                                       with_shap=False)
    assert res.get("skipped") is not True
    assert not res["categorical_features"]
    assert set(res["features"]) <= set(res["numeric_features"])
    assert any("0/1 二值列已按数值" in w for w in res["warnings"])


def test_narrow_binary_columns_keep_categorical_semantics():
    """窄表（≤阈值）不改变 0/1 列的分类语义，避免影响小表建模习惯。"""
    df, feats = _wide_binary_medical(n=120, n_feat=4, seed=23)
    res = ml_engine.train_and_evaluate(df, feats, "y", "classification",
                                       variable_types={f: "categorical" for f in feats},
                                       with_shap=False)
    assert set(feats) <= set(res["categorical_features"])
    assert not set(feats) & set(res["numeric_features"])


def _wide_regression(n=600, n_feat=60, seed=31):
    rng = np.random.default_rng(seed)
    data = {f"x_{i}": rng.normal(size=n) for i in range(n_feat)}
    y = 2.0 * data["x_3"] + 0.8 * data["x_9"] + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(data)
    df["y"] = y
    return df, [f"x_{i}" for i in range(n_feat)]


def test_regression_wide_table_relaxes_topk_for_large_n(monkeypatch):
    """回归宽表在训练集 ≥500 行时放宽到 SCREEN_TOP_K_REGRESSION，保 R²。"""
    df, feats = _wide_regression(n=800, n_feat=60)
    monkeypatch.setattr(ml_engine, "SCREEN_TRIGGER_FEATURES", 50)
    monkeypatch.setattr(ml_engine, "SCREEN_TOP_K", 10)
    monkeypatch.setattr(ml_engine, "SCREEN_TOP_K_REGRESSION", 20)
    monkeypatch.setattr(ml_engine, "SCREEN_TOP_K_MIN_ROWS", 500)
    monkeypatch.setattr(ml_engine, "RF_TREES", 20)
    res = ml_engine.train_and_evaluate(df, feats, "y", "regression",
                                       variable_types={f: "numeric" for f in feats},
                                       with_shap=False)
    assert res["feature_screening"]["triggered"] is True
    assert res["feature_screening"]["top_k"] == 20
    assert 10 < len(res["features"]) <= 20
    assert {"x_3", "x_9"} <= set(res["features"])


def test_small_regression_wide_table_keeps_base_topk(monkeypatch):
    """回归小样本（训练集 <500 行）仍用 base top-k，避免过拟合。"""
    df, feats = _wide_regression(n=200, n_feat=60, seed=32)
    monkeypatch.setattr(ml_engine, "SCREEN_TRIGGER_FEATURES", 50)
    monkeypatch.setattr(ml_engine, "SCREEN_TOP_K", 10)
    monkeypatch.setattr(ml_engine, "SCREEN_TOP_K_REGRESSION", 20)
    monkeypatch.setattr(ml_engine, "SCREEN_TOP_K_MIN_ROWS", 500)
    monkeypatch.setattr(ml_engine, "RF_TREES", 20)
    res = ml_engine.train_and_evaluate(df, feats, "y", "regression",
                                       variable_types={f: "numeric" for f in feats},
                                       with_shap=False)
    assert res["feature_screening"]["top_k"] == 10
    assert len(res["features"]) <= 10


def test_regression_topk_capped_by_sample_size(monkeypatch):
    """回归宽表的特征数不得超过训练集的 1/5，防止几十行配几十列导致 R² 为负。"""
    df, feats = _wide_regression(n=80, n_feat=60, seed=33)
    monkeypatch.setattr(ml_engine, "RF_TREES", 20)
    res = ml_engine.train_and_evaluate(df, feats, "y", "regression",
                                       variable_types={f: "numeric" for f in feats},
                                       with_shap=False)
    assert res["feature_screening"]["triggered"] is True
    assert res["feature_screening"]["top_k"] <= 12     # 训练集 60 行 → 最多 12 个特征
    assert len(res["features"]) <= 12


def _imbalanced_binary(n=200, n_pos=8, seed=41):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    return pd.DataFrame({"x": x, "y": y})


def test_rare_class_between_5_and_10_warns_but_models():
    """最少类别 5~10 例时不再“样本量严重不足”跳过，而是给出波动警示。"""
    df = _imbalanced_binary(n=200, n_pos=8)
    res = ml_engine.train_and_evaluate(df, ["x"], "y", "classification",
                                       variable_types={"x": "numeric"}, with_shap=False)
    assert res.get("skipped") is not True
    assert any("最少类别" in w and "波动" in w for w in res["warnings"])
    assert res["speed"]["cv_folds"] == 5


def test_rare_class_below_5_is_skipped_with_detail():
    """最少类别 <5 例才跳过，并给出具体类别/例数，而不是笼统的“样本量严重不足”。"""
    df = _imbalanced_binary(n=200, n_pos=3, seed=42)
    res = ml_engine.train_and_evaluate(df, ["x"], "y", "classification",
                                       variable_types={"x": "numeric"}, with_shap=False)
    assert res["skipped"] is True
    assert "最少类别" in res["reason"] and "3" in res["reason"]
    assert res["skip_detail"]["min_class"] == 3
    assert res["skip_detail"]["min_class_threshold"] == ml_engine.MIN_CLASS_FOR_MODELING


def test_narrow_table_is_not_screened():
    df = _toy_binary(200, seed=9)
    res = ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                       variable_types={"x1": "numeric", "x2": "numeric"},
                                       with_shap=False)
    assert res["feature_screening"]["triggered"] is False
    assert res["features"] == ["x1", "x2"]


# ------------------------------------------------------------------
# P2：大样本快速模式接入 HistGradientBoosting + 解释链路
# ------------------------------------------------------------------

def test_fast_mode_includes_hist_gradient_boosting(monkeypatch):
    monkeypatch.setattr(ml_engine, "FAST_MODE_ROWS", 10)
    monkeypatch.setattr(ml_engine, "FAST_RF_TREES", 20)
    df = _toy_binary(300, seed=11)
    res = ml_engine.train_and_evaluate(df, ["x1", "x2"], "y", "classification",
                                       variable_types={"x1": "numeric", "x2": "numeric"},
                                       with_shap=False)
    assert res["speed"]["fast_mode"] is True
    names = [m["name"] for m in res["models"]]
    assert "hist_gb" in names and "random_forest" not in names
    assert res["feature_importance"], "推荐模型不论是否为 HGB，都必须给出特征重要性"


def test_hgb_feature_importance_uses_permutation(monkeypatch):
    """HistGradientBoosting 无原生 importance，必须退化为置换重要性而不是返回空。"""
    from sklearn.pipeline import Pipeline
    df = _toy_binary(240, seed=12)
    X, y = df[["x1", "x2"]], df["y"].to_numpy()
    pipe = Pipeline([
        ("prep", ml_engine.build_preprocessor(["x1", "x2"], [])),
        ("model", ml_engine.HistGradientBoostingClassifier(max_iter=30, random_state=0)),
    ])
    pipe.fit(X, y)
    items = ml_engine._feature_importance(pipe, ["x1", "x2"], [], "classification",
                                          [0, 1], X, y)
    assert items and {i["variable"] for i in items} == {"x1", "x2"}
    assert sum(i["importance"] for i in items) > 0.99


def test_near_perfect_auc_adds_warning():
    """测试集 AUC 接近完美时必须提示排查泄露/过拟合，而不是让用户误以为可用。"""
    rng = np.random.default_rng(21)
    n = 600
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    df = pd.DataFrame({"a": a, "b": b, "y": ((a == 1) & (b == 1)).astype(int)})
    res = ml_engine.train_and_evaluate(df, ["a", "b"], "y", "classification",
                                       variable_types={"a": "numeric", "b": "numeric"},
                                       with_shap=False)
    assert any("接近完美" in w for w in res["warnings"])


def test_shap_on_hgb_degrades_with_clear_message(monkeypatch):
    from sklearn.pipeline import Pipeline
    df = _toy_binary(200, seed=13)
    X, y = df[["x1", "x2"]], df["y"].to_numpy()
    pipe = Pipeline([
        ("prep", ml_engine.build_preprocessor(["x1", "x2"], [])),
        ("model", ml_engine.HistGradientBoostingClassifier(max_iter=30, random_state=0)),
    ])
    pipe.fit(X, y)
    with pytest.raises(ml_engine.MLError) as exc:
        ml_engine.shap_global(pipe, X, "classification", ["x1", "x2"], [])
    assert "暂不支持 SHAP" in str(exc.value)

