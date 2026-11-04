# medical-data-analysis/backend/tests/test_app.py
"""FastAPI 接口层测试：上传解析、变量确认、清洗、统计分析、机器学习接口与统一错误格式。"""

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
import llm_service
from config import CORS_ORIGINS, MAX_UPLOAD_MB

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def uploaded(client, medical_df):
    content = medical_df.to_csv(index=False).encode("utf-8-sig")
    resp = client.post("/api/dataset/upload",
                       files={"file": ("medical.csv", content, "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------
# 基础
# ------------------------------------------------------------------

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_cors_never_uses_wildcard():
    assert "*" not in CORS_ORIGINS


def test_upload_gbk_file(client):
    content = "年龄,性别,空腹血糖\n55,1,5.6\n62,0,7.1\n48,1,6.9\n".encode("gbk")
    r = client.post("/api/dataset/upload", files={"file": ("gbk.csv", content, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parse_meta"]["encoding"].lower().startswith("gb")
    assert body["n_rows"] == 3 and body["n_cols"] == 3


def test_upload_rejects_unsupported_format(client):
    r = client.post("/api/dataset/upload", files={"file": ("x.pdf", b"abc", "application/pdf")})
    assert r.status_code == 400
    body = r.json()
    assert {"code", "message", "detail"} <= set(body)


def test_upload_rejects_oversize(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_MB", 0)
    r = client.post("/api/dataset/upload", files={"file": ("a.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 413
    assert "超过限制" in r.json()["message"]


# ------------------------------------------------------------------
# 数据集校验与确认
# ------------------------------------------------------------------

def test_upload_returns_validation_and_variables(uploaded):
    assert uploaded["validation"]["is_medical"] is True
    names = {v["name"] for v in uploaded["variables"]}
    assert "患者ID" in names and "结局" in names
    pid = next(v for v in uploaded["variables"] if v["name"] == "患者ID")
    assert pid["excluded"] is True


def test_confirm_requires_force_for_non_medical(client, non_medical_df):
    content = non_medical_df.to_csv(index=False).encode("utf-8-sig")
    up = client.post("/api/dataset/upload", files={"file": ("sales.csv", content, "text/csv")}).json()
    assert up["validation"]["is_medical"] is False
    r = client.post(f"/api/dataset/{up['dataset_id']}/confirm", json={"types": {}})
    assert r.status_code == 422
    assert "终止分析" in r.json()["message"]
    r2 = client.post(f"/api/dataset/{up['dataset_id']}/confirm",
                     json={"types": {}, "force_continue": True})
    assert r2.status_code == 200
    assert any("手动放行" in n for n in r2.json()["notes"])


def test_confirm_applies_user_type_override(client, uploaded):
    did = uploaded["dataset_id"]
    r = client.post(f"/api/dataset/{did}/confirm", json={"types": {"性别": "numeric"}})
    assert r.status_code == 200
    gender = next(v for v in r.json()["variables"] if v["name"] == "性别")
    assert gender["confirmed_type"] == "numeric"


# ------------------------------------------------------------------
# 清洗
# ------------------------------------------------------------------

def test_cleaning_report_and_apply(client, uploaded):
    did = uploaded["dataset_id"]
    rep = client.get(f"/api/dataset/{did}/cleaning-report").json()
    assert rep["n_duplicated_rows"] == 2
    r = client.post(f"/api/dataset/{did}/clean",
                    json={"drop_duplicates": True, "outlier_action": "keep"})
    assert r.status_code == 200
    assert r.json()["n_rows"] == rep["n_rows"] - 2


# ------------------------------------------------------------------
# 统计分析接口
# ------------------------------------------------------------------

def test_describe_endpoint(client, uploaded):
    did = uploaded["dataset_id"]
    r = client.post("/api/analysis/describe", json={"dataset_id": did})
    body = r.json()
    assert r.status_code == 200
    names = [x["field"] for x in body["numeric"]]
    assert "年龄" in names and "空腹血糖" in names
    assert any(x["field"] == "分组" for x in body["categorical"])


def test_univariate_endpoint(client, uploaded):
    did = uploaded["dataset_id"]
    r = client.post("/api/analysis/univariate",
                    json={"dataset_id": did, "target": "年龄", "group": "分组", "target_type": "numeric"})
    body = r.json()
    assert r.status_code == 200 and body["valid"] is True
    assert body["test"]["p"] is not None
    assert body["group_levels"] == ["对照组", "试验组"] or len(body["group_levels"]) == 2


def test_correlation_endpoint(client, uploaded):
    did = uploaded["dataset_id"]
    r = client.post("/api/analysis/correlation",
                    json={"dataset_id": did, "fields": ["年龄", "体质指数", "收缩压"]})
    body = r.json()
    assert len(body["columns"]) == 3
    assert body["r"][0][0] == 1.0
    assert body["p_fdr"][0][1] is not None


def test_multivariable_endpoint(client, uploaded):
    did = uploaded["dataset_id"]
    r = client.post("/api/analysis/multivariable",
                    json={"dataset_id": did, "outcome": "结局",
                          "predictors": ["空腹血糖"], "covariates": ["年龄", "性别"]})
    body = r.json()
    assert r.status_code == 200, body
    assert body["task"] == "logistic"
    assert body["terms"][0]["variable"] == "空腹血糖"
    assert body["n_used"] > 0 and body["epv"] is not None


def test_multivariable_endpoint_rejects_multilevel_outcome(client, uploaded):
    """连续结局被强制指定为 logistic 时，应返回明确错误而非静默计算。"""
    did = uploaded["dataset_id"]
    r = client.post("/api/analysis/multivariable",
                    json={"dataset_id": did, "outcome": "年龄",
                          "predictors": ["空腹血糖"], "task": "logistic"})
    assert r.status_code == 422
    assert "二分类结局" in r.json()["message"]


def test_multivariable_endpoint_supports_linear_task(client, uploaded):
    did = uploaded["dataset_id"]
    r = client.post("/api/analysis/multivariable",
                    json={"dataset_id": did, "outcome": "年龄",
                          "predictors": ["空腹血糖"], "covariates": ["体质指数"], "task": "linear"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "linear"
    assert body["terms"][0]["estimate_label"] == "回归系数 β"


def test_screen_endpoint(client, uploaded):
    did = uploaded["dataset_id"]
    r = client.post("/api/analysis/screen",
                    json={"dataset_id": did, "outcome": "结局",
                          "candidates": ["年龄", "体质指数", "空腹血糖", "吸烟"]})
    body = r.json()
    assert body["n_tests"] == 4
    assert all("p_fdr" in row for row in body["rows"])


def test_plot_endpoints(client, uploaded):
    did = uploaded["dataset_id"]
    for path in ("histogram", "boxplot", "qqplot"):
        r = client.post(f"/api/analysis/{path}", json={"dataset_id": did, "field": "年龄"})
        assert r.status_code == 200, r.text
    r = client.post("/api/analysis/boxplot",
                    json={"dataset_id": did, "field": "年龄", "group": "分组"})
    assert len(r.json()["groups"]) == 2


def test_outcome_candidates_endpoint(client, uploaded):
    did = uploaded["dataset_id"]
    r = client.get(f"/api/dataset/{did}/outcome-candidates")
    assert "结局" in [c["field"] for c in r.json()["candidates"]]


# ------------------------------------------------------------------
# 错误格式
# ------------------------------------------------------------------

def test_unknown_dataset_returns_404_with_unified_error(client):
    r = client.get("/api/dataset/not_exist")
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == 404 and body["message"] and "detail" in body


def test_unknown_field_returns_422(client, uploaded):
    r = client.post("/api/analysis/univariate",
                    json={"dataset_id": uploaded["dataset_id"], "target": "不存在",
                          "group": "分组", "target_type": "numeric"})
    assert r.status_code == 422
    assert "字段不存在" in r.json()["message"]


def test_upload_limit_constant_exposed():
    # 常量已在 config 中定义，此处仅确认默认值合理
    assert MAX_UPLOAD_MB > 0


def test_automl_prepare_returns_400_on_data_error(client):
    """生存数据非法（事件列缺失）应返回可修正的 400 与建议，而不是 500。"""
    content = ("TIME,STATUS,DURATION\n"
               "10,dead,1\n20,,2\n30,alive,3\n40,dead,4\n50,,5\n60,alive,6\n").encode("utf-8")
    up = client.post("/api/dataset/upload",
                     files={"file": ("survival.csv", content, "text/csv")})
    assert up.status_code == 200, up.text
    ds = up.json()["dataset_id"]
    client.post(f"/api/dataset/{ds}/confirm", json={"types": {}, "force_continue": True})
    r = client.post("/api/automl/prepare", json={"dataset_id": ds})
    assert r.status_code == 400, r.text
    body = r.json()
    assert "缺失" in body["message"]
    assert body.get("suggestion")


# ------------------------------------------------------------------
# 机器学习接口
# ------------------------------------------------------------------

WDBC_FEATURES = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                 "smoothness_mean", "compactness_mean", "concavity_mean",
                 "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"]


@pytest.fixture
def wdbc_uploaded(client):
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "sample_data", "wdbc.csv"))
    up = client.post("/api/dataset/upload",
                     files={"file": ("wdbc.csv", df.to_csv(index=False).encode("utf-8"),
                                     "text/csv")}).json()
    client.post(f"/api/dataset/{up['dataset_id']}/confirm", json={"types": {}})
    return up


def test_ml_model_is_empty_before_training(client, wdbc_uploaded):
    did = wdbc_uploaded["dataset_id"]
    r = client.get(f"/api/ml/model/{did}")
    assert r.status_code == 200
    assert r.json()["trained"] is False
    assert "请先完成当前数据集的建模" in r.json()["message"]


def test_ml_train_and_model_info(client, wdbc_uploaded):
    did = wdbc_uploaded["dataset_id"]
    r = client.post("/api/ml/train", json={"dataset_id": did, "features": WDBC_FEATURES,
                                           "target": "diagnosis", "task": "classification"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["best_model"]["name"] in ("logistic", "random_forest")
    assert body["roc"]["auc"] > 0.9
    assert "calibration" in body and "dca" in body
    assert "_fitted" not in body

    info = client.get(f"/api/ml/model/{did}").json()
    assert info["trained"] is True
    assert "discarded" not in info                     # 模型可用时不带“已丢弃”状态
    assert info["features"] == WDBC_FEATURES
    assert info["numeric_ranges"], "需返回训练集数值范围用于外推警示"
    assert info["categories"] == {}


def test_ml_train_async_task_lifecycle(client, wdbc_uploaded):
    """异步训练：提交拿 task_id → 轮询 → done 带结果且模型已保存；未知任务 404。"""
    import time
    did = wdbc_uploaded["dataset_id"]
    r = client.post("/api/ml/train/async", json={"dataset_id": did, "features": WDBC_FEATURES,
                                                 "target": "diagnosis", "task": "classification",
                                                 "with_shap": False})
    assert r.status_code == 200, r.text
    tid = r.json()["task_id"]
    assert r.json()["status"] == "running"

    st = None
    deadline = time.time() + 120
    while time.time() < deadline:
        st = client.get(f"/api/tasks/{tid}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert st is not None and st["status"] == "done", st
    assert st["progress"] == 100 and st["result"]["best_model"]["name"] in ("logistic", "random_forest")
    assert "_fitted" not in st["result"]

    info = client.get(f"/api/ml/model/{did}").json()
    assert info["trained"] is True          # 后台任务确实完成了模型保存

    miss = client.get("/api/tasks/not-a-task")
    assert miss.status_code == 404
    assert "任务不存在" in miss.json()["message"]


def test_ml_predict_returns_probability_and_shap(client, wdbc_uploaded):
    import pandas as pd
    did = wdbc_uploaded["dataset_id"]
    client.post("/api/ml/train", json={"dataset_id": did, "features": WDBC_FEATURES,
                                       "target": "diagnosis", "task": "classification",
                                       "with_shap": False})
    df = pd.read_csv(os.path.join(ROOT, "sample_data", "wdbc.csv"))
    sample = {f: float(df[f].iloc[0]) for f in WDBC_FEATURES}
    r = client.post("/api/ml/predict", json={"dataset_id": did, "sample": sample})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_class"] in ("B", "M")
    assert body["shap_force_plot"].startswith("iVBOR")
    assert "不构成临床诊断依据" in body["disclaimer"]


def test_ml_predict_without_model_returns_422(client, wdbc_uploaded):
    r = client.post("/api/ml/predict", json={"dataset_id": wdbc_uploaded["dataset_id"],
                                             "sample": {"radius_mean": 12.0}})
    assert r.status_code == 422
    assert "尚未完成建模" in r.json()["message"]


def test_non_predictive_regression_model_is_not_saved(client):
    """回归测试集 R²<0（不如直接用均值）时不落盘：无可用模型，预测接口拒绝。"""
    import random
    random.seed(99)
    lines = ["x1,x2,x3,score"]
    for _ in range(140):
        lines.append(",".join(str(round(random.gauss(0, 1), 3)) for _ in range(3))
                     + "," + str(round(random.gauss(50, 10), 2)))
    up = client.post("/api/dataset/upload",
                     files={"file": ("noise.csv", ("\n".join(lines) + "\n").encode("utf-8"),
                                     "text/csv")}).json()
    did = up["dataset_id"]
    client.post(f"/api/dataset/{did}/confirm", json={"types": {}, "force_continue": True})
    body = client.post("/api/ml/train",
                       json={"dataset_id": did, "features": ["x1", "x2", "x3"],
                             "target": "score", "task": "regression",
                             "with_shap": False}).json()
    assert body["predictive"] is False
    assert body["saved"] is None
    assert any("未保存" in w for w in body["warnings"])

    info = client.get(f"/api/ml/model/{did}").json()
    assert info["trained"] is False
    assert info["discarded"]["r2"] < 0                 # 预测页据此显示“上一次训练已丢弃”
    assert "已丢弃" in info["message"] and "R²" in info["message"]
    r = client.post("/api/ml/predict", json={"dataset_id": did,
                                             "sample": {"x1": 0.5, "x2": -0.2, "x3": 1.1}})
    assert r.status_code == 422
    assert "尚未完成建模" in r.json()["message"]


def test_uploading_new_dataset_invalidates_old_model(client, wdbc_uploaded, medical_df):
    did = wdbc_uploaded["dataset_id"]
    client.post("/api/ml/train", json={"dataset_id": did, "features": WDBC_FEATURES,
                                       "target": "diagnosis", "with_shap": False})
    assert client.get(f"/api/ml/model/{did}").json()["trained"] is True
    client.post("/api/dataset/upload",
                files={"file": ("new.csv", medical_df.to_csv(index=False).encode("utf-8-sig"),
                                "text/csv")})
    assert client.get(f"/api/ml/model/{did}").json()["trained"] is False


def test_ml_delete_model(client, wdbc_uploaded):
    did = wdbc_uploaded["dataset_id"]
    client.post("/api/ml/train", json={"dataset_id": did, "features": WDBC_FEATURES,
                                       "target": "diagnosis", "with_shap": False})
    r = client.delete(f"/api/ml/model/{did}")
    assert r.json()["invalidated"] is True
    assert client.get(f"/api/ml/model/{did}").json()["trained"] is False


# ------------------------------------------------------------------
# 大模型接口
# ------------------------------------------------------------------

def test_llm_status_defaults_to_not_configured(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    r = client.get("/api/llm/status").json()
    assert r["configured"] is False
    assert "内存中" in r["note"]


def test_llm_test_without_key_returns_not_configured(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    r = client.post("/api/llm/test", json={"provider": "modelarts_pangu"}).json()
    assert r["connected"] is False and r["status"] == "not_configured"


def test_llm_config_never_echoes_key(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    r = client.post("/api/llm/config", json={
        "provider": "modelarts_pangu", "base_url": "https://x/api/v2",
        "model": "pangu", "api_key": "SECRET"}).json()
    assert r["configured"] is True
    assert "SECRET" not in str(r)


def test_llm_test_mocked_success(client, monkeypatch):
    monkeypatch.setattr(llm_service, "_post", lambda *a, **k: {"choices": []})
    r = client.post("/api/llm/test", json={"provider": "modelarts_pangu",
                                           "base_url": "https://x/api/v2",
                                           "model": "pangu", "api_key": "K"}).json()
    assert r["connected"] is True and "latency_ms" in r


# ------------------------------------------------------------------
# 安全：访问令牌鉴权 / LLM base_url 校验 / 模型路径穿越
# ------------------------------------------------------------------

def test_auth_token_required_when_enabled(client, monkeypatch):
    monkeypatch.setattr(app_module, "API_TOKEN", "test-secret")
    # 未携带令牌：业务接口 401，健康检查放行
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/health").status_code == 200
    # 错误令牌 401；正确令牌（两种头）放行
    assert client.get("/api/config", headers={"X-API-Token": "wrong"}).status_code == 401
    assert client.get("/api/config", headers={"X-API-Token": "test-secret"}).status_code == 200
    assert client.get("/api/config",
                      headers={"Authorization": "Bearer test-secret"}).status_code == 200


def test_llm_config_rejects_plain_http_url(client):
    r = client.post("/api/llm/config", json={"provider": "custom",
                                             "base_url": "http://evil.example.com/v1"})
    assert r.status_code == 422
    assert "https" in r.json()["message"]


def test_llm_config_rejects_private_ip(client):
    r = client.post("/api/llm/config", json={"provider": "custom",
                                             "base_url": "https://192.168.1.10/v1"})
    assert r.status_code == 422
    assert "内网" in r.json()["message"]


def test_llm_config_allows_loopback_http(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    r = client.post("/api/llm/config", json={"provider": "custom",
                                             "base_url": "http://127.0.0.1:11434/v1"})
    assert r.status_code == 200


def test_ml_model_delete_rejects_path_traversal(client, tmp_path):
    # 非法格式 id（含点号/斜杠等路径字符）：进入路由后被 _model_path 白名单拒绝
    assert client.delete("/api/ml/model/not.valid.id").status_code == 400
    # 编码后的穿越序列：路由层规范化 404 或校验层 400，二者都不得触碰文件系统
    victim = tmp_path / "model_.._.._history.joblib"
    victim.write_bytes(b"keep-me")
    r = client.delete("/api/ml/model/..%2F..%2Fhistory")
    assert r.status_code in (400, 404)
    assert victim.exists()


def test_llm_test_rejects_plain_http_url(client):
    r = client.post("/api/llm/test", json={"provider": "custom",
                                           "base_url": "http://evil.example.com/v1",
                                           "model": "m", "api_key": "K"}).json()
    assert r["connected"] is False and r["status"] == "invalid_url"


# ------------------------------------------------------------------
# 安全：历史查询 limit 上限
# ------------------------------------------------------------------

def test_history_limit_is_capped(client, monkeypatch):
    captured = {}

    def fake_list_history(limit):
        captured["limit"] = limit
        return []

    monkeypatch.setattr(app_module.storage, "list_history", fake_list_history)
    assert client.get("/api/history", params={"limit": 10**9}).status_code == 200
    assert captured["limit"] == 500
    client.get("/api/history", params={"limit": -5})
    assert captured["limit"] == 1


def test_automl_analyze_threshold_is_clamped(client, uploaded, monkeypatch):
    """leakage_threshold 传极端值不得静默关闭泄露检测：钳制到 [0.5, 0.999]。"""
    captured = {}

    def fake_run_auto(df, schema, *a, **kw):
        captured["threshold"] = a[-1] if a else kw.get("leakage_threshold")
        return {"ok": True}

    monkeypatch.setattr("automl.pipeline.run_auto", fake_run_auto)
    r = client.post("/api/automl/analyze",
                    json={"dataset_id": uploaded["dataset_id"], "target": "结局",
                          "leakage_threshold": 999})
    assert r.status_code == 200
    assert captured["threshold"] == 0.999
