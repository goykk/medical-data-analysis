# medical-data-analysis/backend/tests/test_uci_pipeline.py
"""UCI 真实临床数据集等价样本的端到端验收。

真实 UCI 文件由使用者本地提供、不入库，因此这里按它们的真实病灶构造等价样本：
- 慢性肾病（chronic kidney disease）：class 列混入尾部制表符、数值列用 "?" 表缺失
- 宫颈癌（cervical cancer）：STDs: Time since * 两列近乎全为 "?"
- 心脏病（Cleveland）、糖尿病（Pima）：数值列混入 "?" / " ? "
- 乳腺癌（WDBC，仓库自带）：干净数据

链路统一走 HTTP：上传 → 确认 → 清洗，覆盖 parse → trim → 缺失标记 → 类型推断 → 清洗。
"""

import io
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import analyzer
import app as app_module

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DIR = os.path.join(ROOT, "sample_data")


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


def upload(client, df, filename):
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    r = client.post("/api/dataset/upload",
                    files={"file": (filename, buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


def upload_bytes(client, content, filename):
    r = client.post("/api/dataset/upload", files={"file": (filename, content, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


def confirm_clean(client, did, dedupe=True):
    r1 = client.post(f"/api/dataset/{did}/confirm", json={"types": {}, "force_continue": True})
    assert r1.status_code == 200, r1.text
    r2 = client.post(f"/api/dataset/{did}/clean",
                     json={"drop_duplicates": dedupe, "missing_strategy": "none",
                           "outlier_strategy": "none"})
    assert r2.status_code == 200, r2.text
    return r1.json(), r2.json()


def by_name(variables):
    return {v["name"]: v for v in variables}


# ------------------------------------------------------------------
# 1. 慢性肾病：class 不得因尾部制表符凭空多出第三类
# ------------------------------------------------------------------
def test_ckd_class_has_exactly_two_categories(client):
    n = 60
    rng = np.random.default_rng(7)
    ckd = pd.DataFrame({
        "age": [float(v) for v in rng.integers(20, 80, n)],
        "bp": [float(v) for v in rng.integers(60, 180, n)],
        "bgr": [float(v) for v in rng.integers(70, 300, n)],
        "bu": [float(v) for v in rng.integers(10, 120, n)],
        "hemo": rng.normal(12, 2, n).round(1),
        "rbc": ["normal"] * n,
    })
    cls = ["ckd"] * 30 + ["notckd"] * 30
    cls[3] = "ckd\t"       # UCI 原始病灶：尾部制表符
    cls[40] = "notckd\t"
    ckd.insert(0, "class", cls)
    ckd["age"] = ckd["age"].astype(object)
    ckd["bp"] = ckd["bp"].astype(object)
    ckd["bu"] = ckd["bu"].astype(object)
    ckd.loc[5, "age"] = "?"
    ckd.loc[9, "bp"] = "?"
    ckd.loc[12, "bu"] = " ? "   # 带空白的 ?

    did = upload(client, ckd, "chronic_kidney_disease.csv")
    confirmed, cleaned = confirm_clean(client, did)
    schema = by_name(confirmed["variables"])

    # 验收点：唯一值为 2，不再出现带制表符的第三类
    assert schema["class"]["n_unique"] == 2, schema["class"]
    values = sorted(set(app_module.DATASETS[did]["df_raw"]["class"].dropna().astype(str)))
    assert values == ["ckd", "notckd"], values

    # "?" 与 " ? " 均按缺失计，且不影响该列仍被推断为数值型
    assert schema["age"]["n_missing"] == 1, schema["age"]
    assert schema["bu"]["n_missing"] == 1, schema["bu"]
    assert schema["age"]["inferred_type"] == "numeric", schema["age"]
    assert schema["bu"]["inferred_type"] == "numeric", schema["bu"]
    assert cleaned["n_rows"] == n


def test_trim_happens_before_type_inference(client):
    """trim 必须早于类型推断：带空白的数值列应被推断为数值型而非分类。"""
    raw = ('"age","group"\n'
           '" 35.5 ",A\n"42.3\t",B\n" 35.5 ",A\n"51.1 ",C\n"42.3\t",B\n'
           '" 35.5 ",A\n"51.1 ",C\n"42.3\t",B\n" 35.5 ",A\n"51.1 ",C\n')
    df, _ = analyzer.parse_table(raw.encode("utf-8"), "padded.csv")

    assert sorted(set(df["age"].dropna().astype(str))) == ["35.5", "42.3", "51.1"]
    did = upload(client, df, "padded.csv")
    confirmed, _ = confirm_clean(client, did)
    assert by_name(confirmed["variables"])["age"]["inferred_type"] == "numeric"


# ------------------------------------------------------------------
# 2. 宫颈癌：近全缺失列只警示，不阻塞、不自动排除
# ------------------------------------------------------------------
def test_cervical_high_missing_warns_without_blocking(client):
    m = 80
    rng = np.random.default_rng(11)
    cerv = pd.DataFrame({
        # Age 逐行唯一，避免随机整行重复被去重删掉而干扰「是否被阻塞」的判定
        "Age": [float(18 + i) for i in range(m)],
        "Number of sexual partners": [float(v) for v in rng.integers(1, 10, m)],
        "Smokes": rng.choice(["0.0", "1.0"], m),
        "STDs: Number of diagnosis": [float(v) for v in rng.integers(0, 3, m)],
    })
    cerv["STDs: Time since first diagnosis"] = ["?"] * (m - 3) + ["1.0", "2.0", "3.0"]
    cerv["STDs: Time since last diagnosis"] = ["?"] * (m - 2) + ["1.0", "2.0"]

    did = upload(client, cerv, "cervical_cancer.csv")
    confirmed, cleaned = confirm_clean(client, did)
    schema = by_name(confirmed["variables"])

    col = schema["STDs: Time since first diagnosis"]
    assert col["high_missing"] is True, col
    assert col["missing_rate"] > 0.9, col
    assert col["n_missing"] == m - 3, col
    assert col["excluded"] is False, "高缺失只警示，不得自动排除"

    assert schema["Age"]["high_missing"] is False
    # 流程未被阻塞：清洗正常完成
    assert cleaned["n_rows"] == m, cleaned["n_rows"]


# ------------------------------------------------------------------
# 3. 四个 UCI 数据集全部走通 上传 → 确认 → 清洗
# ------------------------------------------------------------------
@pytest.mark.parametrize("label", ["heart_cleveland", "pima_diabetes", "wdbc"])
def test_uci_datasets_walk_through(client, label):
    rng = np.random.default_rng(23)

    if label == "heart_cleveland":
        df = pd.DataFrame({
            "age": [float(v) for v in rng.integers(29, 77, 50)],
            "sex": rng.choice([0.0, 1.0], 50),
            "cp": rng.choice([1.0, 2.0, 3.0, 4.0], 50),
            "trestbps": [float(v) for v in rng.integers(94, 200, 50)],
            "chol": [float(v) for v in rng.integers(126, 564, 50)],
            "thalach": [float(v) for v in rng.integers(71, 202, 50)],
            "num": rng.choice([0.0, 1.0, 2.0], 50),
        })
        df["chol"] = df["chol"].astype(object)
        df["thalach"] = df["thalach"].astype(object)
        df.loc[2, "chol"] = "?"
        df.loc[7, "thalach"] = "?"
    elif label == "pima_diabetes":
        df = pd.DataFrame({
            "Pregnancies": [float(v) for v in rng.integers(0, 12, 40)],
            "Glucose": [float(v) for v in rng.integers(60, 200, 40)],
            "BloodPressure": [float(v) for v in rng.integers(40, 120, 40)],
            "BMI": rng.normal(30, 5, 40).round(1),
            "Age": [float(v) for v in rng.integers(20, 70, 40)],
            "Outcome": rng.choice([0.0, 1.0], 40),
        })
        df["Glucose"] = df["Glucose"].astype(object)
        df.loc[1, "Glucose"] = " ?"
    else:
        df = pd.read_csv(os.path.join(SAMPLE_DIR, "wdbc.csv"))

    did = upload(client, df, f"{label}.csv")
    confirmed, cleaned = confirm_clean(client, did)

    assert confirmed["confirmed"] is True
    assert cleaned["n_rows"] > 0, cleaned
    assert len(cleaned["variables"]) == df.shape[1]


# ------------------------------------------------------------------
# 4. 干净基准数据零副作用（GBK 编码样本，按原始字节走真实编码探测）
# ------------------------------------------------------------------
def test_clean_baseline_has_no_side_effects(client):
    path = os.path.join(SAMPLE_DIR, "medical_sample.csv")
    with open(path, "rb") as fh:
        raw_bytes = fh.read()

    did = upload_bytes(client, raw_bytes, "blood_donation.csv")
    before = app_module.DATASETS[did]["df_raw"]

    confirmed = client.post(f"/api/dataset/{did}/confirm",
                            json={"types": {}, "force_continue": True}).json()
    # 关闭去重与任何改写，使清洗前后可逐值严格比对
    cleaned = client.post(f"/api/dataset/{did}/clean",
                          json={"drop_duplicates": False, "missing_strategy": "none",
                                "outlier_strategy": "none"}).json()
    after = app_module.DATASETS[did]["df"]

    assert before.shape == after.shape, f"{before.shape} vs {after.shape}"
    assert list(before.columns) == list(after.columns)

    numeric = [c for c in before.columns if pd.api.types.is_numeric_dtype(before[c])]
    assert numeric, "基准数据应含数值列"

    for col in numeric:
        a = pd.to_numeric(before[col], errors="coerce")
        b = pd.to_numeric(after[col], errors="coerce")
        assert a.isna().sum() == b.isna().sum(), f"{col} 缺失数变化"
        assert np.allclose(a.dropna().values, b.dropna().values), f"{col} 取值变化"

    # 干净数据不应触发高缺失警示
    assert not any(v["high_missing"] for v in confirmed["variables"])
    # trim 不得把干净数据改坏，也不得残留首尾空白
    for col in before.columns:
        if pd.api.types.is_object_dtype(before[col]):
            bad = [v for v in before[col].dropna()
                   if isinstance(v, str) and v != v.strip()]
            assert not bad, f"{col} 残留空白：{bad[:3]}"
