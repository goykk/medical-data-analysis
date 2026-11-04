# medical-data-analysis/backend/tests/test_dataset_eviction.py
"""数据集内存上限淘汰：压力测试与豁免规则。

覆盖：
1. 连续上传 30 个数据集后，内存中恰好保留最近 8 个
2. 保留的是最近的 8 个（新 id 可正常请求）
3. 已淘汰 id 返回 404 + "重新上传"语义
4. 淘汰时输出 INFO 日志（含 id / 行数 / 占用估算 / 触发来源）
5. 当前激活数据集豁免：即便最旧也不被淘汰
"""

import io
import logging

import pytest
from fastapi.testclient import TestClient

import app as app_module
import ml_engine
from config import MAX_DATASETS_IN_MEMORY

from conftest import make_medical_df


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def reset_memory():
    """每个用例前后清空全局内存状态，避免污染其它测试模块（同一进程内共享）。"""
    def _clear():
        app_module.DATASETS.clear()
        app_module.CURRENT_DATASET_ID = None
        app_module._CURRENT_PINNED = False
        app_module._EVICTED_IDS.clear()
        ml_engine.MODEL_STORE.clear()

    _clear()
    yield
    _clear()


def upload_one(client, df, name="evict_test.csv"):
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    r = client.post("/api/dataset/upload",
                    files={"file": (name, buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


@pytest.fixture(scope="module")
def small_df():
    return make_medical_df(n=60, seed=11, with_issues=False)


# ------------------------------------------------------------------
# 1. 压力测试：上限生效
# ------------------------------------------------------------------
def test_upload_30_keeps_only_max_in_memory(client, small_df):
    for _ in range(30):
        upload_one(client, small_df)

    health = client.get("/api/health").json()
    assert health["datasets_in_memory"] == MAX_DATASETS_IN_MEMORY, health
    assert len(app_module.DATASETS) == MAX_DATASETS_IN_MEMORY


# ------------------------------------------------------------------
# 2. 保留的是最近上传的 8 个
# ------------------------------------------------------------------
def test_retained_are_the_newest(client, small_df):
    ids = [upload_one(client, small_df) for _ in range(30)]

    assert list(app_module.DATASETS.keys()) == ids[-MAX_DATASETS_IN_MEMORY:]

    # 任意一个保留下来的 id 都能正常请求
    for did in ids[-MAX_DATASETS_IN_MEMORY:]:
        r = client.get(f"/api/dataset/{did}")
        assert r.status_code == 200, r.text

    # 当前激活数据集应为最后一个上传的
    assert app_module.CURRENT_DATASET_ID == ids[-1]


# ------------------------------------------------------------------
# 3. 已淘汰 id → 404 + 语义化信息（不得变成 500）
# ------------------------------------------------------------------
def test_evicted_id_returns_404_with_reupload_hint(client, small_df):
    ids = [upload_one(client, small_df) for _ in range(30)]
    evicted_id = ids[0]

    r = client.get(f"/api/dataset/{evicted_id}")
    assert r.status_code == 404, f"应为 404，实际 {r.status_code}"

    body = r.json()
    # 注意：detail 可能为 null，统一按空串处理
    detail = body.get("detail") or ""
    message = body.get("message") or ""
    assert "重新上传" in detail
    assert "重新上传" in message
    # 与"本来就不存在"的 id 区分开：两者都是 404，但已淘汰的有专门语义
    assert "已释放" in detail

    # 不存在的 id 仍是通用语义（detail 为 null，不应出现"已释放"）
    r2 = client.get("/api/dataset/never_existed_1")
    assert r2.status_code == 404
    assert "已释放" not in (r2.json().get("detail") or "")
    assert "重新上传" in (r2.json().get("message") or "")


# ------------------------------------------------------------------
# 4. 淘汰日志
# ------------------------------------------------------------------
def test_eviction_logs_release_records(client, small_df, caplog):
    ids = []
    with caplog.at_level(logging.INFO, logger="app"):
        # 先填满上限，再上传若干个触发淘汰
        for _ in range(MAX_DATASETS_IN_MEMORY):
            ids.append(upload_one(client, small_df))
        for _ in range(5):
            ids.append(upload_one(client, small_df))

    records = [rec for rec in caplog.records if "数据集已释放" in rec.getMessage()]
    assert records, "未捕获到淘汰日志"
    # 5 次超限上传 → 至少 5 条释放记录
    assert len(records) >= 5

    first = records[0]
    msg = first.getMessage()
    assert first.levelno == logging.INFO
    assert ids[0] in msg              # 被释放的 dataset_id
    assert "行数" in msg              # 行数
    assert "占用约" in msg            # 内存占用估算
    assert "触发来源" in msg          # 触发时机
    assert first.name == "app"


# ------------------------------------------------------------------
# 5. 当前激活数据集豁免
# ------------------------------------------------------------------
def test_active_dataset_is_exempt_from_eviction(client, small_df):
    ids = [upload_one(client, small_df) for _ in range(MAX_DATASETS_IN_MEMORY)]
    oldest = ids[0]

    # 显式锁定最旧的一个为"当前激活"（等价模拟"用户正在使用这个数据集"），
    # 锁定后上传不会自动改写激活项，因此豁免可持续生效
    app_module._set_current_dataset(oldest, pinned=True)

    for _ in range(6):                # 持续上传，触发多轮淘汰
        upload_one(client, small_df)

    assert oldest in app_module.DATASETS, "当前激活数据集不应被淘汰"
    assert len(app_module.DATASETS) == MAX_DATASETS_IN_MEMORY

    # 次旧的被淘汰（跳过激活项去淘汰次旧项）
    assert ids[1] not in app_module.DATASETS

    # 激活数据集仍然可用
    r = client.get(f"/api/dataset/{oldest}")
    assert r.status_code == 200, r.text


def test_no_error_level_log_during_eviction(client, small_df, caplog):
    """淘汰与请求已释放 id 的过程中，不得出现 ERROR 级日志堆栈。"""
    with caplog.at_level(logging.DEBUG, logger="app"):
        ids = [upload_one(client, small_df) for _ in range(30)]
        client.get(f"/api/dataset/{ids[0]}")   # 命中已淘汰 id

    errors = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert not errors, [rec.getMessage() for rec in errors]
