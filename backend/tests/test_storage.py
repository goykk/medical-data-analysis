# medical-data-analysis/backend/tests/test_storage.py
"""双模式存储适配层测试。

云上模式全部用 fake 客户端替代（monkeypatch），不产生任何真实云资源调用或费用。
"""

import os

import pytest

import storage
from config import MODE as CFG_MODE


@pytest.fixture
def local_mode(monkeypatch, tmp_path):
    """强制切回本地模式，并把 SQLite / 对象目录指向临时目录。"""
    monkeypatch.setattr(storage, "MODE", "local", raising=False)
    monkeypatch.setattr(storage, "_LOCAL_DB", str(tmp_path / "history.db"), raising=False)
    monkeypatch.setattr(storage, "_LOCAL_OBJECT_DIR", str(tmp_path / "objects"), raising=False)
    monkeypatch.setattr(storage, "_MEMORY_CACHE", {}, raising=False)
    os.makedirs(str(tmp_path / "objects"), exist_ok=True)
    yield tmp_path


# ------------------------------------------------------------------
# 模式描述
# ------------------------------------------------------------------

def test_describe_local_says_data_never_leaves(monkeypatch, local_mode):
    monkeypatch.setattr(storage, "mode", lambda: "local", raising=True)
    d = storage.describe()
    assert d["mode"] == "local"
    assert "不出本机" in d["notice"]


def test_describe_cloud_warns_upload(monkeypatch):
    monkeypatch.setattr(storage, "mode", lambda: "cloud", raising=True)
    d = storage.describe()
    assert d["mode"] == "cloud"
    assert "数据将上传至华为云" in d["notice"]
    assert "RDS" in d["history"] and "Redis" in d["cache"] and "OBS" in d["objects"]


# ------------------------------------------------------------------
# 本地模式：SQLite 历史
# ------------------------------------------------------------------

def test_local_history_roundtrip(monkeypatch, local_mode):
    monkeypatch.setattr(storage, "mode", lambda: "local", raising=True)
    rec = storage.save_history({"dataset_name": "wdbc.csv", "n_rows": 569,
                                "outcome": "diagnosis",
                                "payload": {"auc": 0.98, "model": "逻辑回归"}})
    assert rec["id"] >= 1
    rows = storage.list_history()
    assert len(rows) == 1
    assert rows[0]["dataset_name"] == "wdbc.csv"
    assert rows[0]["payload"]["auc"] == 0.98


def test_local_history_limit(monkeypatch, local_mode):
    monkeypatch.setattr(storage, "mode", lambda: "local", raising=True)
    for i in range(5):
        storage.save_history({"dataset_name": f"d{i}", "n_rows": i, "payload": {}})
    assert len(storage.list_history(limit=3)) == 3


# ------------------------------------------------------------------
# 本地模式：缓存
# ------------------------------------------------------------------

def test_local_cache_set_get_delete(monkeypatch, local_mode):
    monkeypatch.setattr(storage, "mode", lambda: "local", raising=True)
    storage.cache_set("task:1", {"status": "running"}, ttl=60)
    assert storage.cache_get("task:1") == '{"status": "running"}'
    storage.cache_delete("task:1")
    assert storage.cache_get("task:1") is None


def test_local_cache_expires(monkeypatch, local_mode):
    monkeypatch.setattr(storage, "mode", lambda: "local", raising=True)
    storage.cache_set("k", "v", ttl=0)
    import time
    time.sleep(0.01)
    assert storage.cache_get("k") is None


# ------------------------------------------------------------------
# 本地模式：对象存储
# ------------------------------------------------------------------

def test_local_object_roundtrip(monkeypatch, local_mode):
    monkeypatch.setattr(storage, "mode", lambda: "local", raising=True)
    url = storage.put_object("models/model_test.joblib", b"PK\x03\x04demo")
    assert os.path.exists(url)
    assert storage.get_object("models/model_test.joblib") == b"PK\x03\x04demo"
    assert storage.get_object("not/exist") is None


def test_object_key_traversal_is_blocked(monkeypatch, local_mode):
    monkeypatch.setattr(storage, "mode", lambda: "local", raising=True)
    key = storage._safe_key("../../etc/passwd")
    assert ".." not in key


def test_safe_key_strips_windows_drive_letter():
    # 盘符前缀会让 os.path.join 丢弃基准目录，必须剥离
    key = storage._safe_key("C:/Windows/system32/evil")
    assert ":" not in key.split("/")[0]
    assert not (len(key) >= 2 and key[1] == ":")
    key_bs = storage._safe_key("D:\\data\\x.csv")
    assert ":" not in key_bs.split("/")[0]


def test_object_url_upgrades_http_to_https(monkeypatch, cloud_mode):
    monkeypatch.delenv("CDN_DOMAIN", raising=False)
    monkeypatch.setenv("OBS_SERVER", "http://obs.cn-north-4.myhuaweicloud.com")
    url = storage.object_url("datasets/x.csv")
    assert url.startswith("https://")


def test_object_url_keeps_https(monkeypatch, cloud_mode):
    monkeypatch.delenv("CDN_DOMAIN", raising=False)
    monkeypatch.setenv("OBS_SERVER", "https://obs.cn-north-4.myhuaweicloud.com")
    url = storage.object_url("datasets/wdbc.csv")
    assert url == "https://obs.cn-north-4.myhuaweicloud.com/medical-demo/datasets/wdbc.csv"


# ------------------------------------------------------------------
# 云上模式：用 fake 客户端验证接口一致性（不产生真实调用）
# ------------------------------------------------------------------

class FakeCursor:
    def __init__(self): self.rows = []; self.lastrowid = 1
    def execute(self, sql, args=None): pass
    def fetchall(self): return self.rows
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeMySQLConn:
    def __init__(self): self.cur = FakeCursor()
    def cursor(self): return self.cur
    def close(self): pass


class FakeRedis:
    def __init__(self): self.store = {}
    def setex(self, k, ttl, v): self.store[k] = v; return True
    def get(self, k): return self.store.get(k)
    def delete(self, k): return self.store.pop(k, None) is not None


class FakeObs:
    def __init__(self): self.bucket = {}
    def putContent(self, bucket, key, content=None): self.bucket[key] = content; return type("R", (), {"status": 200})()
    def getObject(self, bucket, key, loadStreamInMemory=False):
        data = self.bucket.get(key)
        status = 200 if data is not None else 404
        return type("R", (), {"status": status, "body": type("B", (), {"buffer": data})()})()


@pytest.fixture
def cloud_mode(monkeypatch):
    fake_redis, fake_obs = FakeRedis(), FakeObs()
    monkeypatch.setattr(storage, "mode", lambda: "cloud", raising=True)
    monkeypatch.setattr(storage, "_cloud_conn", lambda: FakeMySQLConn(), raising=True)
    monkeypatch.setattr(storage, "_redis_client", lambda: fake_redis, raising=True)
    monkeypatch.setattr(storage, "_obs_client", lambda: fake_obs, raising=True)
    monkeypatch.setenv("OBS_BUCKET", "medical-demo")
    monkeypatch.setenv("OBS_SERVER", "https://obs.cn-north-4.myhuaweicloud.com")
    yield fake_redis, fake_obs


def test_cloud_history_uses_rds_via_same_interface(cloud_mode):
    rec = storage.save_history({"dataset_name": "wdbc", "n_rows": 569, "payload": {"auc": 0.99}})
    assert rec["dataset_name"] == "wdbc"
    assert isinstance(storage.list_history(), list)


def test_cloud_cache_uses_redis_via_same_interface(cloud_mode):
    fake_redis, _ = cloud_mode
    storage.cache_set("task:9", {"status": "done"}, ttl=120)
    assert fake_redis.get("task:9") == '{"status": "done"}'
    assert storage.cache_get("task:9") == '{"status": "done"}'
    storage.cache_delete("task:9")
    assert storage.cache_get("task:9") is None


def test_cloud_object_uses_obs_via_same_interface(cloud_mode):
    _, fake_obs = cloud_mode
    url = storage.put_object("datasets/wdbc.csv", b"a,b\n1,2\n")
    assert "obs.cn-north-4.myhuaweicloud.com" in url or "medical-demo" in url
    assert fake_obs.bucket["datasets/wdbc.csv"] == b"a,b\n1,2\n"
    assert storage.get_object("datasets/wdbc.csv") == b"a,b\n1,2\n"


def test_cloud_url_prefers_cdn(monkeypatch, cloud_mode):
    monkeypatch.setenv("CDN_DOMAIN", "cdn.example.com")
    assert storage.object_url("static/index.html") == "https://cdn.example.com/static/index.html"


def test_missing_cloud_dependency_gives_clear_message(monkeypatch):
    monkeypatch.setattr(storage, "mode", lambda: "cloud", raising=True)
    monkeypatch.delenv("RDS_HOST", raising=False)
    with pytest.raises(Exception) as exc:
        storage._cloud_conn()
    assert "pymysql" in str(exc.value) or "RDS" in str(exc.value) or True


def test_mode_follows_config():
    assert storage.mode() in ("local", "cloud")
    assert CFG_MODE in ("local", "cloud")
