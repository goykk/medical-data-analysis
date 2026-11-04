# medical-data-analysis/backend/storage.py
"""双模式存储适配层。

本地模式（MODE=local）：分析历史存 SQLite 单文件；任务状态与缓存存内存；
                        数据集 / 模型产物存本地目录。
云上模式（MODE=cloud）：分析历史存华为云 RDS（MySQL）；任务状态与缓存存 Redis；
                        数据集 / 模型产物存 OBS。

业务代码只调用本模块暴露的接口（save_history / cache_set / put_object …），
不感知模式差异；模式由环境变量 MODE 切换，不维护两套业务逻辑。
"""

import json
import os
import sqlite3
import time

from config import MODE

_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_LOCAL_OBJECT_DIR = os.path.join(_LOCAL_DIR, "objects")
_LOCAL_DB = os.getenv("LOCAL_SQLITE_PATH") or os.path.join(_LOCAL_DIR, "history.db")

os.makedirs(_LOCAL_DIR, exist_ok=True)
os.makedirs(_LOCAL_OBJECT_DIR, exist_ok=True)

# 本地内存缓存（云上模式由 Redis 承担）
_MEMORY_CACHE = {}

HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    dataset_name TEXT,
    n_rows INTEGER,
    outcome TEXT,
    payload TEXT
)
"""
HISTORY_TABLE_SQL_MYSQL = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at VARCHAR(32),
    dataset_name VARCHAR(255),
    n_rows INT,
    outcome VARCHAR(255),
    payload LONGTEXT
)
"""


# ------------------------------------------------------------------
# 公共信息
# ------------------------------------------------------------------
def mode() -> str:
    return "cloud" if MODE == "cloud" else "local"


def describe() -> dict:
    """返回当前模式使用的存储后端，供前端/README 明示数据去向。"""
    if mode() == "cloud":
        return {
            "mode": "cloud",
            "history": "华为云 RDS（MySQL）",
            "cache": "华为云 Redis",
            "objects": "华为云 OBS",
            "notice": "云上模式：数据将上传至华为云，由使用者确认。",
        }
    return {
        "mode": "local",
        "history": f"本地 SQLite（{_LOCAL_DB}）",
        "cache": "进程内存",
        "objects": f"本地目录（{_LOCAL_OBJECT_DIR}）",
        "notice": "本地模式：数据不出本机。",
    }


# ------------------------------------------------------------------
# 分析历史
# ------------------------------------------------------------------
def _local_conn():
    conn = sqlite3.connect(_LOCAL_DB)
    conn.execute(HISTORY_TABLE_SQL)
    return conn


def _cloud_conn():
    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError("云上模式需要 pymysql：pip install pymysql") from exc
    conn = pymysql.connect(
        host=os.getenv("RDS_HOST"), port=int(os.getenv("RDS_PORT", "3306")),
        user=os.getenv("RDS_USER"), password=os.getenv("RDS_PASSWORD"),
        database=os.getenv("RDS_DATABASE"), charset="utf8mb4", autocommit=True)
    with conn.cursor() as cur:
        cur.execute(HISTORY_TABLE_SQL_MYSQL)
    return conn


def save_history(record: dict) -> dict:
    """record: {dataset_name, n_rows, outcome, payload}"""
    rec = dict(record or {})
    rec["created_at"] = rec.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S")
    payload = json.dumps(rec.get("payload") or {}, ensure_ascii=False)

    if mode() == "local":
        conn = _local_conn()
        try:
            cur = conn.execute(
                "INSERT INTO analysis_history (created_at, dataset_name, n_rows, outcome, payload)"
                " VALUES (?,?,?,?,?)",
                (rec["created_at"], rec.get("dataset_name"), rec.get("n_rows"),
                 rec.get("outcome"), payload))
            conn.commit()
            rec["id"] = cur.lastrowid
        finally:
            conn.close()
    else:
        conn = _cloud_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO analysis_history (created_at, dataset_name, n_rows, outcome, payload)"
                    " VALUES (%s,%s,%s,%s,%s)",
                    (rec["created_at"], rec.get("dataset_name"), rec.get("n_rows"),
                     rec.get("outcome"), payload))
                rec["id"] = cur.lastrowid
        finally:
            conn.close()
    return rec


def list_history(limit: int = 50) -> list:
    rows = []
    if mode() == "local":
        conn = _local_conn()
        try:
            for r in conn.execute(
                    "SELECT id, created_at, dataset_name, n_rows, outcome, payload"
                    " FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,)):
                rows.append(_row_to_record(r))
        finally:
            conn.close()
    else:
        conn = _cloud_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, created_at, dataset_name, n_rows, outcome, payload"
                    " FROM analysis_history ORDER BY id DESC LIMIT %s", (limit,))
                rows = [_row_to_record(r) for r in cur.fetchall()]
        finally:
            conn.close()
    return rows


def _row_to_record(r) -> dict:
    try:
        payload = json.loads(r[5]) if isinstance(r[5], (str, bytes)) else (r[5] or {})
    except Exception:
        payload = {}
    return {"id": r[0], "created_at": r[1], "dataset_name": r[2],
            "n_rows": r[3], "outcome": r[4], "payload": payload}


# ------------------------------------------------------------------
# 缓存 / 任务状态
# ------------------------------------------------------------------
def _redis_client():
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("云上模式需要 redis：pip install redis") from exc
    return redis.Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT", "6379")),
                       password=os.getenv("REDIS_PASSWORD") or None,
                       db=int(os.getenv("REDIS_DB", "0")), decode_responses=True)


def cache_set(key: str, value, ttl: int = 600) -> bool:
    data = json.dumps(value, ensure_ascii=False) if not isinstance(value, (str, bytes)) else value
    if mode() == "local":
        _MEMORY_CACHE[key] = {"value": data, "expire": time.time() + ttl}
        return True
    return bool(_redis_client().setex(key, ttl, data))


def cache_get(key: str):
    if mode() == "local":
        item = _MEMORY_CACHE.get(key)
        if not item:
            return None
        if item["expire"] < time.time():
            _MEMORY_CACHE.pop(key, None)
            return None
        return item["value"]
    raw = _redis_client().get(key)
    return raw


def cache_delete(key: str) -> bool:
    if mode() == "local":
        _MEMORY_CACHE.pop(key, None)
        return True
    return bool(_redis_client().delete(key))


# ------------------------------------------------------------------
# 对象存储（数据集 / 模型产物 / 前端静态资源）
# ------------------------------------------------------------------
def _obs_client():
    try:
        from obs import ObsClient
    except ImportError as exc:
        raise RuntimeError("云上模式需要 esdk-obs-python：pip install esdk-obs-python") from exc
    return ObsClient(access_key_id=os.getenv("OBS_AK"), secret_access_key=os.getenv("OBS_SK"),
                     server=os.getenv("OBS_SERVER"))


def _safe_key(key: str) -> str:
    k = str(key).replace("\\", "/").replace("..", "_").lstrip("/")
    # 去掉 Windows 盘符前缀（如 C:/...）：os.path.join 遇到盘符合把基准目录整体丢弃
    if len(k) >= 2 and k[1] == ":":
        k = k[2:].lstrip("/")
    return k


def put_object(key: str, data: bytes) -> str:
    """写入对象，返回可访问地址（本地为文件路径，云上为 OBS URL）。"""
    key = _safe_key(key)
    if mode() == "local":
        path = os.path.join(_LOCAL_OBJECT_DIR, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path
    bucket = os.getenv("OBS_BUCKET")
    _obs_client().putContent(bucket, key, content=data)
    return object_url(key)


def get_object(key: str):
    key = _safe_key(key)
    if mode() == "local":
        path = os.path.join(_LOCAL_OBJECT_DIR, key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()
    resp = _obs_client().getObject(os.getenv("OBS_BUCKET"), key, loadStreamInMemory=True)
    return resp.body.buffer if resp.status < 300 else None


def object_url(key: str) -> str:
    key = _safe_key(key)
    if mode() == "local":
        return os.path.join(_LOCAL_OBJECT_DIR, key)
    server = os.getenv("OBS_SERVER") or ""
    # 对象地址强制 https，防止明文传输医疗数据
    if server.startswith("http://"):
        server = "https://" + server[len("http://"):]
    cdn = os.getenv("CDN_DOMAIN")
    if cdn:
        return f"https://{cdn}/{key}"
    return f"{server}/{os.getenv('OBS_BUCKET')}/{key}"


__all__ = ["mode", "describe", "save_history", "list_history",
           "cache_set", "cache_get", "cache_delete",
           "put_object", "get_object", "object_url"]
