# medical-data-analysis/backend/app.py
"""FastAPI 后端入口：数据集上传解析、医疗校验、变量确认、清洗与统计分析接口。

CORS 仅允许 config.CORS_ORIGINS 中的固定来源；统一错误响应 {code, message, detail}。
本轮覆盖 P0 全部分析接口与 P1 的多因素分析接口。
"""

import hmac
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import analyzer
import llm_service
import ml_engine
import multivariable
import storage
import validators
from automl.errors import AutoMLError
from config import (ALLOWED_EXTENSIONS, ALPHA, API_TOKEN, CORS_ORIGINS, FDR_METHOD,
                    FRONTEND_ORIGIN, MAX_DATASETS_IN_MEMORY, MAX_UPLOAD_MB, MODE, PORT,
                    RANDOM_SEED, REFERENCE_RANGES)

log = logging.getLogger(__name__)

# uvicorn 只配置自己的 logger，不会接管应用日志；这里给应用 logger 补一个 handler，
# 否则淘汰等 INFO 级记录（默认 root 级别为 WARNING）会被静默丢弃。
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

app = FastAPI(title="Medical Data Analysis Platform", version="0.1.0",
              description="Backend of the Medical Data Analysis Platform: a frontend/backend separated "
                          "service whose local and cloud modes share the same business code.")

if MODE == "cloud" and not API_TOKEN:
    log.warning("MODE=cloud 但未设置 API_TOKEN：所有接口对公网裸露，请立即在 .env 中配置 API_TOKEN")


# ------------------------------------------------------------------
# 鉴权与限流（纯 ASGI 中间件，早于路由与 CORS 业务处理执行）
# ------------------------------------------------------------------
class _AuthMiddleware:
    """API_TOKEN 非空时，/api/*（/api/health 除外）必须携带匹配令牌。

    比对用 hmac.compare_digest 防时序侧信道；令牌为空（默认本地单用户）时完全放行。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not API_TOKEN or scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path.startswith("/api") and path != "/api/health":
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            token = headers.get(b"x-api-token", b"").decode("latin-1")
            auth = headers.get(b"authorization", b"").decode("latin-1")
            if not token and auth.lower().startswith("bearer "):
                token = auth[7:].strip()
            if not hmac.compare_digest(token, API_TOKEN):
                resp = JSONResponse(status_code=401, content={
                    "code": 401, "message": "未授权：请在设置页填写访问令牌",
                    "detail": "后端已启用 API_TOKEN 鉴权"})
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


# 重计算 / 敏感接口的进程内限流：{(客户端IP, 路径): [时间戳...]}，滑动窗口。
# pytest 环境下自动关闭（测试会短时间内大量调用上传接口，见 test_dataset_eviction）。
_RATE_LIMITS = {                       # 路径: (次数上限, 窗口秒)
    "/api/dataset/upload": (20, 60),
    "/api/sample/load": (20, 60),
    "/api/ml/train": (6, 60),
    "/api/ml/train/async": (6, 60),
    "/api/automl/analyze": (6, 60),
    "/api/automl/train": (6, 60),
    "/api/llm/config": (10, 60),
    "/api/llm/test": (10, 60),
}
_rate_bucket: dict = {}
_rate_lock = threading.Lock()
_RATE_MAX_KEYS = 10000                 # 表上限，防限流表自身被内存打爆


class _RateLimitMiddleware:
    """对 _RATE_LIMITS 中的 POST 接口按客户端 IP 做滑动窗口限流，超限返回 429。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or os.environ.get("PYTEST_CURRENT_TEST"):
            await self.app(scope, receive, send)
            return
        limit = _RATE_LIMITS.get(scope.get("path", ""))
        if limit and scope.get("method") == "POST":
            client = dict(scope.get("headers", [])).get(b"x-forwarded-for")
            client = client.decode("latin-1").split(",")[0].strip() if client else ""
            if not client:
                client = scope.get("client", ("unknown", 0))[0]
            key = (client, scope["path"])
            now = time.monotonic()
            max_n, window = limit
            with _rate_lock:
                if len(_rate_bucket) > _RATE_MAX_KEYS:   # 极端情况下整体清空重来
                    _rate_bucket.clear()
                hits = [t for t in _rate_bucket.get(key, []) if now - t < window]
                allowed = len(hits) < max_n
                if allowed:
                    hits.append(now)
                _rate_bucket[key] = hits
            if not allowed:
                resp = JSONResponse(status_code=429, content={
                    "code": 429, "message": "请求过于频繁，请稍后再试",
                    "detail": f"该接口限制 {max_n} 次/{window} 秒"})
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


# add_middleware 后添加者更靠外层：CORS 必须最外，
# 否则 401/429 响应缺 CORS 头，浏览器会把鉴权失败误报成"无法连接后端"。
app.add_middleware(_RateLimitMiddleware)
app.add_middleware(_AuthMiddleware)
# CORS：仅允许指定前端来源，严禁 "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or [FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 内存中的“当前活动数据集”
# ------------------------------------------------------------------
# 用 OrderedDict 维持插入顺序：最旧的在前，淘汰时 popitem(last=False)。
# 容量上限见 config.MAX_DATASETS_IN_MEMORY，超出后从最旧开始逐个释放。
DATASETS: "OrderedDict[str, dict]" = OrderedDict()

# 当前激活的数据集：淘汰时永远豁免（即使它是最旧的，也跳过它去淘汰次旧的）。
# 默认跟随最新入库的数据集；_CURRENT_PINNED=True 时由外部显式指定，不再自动改写。
CURRENT_DATASET_ID: str | None = None
_CURRENT_PINNED: bool = False

# 记录最近被淘汰的 id，用于对已释放的数据集给出专门的错误语义（有界，避免自身无限增长）
_EVICTED_IDS: "deque[str]" = deque(maxlen=512)


def _set_current_dataset(dataset_id: str, pinned: bool = False) -> None:
    """
    指定当前激活数据集。
    pinned=True 表示显式锁定：后续上传不再自动改写，豁免可长期生效（即使它最旧）。
    """
    global CURRENT_DATASET_ID, _CURRENT_PINNED
    CURRENT_DATASET_ID = dataset_id
    _CURRENT_PINNED = pinned


def _auto_set_current(dataset_id: str) -> None:
    """
    新数据集入库时决定激活项：默认跟随最新（与"上传后成为当前数据集"的行为同源）；
    若已被显式锁定且锁定项仍在内存中，则保持原激活项不动。
    """
    global CURRENT_DATASET_ID, _CURRENT_PINNED
    if _CURRENT_PINNED and CURRENT_DATASET_ID in DATASETS:
        return
    CURRENT_DATASET_ID = dataset_id
    _CURRENT_PINNED = False


def _estimate_bytes(ds: dict) -> int:
    """数据集内存占用估算（字节）。取值失败时返回 0，不阻塞淘汰主流程。"""
    try:
        return int(ds["df"].memory_usage(deep=True).sum())
    except Exception:
        try:
            return int(ds["df"].memory_usage(deep=False).sum())
        except Exception:
            return 0


def _release_dataset(dataset_id: str):
    """释放单个数据集并连带清理其绑定模型，返回 (行数, 估算占用字节)。"""
    ds = DATASETS.pop(dataset_id, None)
    if ds is None:
        return 0, 0
    try:
        rows = int(ds["df"].shape[0])
    except Exception:
        rows = 0
    ml_engine.invalidate_model(dataset_id)
    return rows, _estimate_bytes(ds)


def _evict_if_needed(trigger: str) -> list:
    """
    容量淘汰：数量超过 MAX_DATASETS_IN_MEMORY 时，从最旧的开始逐个释放，直到不超限。
    当前激活数据集（CURRENT_DATASET_ID）豁免——即使它是最旧的，也跳过它去淘汰次旧的。
    注意：这是按插入顺序的容量淘汰，不是 LRU，不追踪访问顺序、不记录访问时间戳。
    """
    evicted = []
    while len(DATASETS) > MAX_DATASETS_IN_MEMORY:
        victim = None
        for did in list(DATASETS.keys()):     # OrderedDict 保持插入顺序：最旧在前
            if did == CURRENT_DATASET_ID:
                continue
            victim = did
            break
        if victim is None:
            break                              # 只剩当前激活数据集，无法再淘汰
        rows, size = _release_dataset(victim)
        _EVICTED_IDS.append(victim)
        evicted.append(victim)
        log.info("数据集已释放（超出内存保留上限 %s）：id=%s，行数=%s，占用约 %.2f MB；"
                 "触发来源：%s；释放后内存剩余 %s 个",
                 MAX_DATASETS_IN_MEMORY, victim, rows, size / (1024 * 1024),
                 trigger, len(DATASETS))
    return evicted


class ApiError(Exception):
    def __init__(self, message: str, code: int = 400, detail=None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.detail = detail


@app.exception_handler(ApiError)
async def api_error_handler(request, exc: ApiError):
    return JSONResponse(status_code=exc.code,
                        content={"code": exc.code, "message": exc.message, "detail": exc.detail})


@app.exception_handler(AutoMLError)
async def automl_error_handler(request, exc: AutoMLError):
    """自动建模的数据形态/生存结构问题属于“可修正的输入错误”，返回 400 而不是 500。

    这些异常本身已带面向用户的中文原因与修正建议（如“事件列存在缺失，请补齐或剔除该行”），
    旧实现未注册处理器，会落进兜底的 Exception handler 变成“服务器内部错误”，用户无从修正。
    """
    log.warning("自动建模输入校验失败：%s（建议：%s）", exc.message, exc.suggestion)
    return JSONResponse(status_code=400,
                        content={"code": 400, "message": exc.message,
                                 "suggestion": exc.suggestion, "detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_handler(request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500,
                        content={"code": 500, "message": "服务器内部错误", "detail": "请联系管理员"})


# ------------------------------------------------------------------
# 请求体模型
# ------------------------------------------------------------------
class ConfirmRequest(BaseModel):
    types: dict = Field(default_factory=dict, description="{字段名: 类型}")
    force_continue: bool = False


class CleaningRequest(BaseModel):
    drop_duplicates: bool = True
    drop_columns: list = Field(default_factory=list)
    drop_rows_missing_in: list = Field(default_factory=list)
    outlier_action: str = "keep"  # keep | drop_rows


class DescribeRequest(BaseModel):
    dataset_id: str
    fields: list | None = None


class NormalityRequest(BaseModel):
    dataset_id: str
    fields: list


class UnivariateRequest(BaseModel):
    dataset_id: str
    target: str
    group: str
    target_type: str = "numeric"


class CorrelationRequest(BaseModel):
    dataset_id: str
    fields: list


class MultivariableRequest(BaseModel):
    dataset_id: str
    outcome: str
    predictors: list
    covariates: list = Field(default_factory=list)
    task: str = "auto"


class ScreenRequest(BaseModel):
    dataset_id: str
    outcome: str
    candidates: list


class PlotRequest(BaseModel):
    dataset_id: str
    field: str
    group: str | None = None


class TrainRequest(BaseModel):
    dataset_id: str
    features: list
    target: str
    task: str = "classification"
    with_shap: bool = True


class PredictRequest(BaseModel):
    dataset_id: str
    sample: dict


def _ml_payload(payload: dict) -> dict:
    """移除不可序列化的训练产物后返回给前端。"""
    return {k: v for k, v in payload.items() if k != "_fitted"}


# ------------------------------------------------------------------
# 工具
# ------------------------------------------------------------------
def _get_dataset(dataset_id: str) -> dict:
    ds = DATASETS.get(dataset_id)
    if not ds:
        if dataset_id in _EVICTED_IDS:
            # 已淘汰：给出专门语义，detail 与 message 同文，便于前端与调用方各自取用
            msg = "数据集已释放（超出内存保留上限），请重新上传"
            raise ApiError(msg, 404, msg)
        raise ApiError("数据集不存在或已失效，请重新上传", 404)
    return ds


def _active_frame(ds: dict):
    return ds["df"]


def _types_map(ds: dict) -> dict:
    return {v["name"]: v["confirmed_type"] for v in ds["schema"] if not v.get("excluded")}


def _analysis_fields(ds: dict, only_numeric=False) -> list:
    out = []
    for v in ds["schema"]:
        if v.get("excluded"):
            continue
        if only_numeric and v["confirmed_type"] != "numeric":
            continue
        out.append(v["name"])
    return out


# ------------------------------------------------------------------
# 基础接口
# ------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "mode": MODE, "version": app.version,
            "datasets_in_memory": len(DATASETS)}


@app.get("/api/config")
def get_config():
    return {"mode": MODE, "max_upload_mb": MAX_UPLOAD_MB,
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS), "frontend_origin": FRONTEND_ORIGIN,
            "storage": storage.describe(),
            # 分析默认值由后端全局配置决定，前端仅如实呈现，不做本地覆盖
            "analysis_defaults": {
                "alpha": ALPHA,
                "fdr_method": FDR_METHOD,
                "random_seed": RANDOM_SEED,
            },
            # 仓库地址来自环境变量，未设置时前端按"未配置"如实显示
            "repository": os.getenv("REPO_URL", "")}


# ------------------------------------------------------------------
# 分析历史（本地 SQLite / 云上 RDS，接口与业务代码一致）
# ------------------------------------------------------------------
class HistoryRequest(BaseModel):
    dataset_name: str | None = None
    n_rows: int | None = None
    outcome: str | None = None
    payload: dict = Field(default_factory=dict)


@app.post("/api/history")
def save_history(req: HistoryRequest):
    return storage.save_history(req.model_dump())


@app.get("/api/history")
def list_history(limit: int = 50):
    # limit 设上下限：防止超大值导致全表扫描拖垮服务
    limit = min(max(int(limit), 1), 500)
    return {"mode": storage.mode(), "items": storage.list_history(limit)}


# ------------------------------------------------------------------
# 内置示例数据
# ------------------------------------------------------------------
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
SAMPLES = {
    "medical": {"file": "medical_sample.csv", "name": "模拟医疗数据集",
                "description": "302 例模拟代谢综合征随访数据（含缺失值、重复行与超参考区间异常值，GBK 编码）",
                "encoding": "gbk"},
    "wdbc": {"file": "wdbc.csv", "name": "WDBC 乳腺癌数据集",
             "description": "569 例乳腺肿物病理测量数据（良性 357 / 恶性 212），统计与建模全流程基准测试数据",
             "encoding": "utf-8"},
}


@app.get("/api/sample/list")
def sample_list():
    items = []
    for key, cfg in SAMPLES.items():
        path = os.path.join(SAMPLE_DIR, cfg["file"])
        rows = cols = None
        if os.path.exists(path):
            try:
                import pandas as pd
                head = pd.read_csv(path, nrows=5, encoding=cfg["encoding"])
                cols = int(head.shape[1])
                with open(path, "r", encoding=cfg["encoding"], errors="replace") as f:
                    rows = sum(1 for _ in f) - 1
            except Exception:
                pass
        items.append({"key": key, "name": cfg["name"], "description": cfg["description"],
                      "available": os.path.exists(path), "rows": rows, "cols": cols})
    return {"samples": items}


@app.post("/api/sample/load")
def sample_load(req: dict):
    key = (req or {}).get("key")
    cfg = SAMPLES.get(key)
    if not cfg:
        raise ApiError("示例数据不存在：" + str(key), 404)
    path = os.path.join(SAMPLE_DIR, cfg["file"])
    if not os.path.exists(path):
        raise ApiError("示例数据文件缺失，请先运行 sample_data/generate_sample_data.py", 404)
    with open(path, "rb") as f:
        content = f.read()
    return _ingest(content, cfg["file"])


def _ingest(content: bytes, filename: str) -> dict:
    """解析并登记数据集（上传与示例数据共用同一套逻辑）。"""
    try:
        df, meta = analyzer.parse_table(content, filename)
    except analyzer.ParseError as exc:
        raise ApiError("文件解析失败", 400, str(exc)) from exc
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ApiError("文件解析后没有有效数据", 400)

    # 列名标准化（模块零）：识别非标准列名对应的标准名，存入映射表。
    # 不重命名 df 列（保持用户原始列名），而是把映射传给 validators 做参考区间匹配。
    try:
        from automl.standardize import standardize_dataframe
        std_result = standardize_dataframe(df)
        rename_map = {m.raw_name: m.standard_name for m in std_result.columns
                      if m.standard_name and m.standard_name != m.raw_name}
        if rename_map:
            log.info(f"列名标准化：识别 {len(rename_map)} 列映射（{list(rename_map.items())[:3]}）")
    except Exception as exc:
        log.warning(f"列名标准化失败（非致命，继续原名列）：{exc}")
        std_result = None
        rename_map = {}

    dataset_id = uuid.uuid4().hex[:12]
    # 内存中只维护“当前活动数据集”，上传新数据集后旧模型立即失效
    for old_id in list(ml_engine.MODEL_STORE.keys()):
        ml_engine.invalidate_model(old_id)

    validation = validators.validate_medical_dataset(df, rename_map=rename_map,
                                                    dataset_name=filename)
    schema = validators.infer_variable_types(df, rename_map=rename_map)
    ds = {"id": dataset_id, "name": filename, "df_raw": df, "df": df, "schema": schema,
          "validation": validation, "parse_meta": meta, "confirmed": False,
          "cleaning": None, "notes": [], "rev": 0,
          "standardization": std_result.to_dict() if std_result else None,
          "rename_map": rename_map}
    if not validation["is_medical"]:
        ds["notes"].append("医疗数据校验未通过，需用户在前端选择「终止分析」或「确认继续」。")
    DATASETS[dataset_id] = ds
    # 新数据集入库：激活项默认跟随最新（除非已被显式锁定）
    _auto_set_current(dataset_id)
    _evict_if_needed(f"载入数据集 {filename}（新 id={dataset_id}）")
    return {"dataset_id": dataset_id, "name": ds["name"],
            "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]),
            "parse_meta": meta, "validation": validation, "variables": schema,
            "standardization": ds["standardization"],
            "rename_map": rename_map}


@app.get("/api/reference-ranges")
def reference_ranges():
    return {"ranges": REFERENCE_RANGES,
            "note": "成人常规检验参考区间，依据《全国临床检验操作规程》（第 4 版）及相关指南；"
                    "实际临床判断应以本院检验科参考区间为准。"}


# ------------------------------------------------------------------
# 数据集：上传 / 校验 / 确认 / 清洗
# ------------------------------------------------------------------
@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    # 先检查声明的文件大小（来自 Content-Length 头）
    if file.size is not None and file.size / (1024 * 1024) > MAX_UPLOAD_MB:
        raise ApiError(f"文件大小 {file.size / (1024 * 1024):.1f} MB 超过限制 {MAX_UPLOAD_MB} MB", 413)
    
    # 分块读取文件，防止超过声明大小的恶意上传
    chunks = []
    total_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
        total_size += len(chunk)
        if total_size / (1024 * 1024) > MAX_UPLOAD_MB:
            raise ApiError(f"文件大小超过限制 {MAX_UPLOAD_MB} MB", 413)
    
    content = b"".join(chunks)

    try:
        df, meta = analyzer.parse_table(content, file.filename or "")
    except analyzer.ParseError as exc:
        raise ApiError("文件解析失败", 400, str(exc)) from exc

    return _ingest(content, file.filename or "uploaded")


@app.get("/api/dataset/{dataset_id}")
def get_dataset(dataset_id: str):
    ds = _get_dataset(dataset_id)
    return {"dataset_id": ds["id"], "name": ds["name"],
            "n_rows": int(ds["df"].shape[0]), "n_cols": int(ds["df"].shape[1]),
            "validation": ds["validation"], "variables": ds["schema"],
            "confirmed": ds["confirmed"], "cleaning": ds["cleaning"],
            "notes": ds["notes"], "parse_meta": ds["parse_meta"]}


@app.post("/api/dataset/{dataset_id}/confirm")
def confirm_schema(dataset_id: str, req: ConfirmRequest):
    ds = _get_dataset(dataset_id)
    if not ds["validation"]["is_medical"] and not req.force_continue:
        raise ApiError("医疗数据校验未通过，请选择「终止分析」或「确认继续」（force_continue=true）", 422,
                       ds["validation"]["message"])
    ds["schema"] = validators.infer_variable_types(ds["df_raw"], overrides=req.types,
                                                    rename_map=ds.get("rename_map"))
    ds["confirmed"] = True
    # 类型确认会改变建模口径：版本号 +1，旧的训练结果缓存自然失效
    ds["rev"] = ds.get("rev", 0) + 1
    ml_engine.clear_cache(dataset_id)
    if req.force_continue and not ds["validation"]["is_medical"]:
        if "用户已手动放行非医疗数据（force_continue），报告中需注明该数据集未通过医疗校验。" not in ds["notes"]:
            ds["notes"].append("用户已手动放行非医疗数据（force_continue），报告中需注明该数据集未通过医疗校验。")
    return {"dataset_id": ds["id"], "variables": ds["schema"], "confirmed": True, "notes": ds["notes"]}


@app.get("/api/dataset/{dataset_id}/cleaning-report")
def cleaning_report(dataset_id: str):
    ds = _get_dataset(dataset_id)
    return analyzer.build_cleaning_report(ds["df"], ds["schema"])


@app.post("/api/dataset/{dataset_id}/clean")
def clean_dataset(dataset_id: str, req: CleaningRequest):
    ds = _get_dataset(dataset_id)
    df2, log = analyzer.apply_cleaning(ds["df"], req.model_dump(), rename_map=ds.get("rename_map"))
    ds["df"] = df2
    ds["schema"] = validators.infer_variable_types(
        df2, overrides={v["name"]: v["confirmed_type"] for v in ds["schema"]},
        rename_map=ds.get("rename_map"))
    ds["cleaning"] = log
    # 数据行/列发生变化：版本号 +1，旧的训练结果缓存自然失效
    ds["rev"] = ds.get("rev", 0) + 1
    ml_engine.clear_cache(dataset_id)
    return {"dataset_id": ds["id"], "cleaning": log,
            "n_rows": int(df2.shape[0]), "variables": ds["schema"]}


@app.get("/api/dataset/{dataset_id}/outcome-candidates")
def outcome_candidates(dataset_id: str):
    ds = _get_dataset(dataset_id)
    return {"candidates": validators.suggest_outcome_candidates(ds["df"], ds["schema"])}


# ------------------------------------------------------------------
# 统计分析接口
# ------------------------------------------------------------------
@app.post("/api/analysis/describe")
def describe(req: DescribeRequest):
    ds = _get_dataset(req.dataset_id)
    df = _active_frame(ds)
    fields = req.fields or _analysis_fields(ds)
    numeric, categorical = [], []
    for f in fields:
        if f not in df.columns:
            continue
        vtype = next((v["confirmed_type"] for v in ds["schema"] if v["name"] == f), "numeric")
        if vtype == "numeric":
            numeric.append(analyzer.describe_numeric(df, f, rename_map=ds.get("rename_map")))
        elif vtype == "categorical":
            categorical.append(analyzer.describe_categorical(df, f))
    return {"n_rows": int(df.shape[0]), "numeric": numeric, "categorical": categorical}


@app.post("/api/analysis/normality")
def normality(req: NormalityRequest):
    ds = _get_dataset(req.dataset_id)
    df = _active_frame(ds)
    out = []
    for f in req.fields:
        if f not in df.columns:
            continue
        r = analyzer.normality_test(pd_numeric(df[f]))
        r["field"] = f
        out.append(r)
    return {"fields": out}


@app.post("/api/analysis/univariate")
def univariate(req: UnivariateRequest):
    ds = _get_dataset(req.dataset_id)
    df = _active_frame(ds)
    _check_fields(df, [req.target, req.group])
    # 分组变量类型护栏：数值型变量不能作为分组做组间比较（569 个"单样本组"会
    # 让 η² 必然趋近 1、p 值退化为随机数），直接 422 说明原因，不静默计算
    group_type = next((v["confirmed_type"] for v in ds["schema"] if v["name"] == req.group), None)
    if group_type == "numeric":
        raise ApiError(
            f"分组变量「{req.group}」是数值型变量，不能直接用于组间比较。"
            f"请选择分类变量（如诊断、分组）作为分组，或先在数据确认页把该字段改为分类类型。", 422)
    return analyzer.compare_groups(df, req.target, req.group, req.target_type)


@app.post("/api/analysis/correlation")
def correlation(req: CorrelationRequest):
    ds = _get_dataset(req.dataset_id)
    df = _active_frame(ds)
    fields = [f for f in req.fields if f in df.columns]
    if len(fields) < 2:
        raise ApiError("相关性分析至少需要 2 个数值字段", 422)
    return analyzer.correlation_matrix(df, fields)


@app.post("/api/analysis/multivariable")
def multivariable_api(req: MultivariableRequest):
    ds = _get_dataset(req.dataset_id)
    df = _active_frame(ds)
    try:
        return multivariable.run_multivariable(df, req.outcome, req.predictors, req.covariates,
                                               _types_map(ds), req.task)
    except multivariable.ModelingError as exc:
        raise ApiError(str(exc), 422) from exc


@app.post("/api/analysis/screen")
def screen(req: ScreenRequest):
    ds = _get_dataset(req.dataset_id)
    df = _active_frame(ds)
    candidates = req.candidates or _analysis_fields(ds)
    try:
        return multivariable.screen_univariate(df, req.outcome, candidates, _types_map(ds))
    except multivariable.ModelingError as exc:
        # 结局变量护栏（数值型/水平数超限等）：422 明确报错，绝不静默输出垃圾表
        raise ApiError(str(exc), 422) from exc


@app.post("/api/analysis/histogram")
def histogram(req: PlotRequest):
    ds = _get_dataset(req.dataset_id)
    return analyzer.histogram_data(_active_frame(ds), req.field, rename_map=ds.get("rename_map"))


@app.post("/api/analysis/boxplot")
def boxplot(req: PlotRequest):
    ds = _get_dataset(req.dataset_id)
    return analyzer.boxplot_data(_active_frame(ds), req.field, req.group)


@app.post("/api/analysis/qqplot")
def qqplot(req: PlotRequest):
    ds = _get_dataset(req.dataset_id)
    return analyzer.qqplot_data(_active_frame(ds), req.field)


# ------------------------------------------------------------------
# 机器学习接口
# ------------------------------------------------------------------
def _save_ml_model(dataset_id: str, payload: dict):
    """按预测能力决定是否落盘：R²<0 的回归模型只保留指标用于诊断。

    这类模型的预测值系统性偏离真实值（还不如直接用均值），保存后容易被当作
    可用模型反复调用，因此不写入模型库，并让该数据集此前的旧模型一并失效。
    """
    if payload.get("predictive") is False:
        payload.setdefault("warnings", []).append(
            "模型不具备预测能力（测试集 R²<0），本次训练结果未保存；请更换特征列或增加样本后重训。")
        ml_engine.invalidate_model(dataset_id)
        ml_engine.mark_model_discarded(dataset_id, payload)
        return None
    saved = ml_engine.save_model(dataset_id, payload)
    return {k: v for k, v in saved.items() if k != "path"}


@app.post("/api/ml/train")
def ml_train(req: TrainRequest):
    ds = _get_dataset(req.dataset_id)
    df = _active_frame(ds)
    _check_fields(df, [req.target] + [f for f in req.features if f])
    try:
        payload = ml_engine.train_and_evaluate(df, req.features, req.target, req.task,
                                               _types_map(ds), req.with_shap,
                                               cache_key=f"{req.dataset_id}:{ds.get('rev', 0)}")
    except ml_engine.MLError as exc:
        raise ApiError(str(exc), 422) from exc
    if payload.get("skipped"):
        return _ml_payload(payload)
    return {**_ml_payload(payload), "saved": _save_ml_model(req.dataset_id, payload)}


# ------------------------------------------------------------------
# 异步训练任务：大样本训练耗时长，改为后台执行 + 轮询，避免 HTTP 长连接超时
# ------------------------------------------------------------------
TSK_STORE: "OrderedDict[str, dict]" = OrderedDict()
_TSK_CAP = 50                      # 最多保留的任务数（超出淘汰最旧）
_TSK_TTL = 3600                    # 已完成任务保留时长（秒）
_tsk_lock = threading.Lock()
_tsk_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ml-task")


def _tsk_update(task_id: str, **kw):
    with _tsk_lock:
        task = TSK_STORE.get(task_id)
        if task:
            task.update(kw)


def _tsk_prune():
    """淘汰过期/超额任务，避免任务表自身无限增长。"""
    now = time.time()
    with _tsk_lock:
        stale = [tid for tid, t in TSK_STORE.items()
                 if t.get("status") in ("done", "error") and now - t.get("finished", now) > _TSK_TTL]
        for tid in stale:
            TSK_STORE.pop(tid, None)
        while len(TSK_STORE) > _TSK_CAP:
            TSK_STORE.popitem(last=False)


@app.post("/api/ml/train/async")
def ml_train_async(req: TrainRequest):
    """提交后台训练任务，立即返回 task_id；前端轮询 /api/tasks/{id} 取结果。"""
    ds = _get_dataset(req.dataset_id)
    df = _active_frame(ds)
    _check_fields(df, [req.target] + [f for f in req.features if f])
    # 提交时快照特征口径与数据版本：任务执行期间用户再清洗/改类型，旧任务结果不会污染新版本
    types = _types_map(ds)
    rev = ds.get("rev", 0)
    task_id = uuid.uuid4().hex
    with _tsk_lock:
        TSK_STORE[task_id] = {"id": task_id, "status": "running", "progress": 0,
                              "stage": "排队中", "created": time.time(),
                              "finished": None, "result": None, "error": None, "code": None}
    _tsk_prune()

    def _run():
        try:
            _tsk_update(task_id, progress=15, stage="训练中")
            payload = ml_engine.train_and_evaluate(df, req.features, req.target, req.task,
                                                   types, req.with_shap,
                                                   cache_key=f"{req.dataset_id}:{rev}")
            _tsk_update(task_id, progress=85, stage="保存模型")
            if payload.get("skipped"):
                result = _ml_payload(payload)
            else:
                result = {**_ml_payload(payload),
                          "saved": _save_ml_model(req.dataset_id, payload)}
            _tsk_update(task_id, status="done", progress=100, stage="完成",
                        result=result, finished=time.time())
        except ml_engine.MLError as exc:
            _tsk_update(task_id, status="error", progress=100, stage="失败",
                        error=str(exc), code=422, finished=time.time())
        except Exception as exc:      # 训练线程内异常必须落库，否则前端只会一直转圈
            log.exception("异步训练失败：%s", exc)
            _tsk_update(task_id, status="error", progress=100, stage="失败",
                        error=f"训练失败：{exc}", code=500, finished=time.time())

    _tsk_pool.submit(_run)
    return {"task_id": task_id, "status": "running"}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    """轮询任务状态：running / done / error；done 时附带 result。"""
    _tsk_prune()
    task = TSK_STORE.get(task_id)
    if not task:
        raise ApiError("任务不存在或已过期，请重新发起训练", 404)
    out = {"task_id": task_id, "status": task["status"],
           "progress": task.get("progress"), "stage": task.get("stage"),
           "error": task.get("error"), "code": task.get("code")}
    if task["status"] == "done":
        out["result"] = task["result"]
    return out


@app.get("/api/ml/model/{dataset_id}")
def ml_model(dataset_id: str):
    """当前数据集的模型信息；未建模返回 null，前端据此显示空状态。"""
    _get_dataset(dataset_id)
    bundle = ml_engine.load_model(dataset_id)
    if bundle is None:
        discarded = ml_engine.discarded_model_info(dataset_id)
        if discarded:
            r2 = discarded.get("r2")
            r2_text = f"R²={float(r2):.3f} " if r2 is not None else ""
            return {"trained": False, "discarded": discarded,
                    "message": (f"上一次训练（回归）测试集 {r2_text}< 0，模型不具备预测能力，"
                                "已丢弃；请更换特征列或增加样本后重训")}
        return {"trained": False,
                "message": "请先完成当前数据集的建模"}
    return {"trained": True,
            "task": bundle["task"], "target": bundle["target"],
            "features": bundle["features"],
            "numeric_features": bundle["numeric_features"],
            "categorical_features": bundle["categorical_features"],
            "numeric_ranges": bundle["numeric_ranges"],
            "categories": bundle["categories"],
            "classes": bundle["classes"],
            "best_model": bundle["best_model"],
            "test_metrics": bundle["test_metrics"],
            "predictive": bundle.get("predictive", True),
            "n_train": bundle["n_train"], "n_test": bundle["n_test"],
            "created_at": bundle["created_at"]}


@app.post("/api/ml/predict")
def ml_predict(req: PredictRequest):
    _get_dataset(req.dataset_id)
    try:
        return ml_engine.predict_new_sample(req.dataset_id, req.sample)
    except ml_engine.MLError as exc:
        raise ApiError(str(exc), 422) from exc


@app.delete("/api/ml/model/{dataset_id}")
def ml_invalidate(dataset_id: str):
    try:
        ml_engine.invalidate_model(dataset_id)
    except ml_engine.MLError as exc:
        # 非法 id（如含路径穿越字符）：明确拒绝，绝不触碰文件系统
        raise ApiError(str(exc), 400) from exc
    return {"dataset_id": dataset_id, "invalidated": True}


# ------------------------------------------------------------------
# 大模型接入（未配置时自动降级，不影响其余功能）
# ------------------------------------------------------------------
@app.get("/api/llm/status")
def llm_status():
    return llm_service.get_config()


@app.post("/api/llm/config")
def llm_config(req: dict):
    req = req or {}
    try:
        return llm_service.set_config(provider=req.get("provider"), base_url=req.get("base_url"),
                                      model=req.get("model"), api_key=req.get("api_key"))
    except ValueError as exc:
        # base_url 未通过 SSRF/https 校验
        raise ApiError(str(exc), 422) from exc


@app.post("/api/llm/test")
def llm_test(req: dict):
    req = req or {}
    if not (req.get("api_key") or llm_service.get_config()["configured"]):
        return {"connected": False, "status": "not_configured",
                "message": "未配置 API Key，请在设置页填写后测试（AI 功能将自动降级为模板解读）"}
    return llm_service.test_connection(req)


# ------------------------------------------------------------------
# 内部辅助
# ------------------------------------------------------------------
def pd_numeric(series):
    import pandas as pd
    return pd.to_numeric(series, errors="coerce")


def _check_fields(df, fields):
    import pandas as pd  # noqa: F401
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ApiError("字段不存在：" + "、".join(missing), 422)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=False)


# ------------------------------------------------------------------
# 智能建模分析引擎（backend/automl，模块零 + 五步流水线 + 手动模式）
# 与现有 /api/ml/* 并行，不影响既有前端看板与报告。
# ------------------------------------------------------------------
from automl.api import create_router as _create_automl_router

app.include_router(_create_automl_router(_get_dataset))
