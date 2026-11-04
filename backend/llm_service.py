# medical-data-analysis/backend/llm_service.py
"""大模型接入层（OpenAI 兼容接口），支持 DeepSeek / 通义千问 / 智谱 GLM /
华为云 ModelArts Studio（盘古）/ 自定义。

约定：
- 密钥只存内存，严禁写入文件或日志；读取优先级：页面配置 > 环境变量 > 未配置；
- 未配置时 AI 功能自动降级，不影响页面其余功能；
- 所有调用带超时、重试与异常捕获。
"""

import ipaddress
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

import prompts

# ------------------------------------------------------------------
# 服务商预设
# ------------------------------------------------------------------
PROVIDERS = {
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
                 "model": "deepseek-chat", "auth_header": "Authorization", "auth_prefix": "Bearer "},
    "qwen": {"label": "通义千问", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "model": "qwen-plus", "auth_header": "Authorization", "auth_prefix": "Bearer "},
    "glm": {"label": "智谱 GLM", "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4-flash", "auth_header": "Authorization", "auth_prefix": "Bearer "},
    "modelarts_pangu": {
        "label": "华为云 ModelArts Studio（盘古）",
        "base_url": "https://cn-north-4.modelarts-infer.com/api/v2",
        "model": "pangu",
        "auth_header": "X-Apig-AppCode", "auth_prefix": "",
    },
    "custom": {"label": "自定义（OpenAI 兼容）", "base_url": "", "model": "",
               "auth_header": "Authorization", "auth_prefix": "Bearer "},
}

TIMEOUT_SECONDS = 20
MAX_RETRIES = 2

# 内存中保存的配置（重启即失效，绝不落盘）
_CONFIG = {"provider": None, "base_url": None, "model": None, "api_key": None}

# 回环地址放行（本机 LLM 如 Ollama 可用 http）；其余一律 https + 拒绝内网 IP
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _validate_base_url(url: str):
    """校验用户提交的 base_url，返回错误信息字符串；合法返回 None。

    防护目标（SSRF / 密钥钓鱼）：
    - 非回环地址强制 https，防止 API Key 被明文外发；
    - 主机为字面量 IP 时拒绝内网/保留/组播地址，防止打向内网服务。
    不做 DNS 解析：域名解析存在 TOCTOU 且会阻塞请求线程；
    域名形态的内网地址由 https 强制 + 调用方鉴权共同兜底。
    """
    u = (url or "").strip()
    if not u:
        return None                                # 空 = 使用服务商预设
    try:
        p = urlparse(u if "://" in u else "https://" + u)
    except Exception:
        return "Base URL 无法解析"
    host = (p.hostname or "").lower()
    if not host:
        return "Base URL 缺少主机名"
    if host in _LOOPBACK_HOSTS:
        return None                                # 本机回环放行（允许 http）
    if p.scheme != "https":
        return "非本机地址必须使用 https://，防止 API Key 明文外发"
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None                                # 域名形态：不做 DNS 解析
    if ip.is_loopback:
        return None
    if ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return f"目标地址 {ip} 位于内网/保留网段，已拒绝（SSRF 防护）"
    return None


def _env_key() -> str:
    return (os.getenv("LLM_API_KEY") or "").strip()


def get_config() -> dict:
    """返回当前配置（不含密钥本体，只说明是否已配置）。"""
    provider = _CONFIG.get("provider") or os.getenv("LLM_PROVIDER") or "modelarts_pangu"
    preset = PROVIDERS.get(provider, PROVIDERS["custom"])
    base_url = _CONFIG.get("base_url") or os.getenv("LLM_BASE_URL") or preset["base_url"]
    model = _CONFIG.get("model") or os.getenv("LLM_MODEL_NAME") or preset["model"]
    key = _CONFIG.get("api_key") or _env_key()
    return {
        "provider": provider,
        "provider_label": preset["label"],
        "base_url": base_url,
        "model": model,
        "configured": bool(key),
        "auth_header": preset["auth_header"],
        "note": "密钥仅保存在服务端内存中，重启后失效，不会写入任何文件。",
    }


def set_config(provider=None, base_url=None, model=None, api_key=None) -> dict:
    """保存配置；api_key 为空表示不修改已有密钥。base_url 非法时抛 ValueError。"""
    if base_url is not None:
        err = _validate_base_url(base_url)
        if err:
            raise ValueError(err)
    if provider:
        _CONFIG["provider"] = provider
    if base_url is not None:
        _CONFIG["base_url"] = base_url
    if model is not None:
        _CONFIG["model"] = model
    if api_key:
        _CONFIG["api_key"] = api_key
    cfg = get_config()
    # 只返回非敏感信息，密钥绝不回传
    return {"provider": cfg["provider"], "provider_label": cfg["provider_label"],
            "base_url": cfg["base_url"], "model": cfg["model"], "configured": cfg["configured"]}


def _endpoint(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _post(url: str, headers: dict, payload: dict, timeout: int = TIMEOUT_SECONDS):
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def chat(messages: list, temperature: float = 0.2, max_tokens: int = 1024,
         override: dict | None = None) -> dict:
    """调用大模型；失败时返回 {'ok': False, 'error': ...}，由调用方降级处理。"""
    cfg = get_config()
    ov = override or {}
    provider = ov.get("provider") or cfg["provider"]
    preset = PROVIDERS.get(provider, PROVIDERS["custom"])
    base_url = ov.get("base_url") or cfg["base_url"]
    model = ov.get("model") or cfg["model"]
    api_key = ov.get("api_key") or _CONFIG.get("api_key") or _env_key()
    if not api_key:
        return {"ok": False, "error": "未配置 API Key，AI 功能未启用"}
    if not base_url or not model:
        return {"ok": False, "error": "未配置 Base URL 或模型名称"}

    headers = {preset["auth_header"]: preset["auth_prefix"] + api_key}
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens}
    url = _endpoint(base_url)

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = _post(url, headers, payload)
            choices = data.get("choices") or []
            content = ""
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or choices[0].get("text") or ""
            return {"ok": True, "content": content, "raw": data}
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            last_err = f"HTTP {exc.code}：{body or exc.reason}"
            if exc.code in (400, 401, 403, 404):
                break                      # 密钥 / 模型名类错误不重试
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.4 * (attempt + 1))
    return {"ok": False, "error": last_err or "调用失败"}


def test_connection(override: dict | None = None) -> dict:
    """测试连接：返回成功（含延迟毫秒）或失败原因（密钥错误 / 网络不通 / 模型名不存在）。"""
    ov = override or {}
    cfg = get_config()
    provider = ov.get("provider") or cfg["provider"]
    preset = PROVIDERS.get(provider, PROVIDERS["custom"])
    base_url = ov.get("base_url") or cfg["base_url"]
    model = ov.get("model") or cfg["model"]
    api_key = ov.get("api_key") or _CONFIG.get("api_key") or _env_key()

    if not api_key:
        return {"connected": False, "status": "not_configured",
                "message": "未配置 API Key，请在设置页填写后测试（AI 功能将自动降级为模板解读）"}
    if not base_url or not model:
        return {"connected": False, "status": "incomplete",
                "message": "未填写 Base URL 或模型名称"}
    url_err = _validate_base_url(ov.get("base_url") or "") if ov.get("base_url") else None
    if url_err:
        return {"connected": False, "status": "invalid_url", "message": url_err}

    headers = {preset["auth_header"]: preset["auth_prefix"] + api_key}
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}],
               "max_tokens": 8, "temperature": 0}
    started = time.time()
    try:
        _post(_endpoint(base_url), headers, payload, timeout=TIMEOUT_SECONDS)
        latency = int((time.time() - started) * 1000)
        return {"connected": True, "status": "ok", "latency_ms": latency,
                "message": f"连接成功（{preset['label']}，模型 {model}，延迟 {latency} ms）"}
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        reason = ("密钥错误或没有访问权限" if exc.code in (401, 403)
                  else "模型名称不存在或接口路径错误" if exc.code == 404
                  else f"服务端返回 {exc.code}")
        return {"connected": False, "status": "http_error",
                "message": f"{reason}：{body or exc.reason}"}
    except urllib.error.URLError as exc:
        return {"connected": False, "status": "network_error",
                "message": f"网络不通或域名无法解析：{exc.reason}"}
    except Exception as exc:
        return {"connected": False, "status": "error", "message": f"连接失败：{exc}"}


def safe_extract_numbers(text: str) -> list:
    """从 AI 生成文本中提取数字，供后端与实际计算结果比对（P2 校验用）。"""
    import re
    return re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(text))


def is_available() -> bool:
    return get_config()["configured"]


__all__ = ["PROVIDERS", "get_config", "set_config", "chat", "test_connection",
           "is_available", "safe_extract_numbers", "prompts"]
