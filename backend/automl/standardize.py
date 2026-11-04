# medical-data-analysis/backend/automl/standardize.py
"""模块零：临床列名智能识别与标准化。

职责边界（与需求逐条对应）：
- 只做"列名 -> 标准名"的规范化与角色提示，**不改任何单元格数据**；
- 分层词典：全局基础词典（lexicon/global_lexicon.json，随版本发布，只读）
  + 用户级词典（data/lexicon/{tenant}.json，运行时累积，只存用户确认过的映射）；
- 多级匹配 L1 原名精确 -> L2 清洗后精确 -> L3 模糊（编辑距离/拼音首字母/全拼）
  -> L4 LLM 辅助（可选开关，默认关，复用 llm_service 配置）-> L5 未知列保留；
- 冲突处理：多列指向同一标准名不合并，全部保留并加来源后缀；
- 序号后缀（_2 / 第3次）仅识别指标身份并保留，重复测量列绝不合并。

清洗规则：
- 去首尾空格、全角转半角（含上标数字）、统一小写；
- 括号内容仅当整体匹配单位模式才剥离；非单位括号内容保留为限定词，
  形成独立标准名（如"血糖(餐后2小时)"），严禁与"血糖"合并；
- 数值括号（白细胞(2)）识别为序号而非限定词。
"""
from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

from config import REFERENCE_RANGES
from automl.errors import DataShapeError

__all__ = [
    "CleanedName", "ColumnMatch", "LexiconEntry", "Lexicon",
    "StandardizationResult", "clean_column_name", "load_global_lexicon",
    "load_user_lexicon", "save_user_mapping", "match_column",
    "standardize_dataframe", "apply_source_suffixes",
]

LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "lexicon", "global_lexicon.json")

# ------------------------------------------------------------------
# 全角 -> 半角（含上标数字，兼容 ×10⁹/L 写法）
# ------------------------------------------------------------------
_FW_MAP: dict[int, int] = {0x3000: 0x20}
_FW_MAP.update({i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)})
_FW_MAP.update({0xB2: ord("2"), 0xB3: ord("3"), 0xB9: ord("1")})
_FW_MAP.update({0x2070 + i: ord("0") + i for i in range(10)})


def _to_half_width(s: str) -> str:
    return s.translate(_FW_MAP)


# 单位模式（大小写不敏感）：仅括号内容整体命中才剥离
_UNIT_PATTERN = re.compile(
    r"^(?:mmol/?l|μmol/?l|umol/?l|g/?dl|u/?l|iu/?l|mg/?l|mg/?dl|ng/?ml|pg/?ml"
    r"|g/?l|fl|pg|%|mmhg|kpa|℃|°c|copies/?ml|次\s*/\s*分|bpm"
    r"|×?\s*10\s*\^?\s*\d+\s*/\s*l)$",
    re.IGNORECASE,
)

_PAREN = re.compile(r"[（(]([^（）()]*)[)）]")
_ORD_PREFIX = re.compile(r"^第\s*(\d+)\s*次\s*(?P<base>.+)$")
_ORD_SUFFIX_PAREN = re.compile(r"^(?P<base>.+?)[\s]*[（(]\s*(\d+)\s*[)）]\s*$")
_ORD_SUFFIX_US = re.compile(r"^(?P<base>.+?)_\s*(\d+)\s*$")
_ASCII_TOKEN = re.compile(r"[a-zA-Z0-9_]+")
_CN_CHAR = re.compile(r"[一-鿿]")
# 中英混合名分解：ascii 前缀/后缀 + 中文主体（如 HGB血红蛋白、血红蛋白HGB）
_MIX_PREFIX = re.compile(r"^([a-zA-Z0-9_]+)([一-鿿][一-鿿A-Za-z0-9_]*)$")
_MIX_SUFFIX = re.compile(r"^([一-鿿][一-鿿A-Za-z0-9_]*?)([a-zA-Z0-9_]+)$")


def _is_unit(text: str) -> bool:
    return bool(_UNIT_PATTERN.match(text.strip()))


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------
@dataclass
class CleanedName:
    """列名清洗结果：normalized 保留全名（含限定词括号），base 为匹配主体。"""

    raw: str
    normalized: str          # 全角转半角 + 去首尾空格 + 小写
    base: str                # 去掉括号与序号后缀后的基名
    qualifier: str | None    # 非单位括号内容（如"餐后2小时"），保留为限定词
    ordinal_suffix: str | None  # 序号（"_2"的内容"2"、"第3次"），仅标识不合并
    unit_stripped: str | None   # 被剥离的单位文本


@dataclass
class LexiconEntry:
    standard_name: str
    category: str = ""
    role_hint: str = "feature"
    english: list = field(default_factory=list)
    aliases: list = field(default_factory=list)
    pinyin_initials: str = ""
    reference_key: str | None = None
    value_domain: str = "any"
    unit: str = ""
    notes: str = ""


@dataclass
class ColumnMatch:
    raw_name: str
    standard_name: str | None
    level: str               # L1|L2|L3|L4|L5
    confidence: float        # 0..1
    status: str              # auto | pending | unknown
    role_hint: str | None
    source_suffix: str | None   # 同标准名多列并存时的来源后缀，如 "[WBC]"
    candidates: list = field(default_factory=list)      # [{standard_name, score}]
    confirm_reason: str | None = None
    reference_key: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "raw_name": self.raw_name,
            "standard_name": self.standard_name,
            "level": self.level,
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
            "role_hint": self.role_hint,
            "source_suffix": self.source_suffix,
            "candidates": self.candidates,
            "confirm_reason": self.confirm_reason,
            "note": self.note,
        }

    @property
    def ref(self) -> str:
        """下游引用键：标准名 + 来源后缀（如 白细胞计数[WBC]），报告双标签用。"""
        if self.standard_name and self.source_suffix:
            return f"{self.standard_name}{self.source_suffix}"
        return self.standard_name or self.raw_name


@dataclass
class StandardizationResult:
    columns: list            # list[ColumnMatch]，与 df 列一一对应
    conflicts: list          # [{standard_name, raw_columns, suffixes}]
    unknown_columns: list    # L5 列名
    mapping_table: list      # API 直出的完整映射表

    @property
    def pending(self) -> list:
        return [c for c in self.columns if c.status == "pending"]

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def ref_of_raw(self, raw_name: str) -> str:
        for c in self.columns:
            if c.raw_name == raw_name:
                return c.ref
        return raw_name

    def to_dict(self) -> dict:
        return {
            "columns": [c.to_dict() for c in self.columns],
            "conflicts": self.conflicts,
            "unknown_columns": self.unknown_columns,
            "pending_count": self.pending_count,
        }


class Lexicon:
    """词典索引：标准名 / 别名（含英文名）精确查 + 拼音首字母查。"""

    def __init__(self, entries: list):
        self.entries = entries
        self.by_standard: dict[str, LexiconEntry] = {}
        self.by_alias: dict[str, LexiconEntry] = {}
        self.by_initials: dict[str, list] = {}
        for e in entries:
            self.by_standard[e.standard_name.lower()] = e
            for a in list(e.english) + list(e.aliases):
                self.by_alias.setdefault(str(a).strip().lower(), e)
            if e.pinyin_initials:
                self.by_initials.setdefault(e.pinyin_initials.lower(), []).append(e)

    def match_exact(self, name: str) -> LexiconEntry | None:
        key = str(name).strip().lower()
        return self.by_alias.get(key) or self.by_standard.get(key)


# ------------------------------------------------------------------
# 列名清洗
# ------------------------------------------------------------------
def clean_column_name(raw: str) -> CleanedName:
    """只规范化名称，不改数据。规则见模块 docstring。"""
    original = "" if raw is None else str(raw)
    if not original.strip():
        raise DataShapeError("存在空白列名", suggestion="请为每一列提供非空列名后再上传")

    normalized = _to_half_width(original).strip().lower()
    work = normalized
    unit_stripped = None
    qualifier = None
    ordinal_suffix = None

    # 括号：单位剥离 / 数值视为序号 / 其余保留为限定词
    m = _PAREN.search(work)
    base_work = work
    if m:
        content = m.group(1).strip()
        if _is_unit(content):
            unit_stripped = content
            base_work = (work[:m.start()] + work[m.end():]).strip()
        elif content.isdigit():
            ordinal_suffix = content
            base_work = (work[:m.start()] + work[m.end():]).strip()
        else:
            qualifier = content          # 限定词保留在 normalized 全名中，base 剔除
            base_work = (work[:m.start()] + work[m.end():]).strip()

    # 序号：第N次前缀 或 _N 后缀
    base = base_work
    m = _ORD_PREFIX.match(base)
    if m:
        ordinal_suffix = f"第{m.group(1)}次"
        base = m.group("base").strip()
    elif ordinal_suffix is None:
        m = _ORD_SUFFIX_PAREN.match(base) or _ORD_SUFFIX_US.match(base)
        if m:
            ordinal_suffix = m.group(2)
            base = m.group("base").strip()
    base = base.strip(" _-")
    if not base:
        base = normalized
    return CleanedName(raw=original, normalized=normalized, base=base,
                       qualifier=qualifier, ordinal_suffix=ordinal_suffix,
                       unit_stripped=unit_stripped)


# ------------------------------------------------------------------
# 词典加载（全局 + 用户级）
# ------------------------------------------------------------------
_LEXICON_CACHE: Lexicon | None = None


def _merge_reference_ranges(entries: list) -> None:
    """把 config.REFERENCE_RANGES 的 key 与中文 label 并入对应条目的别名（程序内合并，不改词典文件）。"""
    by_ref = {e.reference_key: e for e in entries if e.reference_key}
    for key, ref in (REFERENCE_RANGES or {}).items():
        e = by_ref.get(key)
        if e is None:
            continue
        if key.lower() not in [str(x).lower() for x in e.english]:
            e.english.append(key)
        label = ref.get("label")
        if label and label.lower() not in [str(x).lower() for x in e.aliases]:
            e.aliases.append(label)


def load_global_lexicon() -> Lexicon:
    global _LEXICON_CACHE
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE
    if not os.path.exists(LEXICON_PATH):
        raise DataShapeError("全局词典文件缺失", suggestion=f"请检查 {LEXICON_PATH} 是否存在")
    with open(LEXICON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entries = [LexiconEntry(**{k: v for k, v in item.items() if k in LexiconEntry.__dataclass_fields__})
               for item in data.get("entries", [])]
    _merge_reference_ranges(entries)
    _LEXICON_CACHE = Lexicon(entries)
    return _LEXICON_CACHE


def _user_lexicon_path(tenant: str = "default") -> str:
    d = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "data", "lexicon"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{tenant}.json")


def load_user_lexicon(tenant: str = "default") -> dict:
    """返回 {原始列名(小写): 标准名}。文件损坏时按空处理，绝不中断流程。"""
    path = _user_lexicon_path(tenant)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {str(m["raw"]).lower(): m["standard"] for m in data.get("user_mappings", [])}
    except Exception:
        return {}


def save_user_mapping(raw: str, standard: str, tenant: str = "default") -> None:
    """用户确认过的映射写入其所属层级（用户级词典），不污染全局词典。"""
    path = _user_lexicon_path(tenant)
    data = {"user_mappings": []}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"user_mappings": []}
    ms = data.setdefault("user_mappings", [])
    low = str(raw).lower()
    for m in ms:
        if str(m["raw"]).lower() == low:
            m["standard"] = standard
            m["confirmed_at"] = datetime.now().isoformat(timespec="seconds")
            break
    else:
        ms.append({"raw": raw, "standard": standard,
                   "confirmed_at": datetime.now().isoformat(timespec="seconds"),
                   "source": "user_confirm"})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# 多级匹配
# ------------------------------------------------------------------
def _full_pinyin(s: str) -> str:
    try:
        from pypinyin import lazy_pinyin, Style
        return "".join(lazy_pinyin(s, style=Style.NORMAL))
    except Exception:
        return ""


def _mk(raw, standard, level, conf, status, entry, note=None, candidates=None, reason=None):
    return ColumnMatch(
        raw_name=raw, standard_name=standard, level=level, confidence=conf,
        status=status, role_hint=entry.role_hint if entry else None,
        source_suffix=None, candidates=candidates or [], confirm_reason=reason,
        reference_key=entry.reference_key if entry else None, note=note,
    )


def _best_ratio(token: str, keys: list):
    """编辑距离最优候选：(key, ratio) 或 None。用 difflib（标准库，不引新依赖）。"""
    if not keys:
        return None
    close = difflib.get_close_matches(token, keys, n=1, cutoff=0.0)
    if not close:
        return None
    return close[0], difflib.SequenceMatcher(None, token, close[0]).ratio()


def _top_candidates(token: str, lexicon: Lexicon, n: int = 3):
    out = []
    for key in difflib.get_close_matches(token, list(lexicon.by_alias.keys()), n=n, cutoff=0.7):
        e = lexicon.by_alias[key]
        if any(c["standard_name"] == e.standard_name for c in out):
            continue
        out.append({"standard_name": e.standard_name,
                    "score": round(difflib.SequenceMatcher(None, token, key).ratio(), 4)})
    return out


def _match_mixed(name: str, lexicon: Lexicon) -> LexiconEntry | None:
    """中英混合名（HGB血红蛋白 / 血红蛋白HGB）：中文主体须精确命中，
    且 ascii 部分须是该条目的英文别名/别名之一，二者一致才确认。"""
    for m in (_MIX_PREFIX.match(name), _MIX_SUFFIX.match(name)):
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        ascii_part, cn_part = (a, b) if _ASCII_TOKEN.fullmatch(a) else (b, a)
        if not _CN_CHAR.search(cn_part):
            continue
        e = lexicon.match_exact(cn_part)
        if e and ascii_part.lower() in [str(x).lower() for x in list(e.english) + list(e.aliases)]:
            return e
    return None


def _fuzzy_match(cleaned: CleanedName, lexicon: Lexicon):
    """L3：拼音首字母（歧义降级）-> 编辑距离（ascii）-> 全拼模糊（中文）。"""
    raw = cleaned.raw
    token = (cleaned.base or cleaned.normalized).strip()

    # (a) 纯字母串视为拼音首字母：唯一候选自动映射，多候选一律待确认
    if re.fullmatch(r"[a-zA-Z]{2,}", token):
        cands = lexicon.by_initials.get(token.lower(), [])
        if len(cands) == 1:
            return _mk(raw, cands[0].standard_name, "L3", 0.8, "auto", cands[0],
                       note="拼音首字母唯一命中")
        if len(cands) > 1:
            return ColumnMatch(
                raw_name=raw, standard_name=None, level="L3", confidence=0.6,
                status="pending", role_hint=None, source_suffix=None,
                candidates=[{"standard_name": c.standard_name, "score": round(1.0 / len(cands), 2)}
                            for c in cands],
                confirm_reason="拼音首字母存在多个候选，请确认",
            )

    # (b) ascii 缩写/拼写变体：编辑距离
    if _ASCII_TOKEN.fullmatch(token):
        best = _best_ratio(token.lower(), list(lexicon.by_alias.keys()))
        if best and best[1] >= 0.9:
            e = lexicon.by_alias[best[0]]
            return _mk(raw, e.standard_name, "L3", round(best[1], 4), "auto", e,
                       note="编辑距离近似命中")
        if best and best[1] >= 0.75:
            return ColumnMatch(
                raw_name=raw, standard_name=None, level="L3",
                confidence=round(best[1], 4), status="pending", role_hint=None,
                source_suffix=None, candidates=_top_candidates(token.lower(), lexicon),
                confirm_reason="存在近似的候选标准名，请确认",
            )

    # (c) 中文列名：全拼模糊
    if _CN_CHAR.search(token):
        py = _full_pinyin(token)
        if py:
            best_name, best_score = None, 0.0
            for e in lexicon.entries:
                for cand in [e.standard_name] + list(e.aliases):
                    if not _CN_CHAR.search(str(cand)):
                        continue
                    s = difflib.SequenceMatcher(None, py, _full_pinyin(str(cand))).ratio()
                    if s > best_score:
                        best_score, best_name = s, e.standard_name
            if best_name and best_score >= 0.9:
                return _mk(raw, best_name, "L3", round(best_score, 4), "auto",
                           lexicon.by_standard.get(best_name.lower()), note="全拼模糊命中")
            if best_name and best_score >= 0.75:
                return ColumnMatch(
                    raw_name=raw, standard_name=None, level="L3",
                    confidence=round(best_score, 4), status="pending", role_hint=None,
                    source_suffix=None,
                    candidates=[{"standard_name": best_name, "score": round(best_score, 4)}],
                    confirm_reason="全拼相似但不确定，请确认",
                )
    return None


_LLM_PROMPT = (
    "你是医疗数据列名标准化助手。把下列临床数据列名映射到最贴切的标准指标名，"
    "只输出标准名本身，不要任何解释；无法确定输出 UNKNOWN。\n"
    "可选标准名：{names}\n列名：{col}"
)


def _llm_suggest(cleaned: CleanedName, lexicon: Lexicon, sample_values=None):
    """L4（可选，默认关）：只发列名；附样例值（至多 3 个）。任何失败都降级为 None。"""
    try:
        import llm_service
        if not llm_service.is_available():
            return None
        names = "、".join(e.standard_name for e in lexicon.entries)
        col = cleaned.normalized
        if sample_values:
            col += f"（样例值：{list(sample_values)[:3]}）"
        r = llm_service.chat([{"role": "user", "content": _LLM_PROMPT.format(names=names, col=col)}],
                             temperature=0, max_tokens=32)
        if not r.get("ok"):
            return None
        text = (r.get("content") or "").strip()
        for e in lexicon.entries:
            if e.standard_name in text:
                return e.standard_name
        return None
    except Exception:
        return None


def match_column(cleaned: CleanedName, lexicon: Lexicon, user_lexicon: dict | None = None,
                 allow_llm: bool = False, sample_values=None) -> ColumnMatch:
    """对单列执行多级匹配。用户词典优先级最高（用户确认过的映射直接生效）。"""
    raw = cleaned.raw
    raw_key = raw.strip().lower()

    if user_lexicon and raw_key in user_lexicon:
        std = user_lexicon[raw_key]
        e = lexicon.by_standard.get(str(std).lower())
        return _mk(raw, std, "L1", 1.0, "auto", e, note="用户词典确认映射")

    e = lexicon.match_exact(raw.strip())                       # L1 原名精确
    if e:
        return _mk(raw, e.standard_name, "L1", 1.0, "auto", e)

    e = lexicon.match_exact(cleaned.normalized) or lexicon.match_exact(cleaned.base)  # L2
    if e:
        return _mk(raw, e.standard_name, "L2", 0.95, "auto", e)

    e = _match_mixed(cleaned.normalized, lexicon) or _match_mixed(cleaned.base, lexicon)
    if e:
        return _mk(raw, e.standard_name, "L2", 0.95, "auto", e, note="中英混合清洗命中")

    m = _fuzzy_match(cleaned, lexicon)                         # L3
    if m:
        return m

    if allow_llm:                                              # L4
        std = _llm_suggest(cleaned, lexicon, sample_values)
        if std:
            return _mk(raw, std, "L4", 0.6, "auto",
                       lexicon.by_standard.get(str(std).lower()), note="LLM 辅助识别")

    return ColumnMatch(raw_name=raw, standard_name=None, level="L5", confidence=0.0,
                       status="unknown", role_hint=None, source_suffix=None,
                       candidates=[], confirm_reason=None,
                       note="未识别，作为普通特征保留并在报告中列出")


# ------------------------------------------------------------------
# 整表标准化
# ------------------------------------------------------------------
def apply_source_suffixes(columns: list) -> StandardizationResult:
    """同标准名多列并存：不合并、全部保留，自动加来源后缀 [原始名]。"""
    by_std: dict[str, list] = {}
    for m in columns:
        if m.standard_name:
            by_std.setdefault(m.standard_name, []).append(m)

    conflicts = []
    for std, group in by_std.items():
        if len(group) > 1:
            for m in group:
                m.source_suffix = f"[{m.raw_name}]"
            conflicts.append({
                "standard_name": std,
                "raw_columns": [m.raw_name for m in group],
                "suffixes": [m.source_suffix for m in group],
            })

    unknown = [m.raw_name for m in columns if m.level == "L5"]
    table = [{
        "raw_name": m.raw_name,
        "standard_name": m.standard_name,
        "level": m.level,
        "status": m.status,
        "source_suffix": m.source_suffix,
        "role_hint": m.role_hint,
        "ref": m.ref,
    } for m in columns]
    return StandardizationResult(columns=columns, conflicts=conflicts,
                                 unknown_columns=unknown, mapping_table=table)


def standardize_dataframe(df, allow_llm: bool = False, tenant: str = "default",
                          sample_map: dict | None = None) -> StandardizationResult:
    """模块零入口：对整表列名做 清洗 -> 多级匹配 -> 冲突加后缀。

    只读列名，不触碰任何数据值；列顺序与 df 完全一致（重复测量列不合并）。
    """
    if df is None or getattr(df, "columns", None) is None or len(df.columns) == 0:
        raise DataShapeError("数据集为空或没有列",
                             suggestion="请上传包含列名与数据的文件后再分析")
    lexicon = load_global_lexicon()
    user = load_user_lexicon(tenant)
    columns = []
    for name in df.columns:
        cleaned = clean_column_name(str(name))
        columns.append(match_column(cleaned, lexicon, user,
                                    allow_llm=allow_llm,
                                    sample_values=(sample_map or {}).get(str(name))))
    return apply_source_suffixes(columns)
