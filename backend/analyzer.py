# medical-data-analysis/backend/analyzer.py
"""文件解析、数据清洗、描述性统计、正态性检验、单变量组间比较与相关性分析。

准确性红线：所有统计量均调用 pandas / numpy / scipy / statsmodels 成熟接口，
本模块不手写任何统计检验公式；仅对库返回结果做展示层换算（如 Cohen's d 由
库给出的合并标准误换算得到），并在返回值中注明依据。
"""

import json
import os
import re
import tempfile

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from config import (ALPHA, ALLOWED_EXTENSIONS, ENCODING_CANDIDATES, FDR_METHOD,
                    MIN_AVG_GROUP_SIZE, RANDOM_SEED, SHAPIRO_MAX_N)
from validators import detect_medical_outliers, reference_of

# ------------------------------------------------------------------
# 1. 文件解析
# ------------------------------------------------------------------

class ParseError(Exception):
    """文件解析失败，携带面向用户的明确原因。"""


# 无表头检测：首行数据被 pandas 当成列名（如 "1.0000000e+000"）时按数据行重读。
_NUMERIC_HEADER_RE = re.compile(r"^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?$")


def _looks_headerless(df: pd.DataFrame) -> bool:
    """判断文本表是否「没有表头」。

    真实医疗数据无表头时，首行数据会被 pandas 当成列名，表现不一定是纯数字：
    - 数值型文件：列名是 `1.0000000e+000`、`1`、`0`；
    - 符号/序列型文件：列名是 `+`/`-`，或整段 DNA 序列（Promoter 数据集）。
    因此只要 ≥50% 的列名“像数据值”（纯数字 / 单符号 / 超长串），即判定无表头。
    """
    if df.shape[0] < 2 or df.shape[1] < 2:
        return False

    def _suspicious(col) -> bool:
        n = str(col).strip()
        return bool(_NUMERIC_HEADER_RE.match(n)) or n in {"+", "-", "#", "?"} or len(n) >= 40

    names = [str(c).strip() for c in df.columns]
    hits = sum(1 for n in names if _suspicious(n))
    return hits >= max(1, int(0.5 * len(names)))


# SPSS/Stata 解析需要真实文件路径：临时文件放应用目录内而非系统临时目录，
# 避免患者数据明文散落在 /tmp（多用户机器可读）；进程重启时清空，覆盖崩溃残留场景。
_TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tmp")
os.makedirs(_TMP_DIR, exist_ok=True)


def _clean_tmp_dir() -> None:
    try:
        for name in os.listdir(_TMP_DIR):
            try:
                os.unlink(os.path.join(_TMP_DIR, name))
            except OSError:
                pass
    except OSError:
        pass


_clean_tmp_dir()


def _ext_of(filename: str) -> str:
    return os.path.splitext(str(filename))[1].lower()


def parse_table(content: bytes, filename: str) -> tuple:
    """统一把各类文件解析为 DataFrame，返回 (df, meta)。

    文本类文件按 ENCODING_CANDIDATES 顺序自动检测编码并重试；
    解析失败抛出 ParseError，错误信息可直接展示给用户。
    """
    ext = _ext_of(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ParseError(f"不支持的文件格式：{ext or '（无扩展名）'}。"
                         f"支持 {', '.join(sorted(ALLOWED_EXTENSIONS))}。"
                         "说明：ARFF（.arff）、医学影像（.dcm/.png/.jpg）、信号矩阵（.mat/.rar）"
                         "等非表格格式暂不支持，请先在本地转换为 csv/xlsx 后再上传。")

    import io
    buf = io.BytesIO(content)

    if ext in (".sav", ".dta"):
        try:
            import pyreadstat
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
            raise ParseError("读取 SPSS/Stata 文件需要 pyreadstat，请先 pip install pyreadstat") from exc
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=_TMP_DIR) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            reader = pyreadstat.read_sav if ext == ".sav" else pyreadstat.read_dta
            df, _meta = reader(tmp_path)
        except Exception as exc:
            raise ParseError(f"{ext} 文件解析失败：{exc}") from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return _tidy_frame(df), {"format": ext[1:], "encoding": "binary"}

    if ext in (".xlsx", ".xls"):
        try:
            sheets = pd.read_excel(buf, sheet_name=None)
        except Exception as exc:
            raise ParseError(f"Excel 文件解析失败：{exc}") from exc
        first = next(iter(sheets))
        meta = {"format": ext[1:], "encoding": "binary"}
        if len(sheets) > 1:
            # 多工作表文件只分析第一个表，但必须告知用户其余表的存在，
            # 避免数据被静默丢弃
            meta["sheets"] = list(sheets.keys())
            meta["note"] = f"检测到 {len(sheets)} 个工作表，当前分析使用第一个「{first}」"
        return _tidy_frame(sheets[first]), meta

    if ext == ".json":
        try:
            df = pd.read_json(buf)
            if df.empty:
                raise ValueError("空表")
        except Exception:
            try:
                obj = json.loads(content.decode("utf-8", errors="replace"))
                df = pd.json_normalize(obj if isinstance(obj, list) else [obj])
            except Exception as exc:
                raise ParseError(f"JSON 文件解析失败：{exc}") from exc
        return _tidy_frame(df), {"format": "json", "encoding": "utf-8"}

    # csv / tsv / txt：依次尝试候选编码
    sep = "\t" if ext == ".tsv" else None
    last_err = None
    for enc in ENCODING_CANDIDATES:
        try:
            buf.seek(0)
            bad_lines: list = []

            def _skip_bad(line, _bad=bad_lines):
                # 回调返回 None 表示跳过该行；保存原始字段列表用于提示样例
                _bad.append(line)
                return None

            try:
                if sep is None:
                    df = pd.read_csv(buf, encoding=enc, sep=None, engine="python")
                else:
                    df = pd.read_csv(buf, encoding=enc, sep=sep)
            except ParserError:
                # 真实医疗数据常见「行字段数不齐」（如 Echocardiogram 第 51 行多一列）。
                # 严格模式会让整份文件解析失败；这里降级为跳过坏行，并把跳过数量写进
                # meta 供确认页提示，绝不静默丢数据。
                buf.seek(0)
                if sep is None:
                    df = pd.read_csv(buf, encoding=enc, sep=None, engine="python",
                                     on_bad_lines=_skip_bad)
                else:
                    df = pd.read_csv(buf, encoding=enc, sep=sep, on_bad_lines=_skip_bad)
            if df.shape[1] == 0:
                raise ValueError("未解析出任何列")

            # 无表头文件（首行即数据）：pandas 会把首行数值当成列名，导致列名变成
            # "1.0000000e+000" 等，后续统计/建模全部错位。检测到后按数据行重读并
            # 以 col_1…col_N 命名，同时在 meta.note 中明确告知，绝不静默改写。
            headerless = False
            if _looks_headerless(df):
                try:
                    buf.seek(0)
                    if sep is None:
                        df = pd.read_csv(buf, encoding=enc, sep=None, engine="python",
                                         header=None, on_bad_lines=_skip_bad)
                    else:
                        df = pd.read_csv(buf, encoding=enc, sep=sep, header=None,
                                         on_bad_lines=_skip_bad)
                    df.columns = [f"col_{i + 1}" for i in range(df.shape[1])]
                    headerless = True
                except Exception:
                    headerless = False

            meta = {"format": ext[1:], "encoding": enc}
            notes = []
            if headerless:
                meta["headerless"] = True
                notes.append("未检测到表头：首行为数据，列名已按 col_1…col_N 命名。")
            if bad_lines:
                sample = " | ".join(str(v) for v in bad_lines[0][:4])[:60]
                meta["skipped_rows"] = len(bad_lines)
                notes.append(f"检测到 {len(bad_lines)} 行字段数与表头不一致，已跳过这些行"
                             f"（样例：{sample}）。如需保留请先在 Excel 中整理列数。")
            if notes:
                meta["note"] = " ".join(notes)
            return _tidy_frame(df), meta
        except Exception as exc:
            last_err = exc
            continue
    raise ParseError(f"文本文件解析失败，已尝试编码 {', '.join(ENCODING_CANDIDATES)}。"
                     f"最后一次错误：{last_err}")


# 额外的缺失值标记（大小写不敏感）：
# - "?"：UCI 临床数据集惯用的缺失标记；
# - "NaNN" / "NANN" 等：真实数据集里把 NaN 写串了的脏值（Cirrhosis 因此让
#   SimpleImputer 中位数填充拿到 object 列而抛 ValueError，最终整次建模 500）。
# pandas 默认的 NaN / nan / NULL / NA / N/A 识别保持不变（keep_default_na 仍为 True）。
EXTRA_NA_TOKENS = {"?", "nann"}


def _tidy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    读取后的统一规整，顺序固定：列名 trim → 单元格 trim → 缺失标记识别。

    必须在类型推断之前完成，否则：
      - 带尾部空白的值（如 "ckd\\t"）会被当成独立类别，污染唯一值统计与分类基数判断；
      - "?" 会被当成合法字符串值，而不是缺失。

    strip 后变为空串的单元格一律按缺失处理。
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    for col in out.columns:
        s = out[col]
        # 仅处理字符串/object 列：数值列不存在首尾空白与 "?" 问题
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        # 1) 逐值 strip 空白字符（空格 / \t / \r / \n）
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        # 2) strip 后为空串 → 缺失
        s = s.where(s != "", np.nan)
        # 3) 扩展缺失标记（"?"、"NaNN" 等，大小写不敏感）
        s = s.map(lambda v: np.nan if (isinstance(v, str) and v.strip().lower() in EXTRA_NA_TOKENS) else v)
        # 4) 数值型文本列转回数值：脏值（NaNN / 空格 / 异常符号）会让整列被判成 object，
        #    下游 imputer 与建模按 numeric 处理时就会出现 "median strategy with non-numeric data"。
        #    只要非空值里 ≥95% 能转成数值，就认定该列本质是数值列（脏值统一变缺失）。
        nonnull = s.notna().sum()
        if nonnull:
            conv = pd.to_numeric(s, errors="coerce")
            if int(conv.notna().sum()) / int(nonnull) >= 0.95:
                out[col] = conv
                continue
        out[col] = s.astype(object)
    return out


# ------------------------------------------------------------------
# 2. 数据清洗
# ------------------------------------------------------------------

def build_cleaning_report(df: pd.DataFrame, schema: list) -> dict:
    """生成清洗报告：重复行、缺失值、医学合理范围 / IQR 异常值。"""
    numeric_fields = [v["name"] for v in schema
                      if v.get("confirmed_type") == "numeric" and not v.get("excluded")]
    active = [v["name"] for v in schema if not v.get("excluded")]
    n_rows = int(len(df))

    # 从 schema 提取列名→标准名映射，供医学参考区间查找用
    rename_map = {v["name"]: v["standard_name"] for v in schema
                  if v.get("standard_name")}

    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())

    missing = []
    for col in active:
        if col not in df.columns:
            continue
        n_miss = int(df[col].isna().sum())
        if n_miss:
            missing.append({"field": col, "n_missing": n_miss,
                            "missing_rate": round(n_miss / max(1, n_rows), 4)})
    missing.sort(key=lambda x: -x["n_missing"])

    outliers = detect_medical_outliers(df, numeric_fields, rename_map=rename_map)
    outliers = [o for o in outliers if o["n_iqr_outlier"] or o["n_below_reference"] or o["n_above_reference"]]

    total_cells = max(1, n_rows * max(1, len(active)))
    total_missing = sum(m["n_missing"] for m in missing)
    return {
        "n_rows": n_rows,
        "n_cols": int(df.shape[1]),
        "n_active_cols": len(active),
        "n_duplicated_rows": n_dup,
        "duplicated_row_index": [int(i) for i in df.index[dup_mask].tolist()[:20]],
        "missing": missing,
        "total_missing_rate": round(total_missing / total_cells, 4),
        "outliers": outliers,
        "excluded_fields": [{"field": v["name"], "reason": v.get("exclude_reason")}
                            for v in schema if v.get("excluded")],
        "note": "缺失值默认按成对删除（pairwise deletion）处理，每项分析单独报告实际纳入样本量。",
    }


def apply_cleaning(df: pd.DataFrame, decisions: dict, rename_map: dict | None = None) -> tuple:
    """按用户确认的决策执行清洗，返回 (清洗后 df, 决策记录)。

    decisions:
      drop_duplicates: bool          是否删除重复行
      drop_columns: list             需要删除的字段
      outlier_action: keep|drop_rows 参考区间外异常值的处理
      drop_rows_missing_in: list     在指定字段上删除含缺失的行
    """
    decisions = decisions or {}
    rename_map = rename_map or {}
    out = df.copy()
    log = []
    n0 = len(out)

    if decisions.get("drop_duplicates"):
        before = len(out)
        out = out.drop_duplicates(keep="first")
        removed = before - len(out)
        rate = removed / max(1, before)
        detail = f"删除 {removed} 行，剩余 {len(out)} 行（占 {rate:.1%}）"
        if rate >= 0.5:
            detail += "；删除比例超过 50%，请确认这些行确为重复记录，而不是有效重复测量。"
        log.append({"action": "删除重复行", "detail": detail,
                    "removed": removed, "rate": round(rate, 4)})

    for col in decisions.get("drop_columns") or []:
        if col in out.columns:
            out = out.drop(columns=[col])
            log.append({"action": "删除字段", "detail": f"已删除字段「{col}」"})

    for col in decisions.get("drop_rows_missing_in") or []:
        if col in out.columns:
            before = len(out)
            out = out[out[col].notna()]
            log.append({"action": "删除缺失行", "detail": f"字段「{col}」缺失行删除 {before - len(out)} 行"})

    if decisions.get("outlier_action") == "drop_rows":
        before = len(out)
        mask = pd.Series(False, index=out.index)
        used = []
        for col in out.columns:
            std_name = rename_map.get(str(col))
            ref = reference_of(col, std_name)
            if not ref:
                continue
            s = pd.to_numeric(out[col], errors="coerce")
            col_mask = (s < ref["low"]) | (s > ref["high"])
            if col_mask.any():
                used.append(col)
                mask = mask | col_mask.fillna(False)
        out = out[~mask]
        log.append({"action": "剔除超参考区间样本",
                    "detail": f"涉及字段 {('、'.join(used) or '无')}，删除 {before - len(out)} 行"})

    if not log:
        log.append({"action": "未做修改", "detail": "保留全部样本与字段，缺失值按成对删除处理"})

    meta = {"n_before": n0, "n_after": len(out), "actions": log}
    drop_rate = 1 - (len(out) / max(1, n0))
    meta["drop_rate"] = round(drop_rate, 4)
    if drop_rate >= 0.2:
        meta["warning"] = (f"本次清洗共删除 {n0 - len(out)} 行（占 {drop_rate:.1%}），"
                           "请核对去重/缺失剔除规则是否符合预期。")
    return out.reset_index(drop=True), meta


# ------------------------------------------------------------------
# 3. 数值格式化（展示层）
# ------------------------------------------------------------------

def format_p(p) -> str:
    """p 值展示规范：p<0.001 显示 “<0.001”，其余保留三位小数。"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "-"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd)


def full_precision_p(x):
    """p 值以完整精度返回：任何统计检验的 p 值都不得在计算/校正环节截断舍入。

    原因：p=1.57e-6 一旦被 round(p, 3) 就塌缩成 0.0，BH-FDR 校正与
    "重要性排序"将随之失去区分度（多个真实 p 在 1e-6~1e-78 的特征会全部并列）。
    规则：
      - 检验结果与 FDR 校正必须基于本函数返回的完整精度值；
      - NaN / Inf 归一为 None，避免响应 JSON 序列化（allow_nan=False）崩溃；
      - 展示层格式化由 format_p（后端文案）与前端 fmtP 各自完成。
    """
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# ------------------------------------------------------------------
# 4. 描述性统计
# ------------------------------------------------------------------

def describe_numeric(df: pd.DataFrame, field: str, rename_map: dict | None = None) -> dict:
    s = pd.to_numeric(df[field], errors="coerce").dropna()
    n = int(s.size)
    std_name = (rename_map or {}).get(str(field))
    res = {"field": field, "n": n, "n_missing": int(df[field].isna().sum()),
           "reference_range": reference_of(field, std_name)}
    if n == 0:
        res.update({"note": "无有效数值，无法计算描述统计量", "valid": False})
        return res
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    q1, median, q3 = [float(v) for v in np.percentile(s, [25, 50, 75])]
    ci_lo = ci_hi = None
    if n > 1:
        ci_lo, ci_hi = DescrStatsW(s.to_numpy()).tconfint_mean(alpha=ALPHA)
    res.update({
        "valid": True,
        "mean": _f(s.mean()), "sd": _f(sd), "median": _f(median),
        "q1": _f(q1), "q3": _f(q3), "min": _f(s.min()), "max": _f(s.max()),
        "ci95": [_f(ci_lo), _f(ci_hi)],
        "skewness": _f(stats.skew(s, bias=False)) if n > 2 else None,
        "kurtosis": _f(stats.kurtosis(s, bias=False)) if n > 3 else None,
    })
    return res


def describe_categorical(df: pd.DataFrame, field: str) -> dict:
    s = df[field]
    vc = s.value_counts(dropna=True)
    total = int(vc.sum())
    levels = []
    for level, cnt in vc.items():
        levels.append({"level": str(level), "count": int(cnt),
                       "percent": round(int(cnt) / total, 4) if total else 0.0})
    return {"field": field, "n": total, "n_missing": int(s.isna().sum()),
            "n_unique": int(vc.size), "levels": levels, "valid": total > 0}


def histogram_data(df: pd.DataFrame, field: str, bins: int = 20, rename_map: dict | None = None) -> dict:
    s = pd.to_numeric(df[field], errors="coerce").dropna()
    if s.empty:
        return {"field": field, "counts": [], "edges": [], "n": 0}
    counts, edges = np.histogram(s, bins=bins)
    std_name = (rename_map or {}).get(str(field))
    return {"field": field, "n": int(s.size),
            "counts": [int(c) for c in counts],
            "edges": [float(e) for e in edges],
            "reference_range": reference_of(field, std_name)}


def boxplot_data(df: pd.DataFrame, field: str, group: str = None) -> dict:
    """箱线图数据：IQR 由 numpy 百分位数给出，异常点按 1.5×IQR 规则标注。"""
    if group:
        data = df[[field, group]].dropna()
        items = []
        for level, sub in data.groupby(group, observed=True):
            items.append({"level": str(level), **_box_one(pd.to_numeric(sub[field], errors="coerce"))})
        return {"field": field, "group": group, "groups": items}
    s = pd.to_numeric(df[field], errors="coerce").dropna()
    return {"field": field, **_box_one(s)}


def _box_one(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"n": 0}
    q1, med, q3 = [float(v) for v in np.percentile(s, [25, 50, 75])]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = s[(s >= lo) & (s <= hi)]
    return {"n": int(s.size), "q1": _f(q1), "median": _f(med), "q3": _f(q3),
            "lower_fence": _f(float(inside.min())) if not inside.empty else _f(q1),
            "upper_fence": _f(float(inside.max())) if not inside.empty else _f(q3),
            "outliers": [float(v) for v in s[(s < lo) | (s > hi)].tolist()[:50]],
            "mean": _f(s.mean())}


def qqplot_data(df: pd.DataFrame, field: str) -> dict:
    """Q-Q 图数据，理论分位数由 scipy.stats.probplot 给出。"""
    s = pd.to_numeric(df[field], errors="coerce").dropna()
    if s.size < 3:
        return {"field": field, "n": int(s.size), "points": []}
    (osm, osr), (slope, intercept, r) = stats.probplot(s.to_numpy(), dist="norm")
    return {"field": field, "n": int(s.size),
            "theoretical": [float(v) for v in osm],
            "sample": [float(v) for v in osr],
            "slope": _f(slope), "intercept": _f(intercept), "r": _f(r)}


# ------------------------------------------------------------------
# 5. 正态性检验
# ------------------------------------------------------------------

def normality_test(values) -> dict:
    """Shapiro-Wilk 正态性检验；n>500 时结果标记为仅参考。"""
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = int(arr.size)
    base = {"field": None, "n": n, "method": "Shapiro-Wilk", "usable": False}
    if n < 3:
        return {**base, "statistic": None, "p": None, "normal": None,
                "note": "有效样本量 < 3，无法进行正态性检验"}
    if n > SHAPIRO_MAX_N:
        arr_stat = arr if n <= 5000 else np.random.default_rng(RANDOM_SEED).choice(arr, 5000, replace=False)
        w, p = stats.shapiro(arr_stat)
        return {**base, "usable": True, "statistic": _f(w), "p": full_precision_p(p), "normal": bool(p >= ALPHA),
                "reference_only": True,
                "note": (f"样本量 n={n}>{SHAPIRO_MAX_N}，Shapiro-Wilk 对大样本过于敏感，"
                         f"结果仅作参考，请结合直方图与 Q-Q 图综合判断")}
    w, p = stats.shapiro(arr)
    return {**base, "usable": True, "statistic": _f(w), "p": full_precision_p(p), "normal": bool(p >= ALPHA),
            "reference_only": False,
            "note": f"α={ALPHA}，{'不拒绝正态性假设' if p >= ALPHA else '拒绝正态性假设'}"}


# ------------------------------------------------------------------
# 6. 单变量组间比较
# ------------------------------------------------------------------

def compare_groups(df: pd.DataFrame, target: str, group: str, target_type: str) -> dict:
    """按变量类型、组数、正态性与方差齐性自动选择检验方法。

    target_type: 'numeric' 连续型结局 / 'categorical' 分类型结局
    """
    data = df[[target, group]].dropna()
    n_total = int(len(data))
    out = {"target": target, "group": group, "target_type": target_type,
           "n_total": n_total, "alpha": ALPHA, "warnings": []}
    if n_total < 3:
        return {**out, "valid": False, "note": "有效样本量 < 3，无法进行组间比较"}
    levels = [str(v) for v in pd.unique(data[group])]
    if len(levels) < 2:
        return {**out, "valid": False, "note": f"分组变量「{group}」只有 {len(levels)} 个水平，无法比较"}
    if n_total / len(levels) < MIN_AVG_GROUP_SIZE:
        return {**out, "valid": False,
                "note": (f"分组变量「{group}」共有 {len(levels)} 个不同取值、"
                         f"平均每组样本量不足 {MIN_AVG_GROUP_SIZE}，"
                         "疑似把连续变量误用作分组变量：组数接近样本量时 η² 必然趋近 1、"
                         "p 值无意义。请选择分类变量（如诊断、分组）作为分组，"
                         "或先将其分箱/合并类别后再比较。")}

    out["group_levels"] = levels
    out["group_summary"] = _group_summary(data, target, group, target_type)
    if any(s["n"] < 2 for s in out["group_summary"]):
        out["warnings"].append("存在样本量 < 2 的分组，检验效能不足")
    if any(s["n"] < 5 for s in out["group_summary"]):
        out["warnings"].append("存在样本量 < 5 的分组，检验结论需谨慎")

    if target_type == "numeric":
        result = _compare_numeric(data, target, group, levels)
    else:
        result = _compare_categorical(data, target, group, levels)
    out.update(result)
    out["valid"] = True
    return out


def _group_summary(data, target, group, target_type):
    summary = []
    if target_type == "numeric":
        for level, sub in data.groupby(group, observed=True):
            s = pd.to_numeric(sub[target], errors="coerce").dropna()
            summary.append({
                "level": str(level), "n": int(s.size),
                "mean": _f(s.mean()) if s.size else None,
                "sd": _f(s.std(ddof=1)) if s.size > 1 else None,
                "median": _f(np.median(s)) if s.size else None,
                "q1": _f(np.percentile(s, 25)) if s.size else None,
                "q3": _f(np.percentile(s, 75)) if s.size else None,
            })
    else:
        table = pd.crosstab(data[group], data[target])
        for level in table.index:
            row = table.loc[level]
            total = int(row.sum())
            summary.append({"level": str(level), "n": total,
                            "counts": {str(c): int(v) for c, v in row.items()},
                            "percents": {str(c): round(int(v) / total, 4) if total else 0.0
                                         for c, v in row.items()}})
    return summary


def _compare_numeric(data, target, group, levels):
    samples = [pd.to_numeric(data.loc[data[group] == lv, target], errors="coerce").dropna().to_numpy(float)
               for lv in pd.unique(data[group])]
    samples = [s for s in samples if s.size > 0]
    k = len(samples)

    norms = [normality_test(s) for s in samples]
    all_normal = all(n["usable"] and n["normal"] for n in norms)
    lev_res = stats.levene(*samples)
    levene_stat, levene_p = float(lev_res.statistic), float(lev_res.pvalue)
    equal_var = bool(levene_p >= ALPHA)

    checks = [
        {"name": "正态性检验（Shapiro-Wilk）",
         "detail": "；".join(f"组「{lv}」p={format_p(n['p'])}" for lv, n in zip(levels, norms)),
         "passed": all_normal},
        {"name": "方差齐性检验（Levene）",
         "detail": f"统计量={_f(levene_stat)}, p={format_p(levene_p)}", "passed": equal_var},
    ]

    # η²：由 statsmodels 方差分析分解给出，作为多组比较的描述性效应量
    eta_sq = _eta_squared(data, target, group)

    if k == 2:
        a, b = samples[0], samples[1]
        if all_normal:
            t_res = stats.ttest_ind(a, b, equal_var=equal_var)
            t_stat, p = float(t_res.statistic), float(t_res.pvalue)
            d1, d2 = DescrStatsW(a), DescrStatsW(b)
            cm = CompareMeans(d1, d2)
            usevar = "pooled" if equal_var else "unequal"
            ci_lo, ci_hi = cm.tconfint_diff(alpha=ALPHA, alternative="two-sided", usevar=usevar)
            diff = float(d1.mean - d2.mean)
            # Cohen's d：由 CompareMeans 给出的合并标准误换算，非独立检验公式
            se = cm.std_meandiff_pooledvar
            sp = float(se) / np.sqrt(1.0 / len(a) + 1.0 / len(b))
            cohen_d = diff / sp if sp > 0 else None
            name = "独立样本 t 检验（Student）" if equal_var else "独立样本 t 检验（Welch 校正）"
            return {
                "test": {"name": name, "statistic": _f(t_stat), "p": full_precision_p(p),
                         "assumption_checks": checks,
                         "note": ("" if equal_var else "Levene 检验提示方差不齐，已自动切换 Welch 校正。")},
                "effect": {"name": "Cohen's d", "value": _f(cohen_d),
                           "interpretation": _d_interpret(cohen_d)},
                "estimate": {"label": f"均值差（{levels[0]} − {levels[1]}）", "value": _f(diff),
                             "ci": [_f(ci_lo), _f(ci_hi)]},
            }
        mw_res = stats.mannwhitneyu(a, b, alternative="two-sided")
        u_stat, p = float(mw_res.statistic), float(mw_res.pvalue)
        # 秩二列相关：由 scipy 给出的 U 统计量换算得到的效应量
        r_rb = 2.0 * float(u_stat) / (len(a) * len(b)) - 1.0
        return {
            "test": {"name": "Mann-Whitney U 检验", "statistic": _f(u_stat), "p": full_precision_p(p),
                     "assumption_checks": checks,
                     "note": "至少一组不服从正态分布，故采用非参数检验；该路径不提供均值差置信区间。"},
            "effect": {"name": "秩二列相关 r", "value": _f(r_rb), "interpretation": _r_interpret(r_rb)},
            "estimate": {"label": f"中位数差（{levels[0]} − {levels[1]}）",
                         "value": _f(float(np.median(a) - np.median(b))), "ci": None},
        }

    if all_normal:
        if equal_var:
            model = ols(f'Q("{target}") ~ C(Q("{group}"))', data=data).fit()
            table = anova_lm(model, typ=2)
            row = table.loc[f'C(Q("{group}"))']
            return {
                "test": {"name": "单因素方差分析（One-way ANOVA）", "statistic": _f(row["F"]),
                         "p": full_precision_p(row["PR(>F)"]), "assumption_checks": checks, "note": ""},
                "effect": {"name": "η²", "value": _f(eta_sq), "interpretation": _eta_interpret(eta_sq)},
                "estimate": {"label": "组间方差分解 η²", "value": _f(eta_sq), "ci": None},
            }
        ag_res = stats.alexandergovern(*samples)
        ag_stat, p = float(ag_res.statistic), float(ag_res.pvalue)
        return {
            "test": {"name": "Alexander-Govern 检验（方差不齐时的 Welch 型校正）",
                     "statistic": _f(ag_stat), "p": full_precision_p(p), "assumption_checks": checks,
                     "note": "Levene 检验提示方差不齐，已自动切换为 Welch 型校正检验。"},
            "effect": {"name": "η²", "value": _f(eta_sq), "interpretation": _eta_interpret(eta_sq)},
            "estimate": {"label": "组间方差分解 η²", "value": _f(eta_sq), "ci": None},
        }

    kw_res = stats.kruskal(*samples)
    h_stat, p = float(kw_res.statistic), float(kw_res.pvalue)
    return {
        "test": {"name": "Kruskal-Wallis H 检验", "statistic": _f(h_stat), "p": full_precision_p(p),
                 "assumption_checks": checks,
                 "note": "至少一组不服从正态分布，故采用非参数检验。"},
        "effect": {"name": "η²", "value": _f(eta_sq), "interpretation": _eta_interpret(eta_sq)},
        "estimate": {"label": "组间方差分解 η²", "value": _f(eta_sq), "ci": None},
    }


def _eta_squared(data, target, group):
    """η² = 组间平方和 / 总平方和，平方和由 statsmodels anova_lm 分解得到。"""
    try:
        model = ols(f'Q("{target}") ~ C(Q("{group}"))', data=data).fit()
        table = anova_lm(model, typ=2)
        ss_total = float(table["sum_sq"].sum())
        ss_group = float(table.loc[f'C(Q("{group}"))', "sum_sq"])
        return ss_group / ss_total if ss_total > 0 else None
    except Exception:
        return None


def _compare_categorical(data, target, group, levels):
    table = pd.crosstab(data[group], data[target])
    observed = table.to_numpy()
    chi_res = stats.chi2_contingency(observed)
    chi2, p_chi, dof, expected = (float(chi_res.statistic), float(chi_res.pvalue),
                                   int(chi_res.dof), chi_res.expected_freq)
    min_expected = float(expected.min())
    shape = observed.shape

    checks = [{"name": "卡方检验适用性（期望频数）",
               "detail": f"最小期望频数={min_expected:.2f}", "passed": min_expected >= 5}]

    try:
        cramers_v = float(stats.contingency.association(observed, method="cramer"))
    except Exception:  # pragma: no cover - 低版本 scipy 兜底
        cramers_v = None

    if shape == (2, 2) and min_expected < 5:
        fisher_res = stats.fisher_exact(observed)
        or_point, p = float(fisher_res[0]), float(fisher_res[1])
        orr = stats.contingency.odds_ratio(observed)
        ci = orr.confidence_interval(1 - ALPHA)
        return {
            "test": {"name": "Fisher 精确检验", "statistic": None, "p": full_precision_p(p),
                     "assumption_checks": checks,
                     "note": f"存在期望频数 <5 的格子（最小期望频数={min_expected:.2f}），已自动改用 Fisher 精确检验。"},
            "effect": {"name": "比值比 OR", "value": _f(or_point), "interpretation": "OR>1 提示暴露组事件风险更高"},
            "estimate": {"label": "比值比 OR", "value": _f(float(orr.statistic)),
                         "ci": [_f(ci.low), _f(ci.high)],
                         "note": "OR 与置信区间为 scipy 条件最大似然估计结果"},
            "contingency": _table_payload(table),
        }

    note = ""
    if min_expected < 5:
        note = (f"存在期望频数 <5 的格子（最小期望频数={min_expected:.2f}），卡方检验结果不稳定；"
                f"当前为 {shape[0]}×{shape[1]} 表，scipy 未提供对应的精确 Fisher 实现，"
                f"建议合并稀疏类别或扩大样本量。")
    estimate = {"label": "Cramér's V", "value": _f(cramers_v), "ci": None}
    if shape == (2, 2):
        orr = stats.contingency.odds_ratio(observed)
        ci = orr.confidence_interval(1 - ALPHA)
        estimate = {"label": "比值比 OR", "value": _f(float(orr.statistic)),
                    "ci": [_f(ci.low), _f(ci.high)],
                    "note": "OR 与置信区间为 scipy 条件最大似然估计结果"}
    return {
        "test": {"name": "卡方检验（Pearson）", "statistic": _f(chi2), "p": full_precision_p(p_chi),
                 "assumption_checks": checks, "note": note},
        "effect": {"name": "Cramér's V", "value": _f(cramers_v),
                   "interpretation": _v_interpret(cramers_v)},
        "estimate": estimate,
        "contingency": _table_payload(table),
    }


def _table_payload(table: pd.DataFrame) -> dict:
    return {"rows": [str(i) for i in table.index],
            "cols": [str(c) for c in table.columns],
            "values": [[int(v) for v in row] for row in table.to_numpy()]}


def _d_interpret(d):
    if d is None:
        return "无法计算"
    d = abs(d)
    if d < 0.2:
        return "效应量极小"
    if d < 0.5:
        return "效应量小"
    if d < 0.8:
        return "效应量中等"
    return "效应量大"


def _r_interpret(r):
    if r is None:
        return "无法计算"
    r = abs(r)
    if r < 0.1:
        return "效应量极小"
    if r < 0.3:
        return "效应量小"
    if r < 0.5:
        return "效应量中等"
    return "效应量大"


def _eta_interpret(e):
    if e is None:
        return "无法计算"
    if e < 0.01:
        return "效应量极小"
    if e < 0.06:
        return "效应量小"
    if e < 0.14:
        return "效应量中等"
    return "效应量大"


def _v_interpret(v):
    if v is None:
        return "无法计算"
    if v < 0.1:
        return "关联极弱"
    if v < 0.3:
        return "关联较弱"
    if v < 0.5:
        return "关联中等"
    return "关联较强"


# ------------------------------------------------------------------
# 7. 相关性分析
# ------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame, fields: list) -> dict:
    """数值变量相关矩阵：两变量均服从正态用 Pearson，否则用 Spearman。

    p 值经 Benjamini-Hochberg（FDR）校正，结论以校正后 p 值为准。
    """
    sub = df[fields].apply(pd.to_numeric, errors="coerce")
    cols = list(fields)
    m = len(cols)
    r_mat = [[1.0 if i == j else None for j in range(m)] for i in range(m)]
    p_mat = [[None] * m for _ in range(m)]
    pf_mat = [[None] * m for _ in range(m)]
    n_mat = [[0] * m for _ in range(m)]
    method_mat = [["-"] * m for _ in range(m)]

    pair_items = []
    for i in range(m):
        for j in range(i + 1, m):
            pair = sub[[cols[i], cols[j]]].dropna()
            n = int(len(pair))
            n_mat[i][j] = n_mat[j][i] = n
            if n < 3:
                continue
            xi = pair[cols[i]].to_numpy(float)
            xj = pair[cols[j]].to_numpy(float)
            nrm_i = normality_test(xi)["normal"]
            nrm_j = normality_test(xj)["normal"]
            method = "pearson" if (nrm_i and nrm_j) else "spearman"
            fn = stats.pearsonr if method == "pearson" else stats.spearmanr
            res = fn(xi, xj)
            r_val, p_val = float(res.statistic), float(res.pvalue)
            r_mat[i][j] = r_mat[j][i] = _f(r_val, 4)
            p_mat[i][j] = p_mat[j][i] = full_precision_p(p_val)
            method_mat[i][j] = method_mat[j][i] = method
            pair_items.append((i, j))

    if pair_items:
        raw = [p_mat[i][j] for i, j in pair_items]
        # 统一走 fdr_correct：常量/退化 pair 会产生 None，不能直接交给 multipletests
        corrected = fdr_correct(raw)["p_fdr"]
        for (i, j), pc in zip(pair_items, corrected):
            pf_mat[i][j] = pf_mat[j][i] = pc

    return {"columns": cols, "r": r_mat, "p": p_mat, "p_fdr": pf_mat,
            "n": n_mat, "method": method_mat,
            "note": "两变量均满足正态性时采用 Pearson，否则采用 Spearman；p 值为 Benjamini-Hochberg 校正后结果。"}


def fdr_correct(p_values: list) -> dict:
    """对一组 p 值做 Benjamini-Hochberg 校正。

    输入的 p 值应来自 full_precision_p（未舍入）；返回的 p / p_fdr 同样保持
    完整精度——若在此处舍入，1e-78 量级的 p 会塌缩成 0.0 并列，排序失去区分度。

    非有限 p 值（None / NaN / Inf）通常来自常量输入、样本不足或检验退化，
    本身没有可比较的数值。为避免“无定义检验被漏掉、多重比较分母变小”的
    保守性争议，这里按 p=1 计入 BH 校正的分母（不拒绝原假设），但仍不对
    这些位置返回 p_fdr / 显著性：其 p_fdr 保持 None、significant=False，
    展示层继续显示“-”，不会把无定义检验伪装成 p=1 的正式检验。
    修复前直接把 None 传给 statsmodels.multipletests 会触发
    "TypeError: '<' not supported between instances of 'NoneType' and 'float'"
    （相关矩阵遇到常量列即 500），此处在共享入口统一兜底。
    """
    if not p_values:
        return {"p": [], "p_fdr": [], "significant": []}

    cleaned = [full_precision_p(v) for v in p_values]
    valid_idx = [i for i, v in enumerate(cleaned) if v is not None]
    p_fdr = [None] * len(cleaned)
    significant = [False] * len(cleaned)
    if valid_idx:
        # 无定义检验按 p=1 参与 BH：保留完整测试数 m，使有效检验的校正更保守
        tests = [p if p is not None else 1.0 for p in cleaned]
        reject, corrected, _, _ = multipletests(tests, alpha=ALPHA, method=FDR_METHOD)
        for i in valid_idx:
            significant[i] = bool(reject[i])
            p_fdr[i] = full_precision_p(corrected[i])
    return {"p": cleaned, "p_fdr": p_fdr, "significant": significant}
