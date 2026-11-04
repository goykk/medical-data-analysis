# medical-data-analysis/backend/multivariable.py
"""多因素分析：二分类结局用 Logistic 回归，连续结局用多元线性回归（statsmodels）。

输出调整后 OR / 回归系数、95% 置信区间、p 值、VIF，并给出单变量（校正前）
结果用于"校正前 vs 校正后"对照与混杂因素提示。
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from analyzer import compare_groups, fdr_correct, full_precision_p
from config import ALPHA, MIN_AVG_GROUP_SIZE, MIN_EPV, VIF_THRESHOLD


class ModelingError(Exception):
    """多因素建模无法执行时抛出，携带面向用户的明确原因。"""


# 事件水平启发式：二分类结局中若恰好一个水平是"阳性类"取值，则以它为事件。
# 纯按排序取 levels[1] 在「阴性/阳性」上会错误建模 P(阴性)（Unicode 码点 阳<阴），
# 与临床"P(阳性/事件发生)"的直觉相反。
_POSITIVE_LEVELS = {"1", "yes", "y", "true", "positive", "pos", "event", "dead",
                    "death", "malignant", "abnormal", "m",
                    "阳性", "是", "有", "异常", "恶性", "死亡", "复发", "转移", "发生"}


def _pick_event_level(levels: list) -> tuple:
    """返回 (参照水平, 事件水平)：恰好一个阳性类取值时以它为事件，否则按排序取后者。"""
    pos = [lv for lv in levels if str(lv).strip().lower() in _POSITIVE_LEVELS]
    if len(pos) == 1:
        event = pos[0]
        ref = next(lv for lv in levels if lv != event)
        return ref, event
    return levels[0], levels[1]


def _design_block(data: pd.DataFrame, var: str, kind: str) -> tuple:
    """为单个变量构造设计矩阵分块，并返回编码说明（含分类变量参照组）。"""
    if kind == "categorical":
        s = data[var].astype(str)
        levels = sorted(pd.unique(s))
        if len(levels) < 2:
            raise ModelingError(f"分类变量「{var}」在本分析中有效水平不足 2 个，无法编码")
        dummies = pd.get_dummies(s, prefix=var, drop_first=True).astype(float)
        return dummies, {"variable": var, "kind": "categorical", "levels": levels,
                         "reference": levels[0], "columns": list(dummies.columns)}
    s = pd.to_numeric(data[var], errors="coerce")
    if s.notna().sum() == 0:
        raise ModelingError(f"数值变量「{var}」无有效取值")
    block = pd.DataFrame({var: s.astype(float)})
    return block, {"variable": var, "kind": "numeric", "levels": None,
                   "reference": None, "columns": [var]}


def _column_owner(col: str, coding: list) -> str:
    """把设计矩阵列映射回原始变量名（分类哑变量归到其原变量）。"""
    for info in coding:
        if col in info["columns"]:
            return info["variable"]
    return col


def _check_design_rank(X: pd.DataFrame, coding: list) -> None:
    """拟合前检查设计矩阵是否可估计，避免把 statsmodels 的 Singular matrix 直接抛给用户。

    - 设计列数（含截距）不少于样本量：参数多于观测，模型无法估计；
    - 常量变量：在整个完整病例子集中只有一个取值，与截距完全共线；
    - 完全共线变量：设计矩阵不满秩（如 LBE 与 LB 相关系数恰为 1）。
    这些情况都无法得到有意义的系数/OR，必须在拟合前明确指出并让用户调整。
    """
    Xc = sm.add_constant(X, has_constant="add")
    n_rows = int(Xc.shape[0])
    if Xc.shape[1] >= n_rows:
        raise ModelingError(
            f"多因素设计矩阵不可估计：编码后的自变量列数（{Xc.shape[1]}，含截距）"
            f"不少于完整病例样本量（{n_rows}）。请减少自变量、合并稀疏类别，"
            "或增加样本量后重试。")

    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        owners = sorted({_column_owner(c, coding) for c in const_cols})
        raise ModelingError(
            f"多因素设计矩阵不可估计：检测到常量变量「{'、'.join(owners)}」"
            "（在整个完整病例子集中只有一个取值）。常量变量不提供信息，"
            "请剔除后重新拟合。")

    # 中心化 + 归一化后再判断秩：matrix_rank 的容差与绝对尺度和均值都相关。
    # 不处理时，“年龄 0~100”与“浓度 1e-12”或“均值 1e9、方差很小”的独立变量
    # 可能被误判为 0 方差/共线；中心化后只反映真实的线性相关性。
    mat = Xc.to_numpy(dtype=float)
    scaled_cols = []
    for j, col in enumerate(Xc.columns):
        v = mat[:, j]
        if col == "const":
            scaled_cols.append(np.full(len(v), 1.0 / np.sqrt(len(v))))
            continue
        v = v - np.mean(v)
        norm = float(np.linalg.norm(v))
        scaled_cols.append(v / norm if norm > 0 else v)
    scaled = np.column_stack(scaled_cols)
    rank = int(np.linalg.matrix_rank(scaled))
    if rank >= scaled.shape[1]:
        return

    # 逐个删除设计列：若删除后秩不变，说明该列可由其余列线性表示
    redundant = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        if np.linalg.matrix_rank(np.delete(scaled, i, axis=1)) == rank:
            redundant.append(col)
    owners = sorted({_column_owner(c, coding) for c in redundant}) or redundant
    raise ModelingError(
        f"多因素设计矩阵不可估计：检测到完全共线变量「{'、'.join(map(str, owners))}」"
        f"（设计矩阵秩 {rank} < 列数 {mat.shape[1]}）。完全共线时系数与 OR 无法唯一确定，"
        "请剔除其中冗余变量、合并高度相关变量，或先做相关性/VIF 检查后重新拟合。")


def _fit(X: pd.DataFrame, y: np.ndarray, task: str):
    Xc = sm.add_constant(X, has_constant="add")
    if task == "logistic":
        try:
            return sm.Logit(y, Xc).fit(disp=0), Xc
        except PerfectSeparationError as exc:
            raise ModelingError("出现完全分离（perfect separation），Logistic 回归无法收敛。"
                                "建议减少自变量、合并稀疏类别或改用精确 Logistic 回归。") from exc
        except np.linalg.LinAlgError as exc:
            raise ModelingError(
                "Logistic 回归无法收敛：似然函数的 Hessian 矩阵奇异。"
                "常见原因是完全分离（某个自变量几乎由结局决定，或由结局派生，"
                "如吸烟状态与吸烟年数/包年数同时入选），或极端共线。"
                "请检查并剔除与结局高度相关的变量，或减少自变量后重试。") from exc
        except Exception as exc:
            text = str(exc)
            if "singular" in text.lower():
                raise ModelingError(
                    "Logistic 回归无法收敛：设计矩阵奇异，通常由完全分离或完全共线导致。"
                    "请检查并剔除与结局高度相关的变量，或减少自变量后重试。") from exc
            raise ModelingError(f"Logistic 回归拟合失败：{exc}") from exc
    try:
        return sm.OLS(y, Xc).fit(), Xc
    except np.linalg.LinAlgError as exc:
        raise ModelingError(
            "线性回归无法拟合：设计矩阵奇异（完全共线或常量变量），"
            "请剔除冗余变量后重试。") from exc
    except Exception as exc:
        raise ModelingError(f"线性回归拟合失败：{exc}") from exc


def _term_rows(res, Xc, coding, task: str) -> list:
    """把模型系数整理为森林图所需的行（每行一个设计矩阵列）。"""
    conf = res.conf_int(alpha=ALPHA)
    rows = []
    for col in Xc.columns:
        if col == "const":
            continue
        coef = float(res.params[col])
        lo, hi = float(conf.loc[col, 0]), float(conf.loc[col, 1])
        p = float(res.pvalues[col])
        if task == "logistic":
            or_val = float(np.exp(coef))
            if not np.isfinite(or_val) or or_val > 1e6:
                # 完全分离时系数发散，不返回无意义的大数值
                row = {"term": col, "estimate": None, "ci": [None, None], "estimate_label": "OR",
                       "note": "估计值发散（疑似完全分离），该变量效应无法估计"}
            else:
                row = {"term": col, "estimate": _r(or_val), "ci": [_r(np.exp(lo)), _r(np.exp(hi))],
                       "estimate_label": "OR"}
        else:
            row = {"term": col, "estimate": _r(coef), "ci": [_r(lo), _r(hi)]}
            row["estimate_label"] = "回归系数 β"
        # p 值保持完整精度（严禁 round 截断：1e-78 量级的 p 舍入后会变 0.0），
        # p_text 专供展示层使用
        row.update({"p": full_precision_p(p), "p_text": _p_text(p), "significant": bool(p < ALPHA)})
        owner = next((c for c in coding if col in c["columns"]), None)
        if owner:
            row["variable"] = owner["variable"]
            row["variable_kind"] = owner["kind"]
            row["level"] = None if owner["kind"] == "numeric" else _level_of(col, owner)
            row["reference"] = owner["reference"]
        else:
            row["variable"], row["level"], row["reference"], row["variable_kind"] = col, None, None, "numeric"
        rows.append(row)
    return rows


def _level_of(col: str, owner: dict) -> str:
    prefix = owner["variable"] + "_"
    return col[len(prefix):] if col.startswith(prefix) else col


def _r(x, nd=4):
    if x is None or np.isnan(x) or np.isinf(x):
        return None
    return round(float(x), nd)


def _p_text(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def run_multivariable(df: pd.DataFrame, outcome: str, predictors: list, covariates: list,
                      variable_types: dict, task: str = "auto") -> dict:
    """执行多因素分析。

    df              原始 DataFrame（含全部字段）
    outcome         结局变量名
    predictors      目标自变量（关注变量）
    covariates      待校正协变量
    variable_types  {字段名: 'numeric'|'categorical'}
    task            'auto' | 'logistic' | 'linear'
    """
    # 去重且保序：同一变量重复传入会让设计矩阵出现重复列
    predictors = list(dict.fromkeys(p for p in (predictors or []) if p != outcome))
    covariates = list(dict.fromkeys(c for c in (covariates or [])
                                    if c != outcome and c not in predictors))
    if not predictors:
        raise ModelingError("请至少选择一个目标自变量")

    cols = [outcome] + predictors + covariates
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ModelingError("字段不存在：" + "、".join(missing))
    data = df[cols].dropna()
    n_used = int(len(data))
    if n_used < 10:
        raise ModelingError(f"完整病例仅 {n_used} 例，无法进行多因素分析（建议 ≥10 且满足 EPV≥10）")

    # 任务判定
    y_raw = data[outcome]
    outcome_kind = variable_types.get(outcome)
    if task == "auto":
        if outcome_kind == "categorical":
            task = "logistic"
        elif outcome_kind == "numeric":
            task = "linear"
        else:
            task = "logistic" if y_raw.nunique(dropna=True) == 2 else "linear"

    if task == "logistic":
        levels = sorted(pd.unique(y_raw.astype(str)))
        if len(levels) != 2:
            raise ModelingError(f"Logistic 回归要求二分类结局，当前结局「{outcome}」有 {len(levels)} 个水平："
                                f"{'、'.join(levels)}。建议改用线性回归或先合并类别。")
        reference, event_level = _pick_event_level(levels)
        y = (y_raw.astype(str) == event_level).astype(int).to_numpy()
        n_events = int(y.sum())
        outcome_info = {"kind": "categorical", "levels": levels,
                        "reference": reference, "event_level": event_level}
    else:
        y = pd.to_numeric(y_raw, errors="coerce").astype(float).to_numpy()
        n_events = None
        outcome_info = {"kind": "numeric", "levels": None, "reference": None, "event_level": None}
        if np.isnan(y).any():
            raise ModelingError(f"连续结局「{outcome}」存在无法转换为数值的取值")

    # 设计矩阵
    blocks, coding = [], []
    for var in predictors + covariates:
        kind = variable_types.get(var, "numeric")
        block, info = _design_block(data, var, kind)
        blocks.append(block)
        coding.append(info)
    X = pd.concat(blocks, axis=1)
    dup = X.columns[X.columns.duplicated()].tolist()
    if dup:
        raise ModelingError(f"设计矩阵存在重名列：{'、'.join(map(str, dup))}。"
                            "通常是分类变量的类别名与另一变量名冲突，请调整变量选择。")

    _check_design_rank(X, coding)
    res, Xc = _fit(X, y, task)
    adjusted = _term_rows(res, Xc, coding, task)

    # 单变量（校正前）结果：同一完整病例子集下逐个变量单独拟合，便于与校正后对照
    crude_map = {}
    for info in coding:
        b, _ = _design_block(data, info["variable"], info["kind"])
        try:
            r2, Xc2 = _fit(b, y, task)
        except ModelingError:
            continue
        for row in _term_rows(r2, Xc2, [info], task):
            crude_map[row["term"]] = row

    for row in adjusted:
        c = crude_map.get(row["term"])
        row["crude_estimate"] = c["estimate"] if c else None
        row["crude_ci"] = c["ci"] if c else None
        row["crude_p"] = c["p"] if c else None
        row["crude_significant"] = bool(c["significant"]) if c else None

    # VIF（逐设计列计算，映射到原始变量）
    vif_rows, vif_warnings = [], []
    try:
        Xv = Xc.to_numpy(dtype=float)
        for idx, col in enumerate(Xc.columns):
            if col == "const":
                continue
            try:
                v = float(variance_inflation_factor(Xv, idx))
            except Exception:
                v = float("nan")
            owner = next((c for c in coding if col in c["columns"]), None)
            vif_rows.append({"term": col, "variable": owner["variable"] if owner else col,
                             "vif": None if np.isnan(v) else round(v, 3)})
        high = [r for r in vif_rows if r["vif"] and r["vif"] > VIF_THRESHOLD]
        if high:
            names = "、".join(sorted({r["variable"] for r in high}))
            vif_warnings.append(f"变量 {names} 的 VIF>{VIF_THRESHOLD}，提示存在多重共线性风险，"
                                f"建议剔除或合并相关变量后重新拟合。")
    except Exception as exc:
        vif_warnings.append(f"VIF 计算失败：{exc}")

    # 混杂提示：校正前后显著性不一致，或估计值变化超过 10%
    confounding = []
    by_var = {}
    for row in adjusted:
        by_var.setdefault(row["variable"], []).append(row)
    for var, rows in by_var.items():
        crude_sig = any(r.get("crude_significant") for r in rows)
        adj_sig = any(r["significant"] for r in rows)
        changed = []
        for r in rows:
            ce, ae = r.get("crude_estimate"), r.get("estimate")
            if ce is None or ae is None:
                continue
            # 以对数尺度（OR）或原始尺度（回归系数）衡量估计值变化幅度
            if task == "logistic":
                if ce <= 0 or ae <= 0:
                    continue
                rel = abs(np.log(ae) - np.log(ce)) / max(abs(np.log(ce)), 1e-9)
            else:
                if abs(ce) < 1e-9:
                    continue
                rel = abs(ae - ce) / abs(ce)
            if rel > 0.10:
                changed.append(r["term"])
        if crude_sig != adj_sig:
            direction = "校正后由显著变为不显著" if crude_sig else "校正后由不显著变为显著"
            confounding.append({
                "variable": var, "type": "significance_changed",
                "message": f"「{var}」{direction}，提示校正变量可能是混杂因素或存在过度校正，"
                           f"建议结合临床意义与样本量综合判断。",
                "terms": changed or [r["term"] for r in rows],
            })
        elif changed:
            confounding.append({
                "variable": var, "type": "estimate_changed",
                "message": f"「{var}」校正前后估计值变化超过 10%（{'、'.join(changed)}），"
                           f"提示校正变量对其存在一定影响，建议报告校正后结果。",
                "terms": changed,
            })

    warnings = list(vif_warnings)
    if not getattr(res, "converged", True):
        warnings.append("模型迭代未收敛（最大似然优化失败），参数估计不可靠，"
                        "提示可能存在完全分离、样本量不足或变量尺度问题。")
    n_terms = X.shape[1]
    epv = None
    if task == "logistic":
        epv = n_events / n_terms if n_terms else None
        if epv is not None and epv < MIN_EPV:
            warnings.append(f"事件数/自变量数（EPV）={epv:.1f} < {MIN_EPV}，模型存在过拟合风险，"
                            f"建议减少自变量或增加样本量。")
    if n_used < 100:
        warnings.append(f"完整病例样本量仅 {n_used} 例，多因素模型结果不稳定，仅供参考。")
    if task == "logistic" and min(n_events, n_used - n_events) < 10:
        warnings.append("结局事件数过少（任一类别 <10），Logistic 回归估计可能偏倚。")

    return {
        "task": task,
        "outcome": outcome,
        "outcome_info": outcome_info,
        "predictors": predictors,
        "covariates": covariates,
        "coding": coding,
        "n_used": n_used,
        "n_events": n_events,
        "n_terms": int(n_terms),
        "epv": round(epv, 2) if epv is not None else None,
        "terms": adjusted,
        "vif": vif_rows,
        "confounding": confounding,
        "warnings": warnings,
        "alpha": ALPHA,
        "model_summary": {
            "log_likelihood": _r(getattr(res, "llf", np.nan), 3),
            "aic": _r(getattr(res, "aic", np.nan), 3),
            "bic": _r(getattr(res, "bic", np.nan), 3),
            "r_squared": _r(getattr(res, "rsquared", np.nan), 4) if task == "linear" else None,
            "adj_r_squared": _r(getattr(res, "rsquared_adj", np.nan), 4) if task == "linear" else None,
            "pseudo_r_squared": _r(getattr(res, "prsquared", np.nan), 4) if task == "logistic" else None,
            "converged": bool(getattr(res, "converged", True)),
        },
        "note": "分类自变量采用哑变量编码（drop_first），参照组见 coding.reference；"
                "校正前结果基于与多因素模型相同的完整病例子集。",
    }


def screen_univariate(df: pd.DataFrame, outcome: str, candidates: list,
                      variable_types: dict) -> dict:
    """单变量筛选：逐候选变量与结局做组间比较，并给出 FDR 校正后 p 值。

    结论以校正后 p 值为准；本结果仅用于描述性报告，若用于特征选择必须在交叉验证内部完成。

    结局变量护栏：组间比较要求结局是水平数有限的分类变量。连续变量被误选为结局时
    （每个取值近似自成一组），Kruskal-Wallis 的 η² 会必然趋近 1、p 值退化为随机数，
    因此直接抛 ModelingError（接口层转 422），绝不静默输出垃圾结果。
    """
    if outcome not in df.columns:
        raise ModelingError(f"结局变量「{outcome}」不存在，无法进行单变量筛选")
    outcome_type = variable_types.get(outcome)
    n_levels = int(df[outcome].dropna().nunique())
    if outcome_type == "numeric":
        raise ModelingError(
            f"结局变量「{outcome}」是数值型变量（共 {n_levels} 个不同取值），"
            f"不能作为分组进行组间比较筛选。请选择分类变量（如诊断、分组、是否发生事件）作为结局；"
            f"连续结局请改用多因素线性回归或机器学习回归。")
    if outcome_type != "categorical":
        raise ModelingError(
            f"结局变量「{outcome}」的类型为「{outcome_type or '未确认'}」，"
            f"不支持组间比较筛选，请选择分类变量作为结局。")
    if n_levels < 2:
        raise ModelingError(f"结局变量「{outcome}」有效水平不足 2 个，无法进行组间比较")
    n_valid_outcome = int(df[outcome].notna().sum())
    if n_valid_outcome / n_levels < MIN_AVG_GROUP_SIZE:
        raise ModelingError(
            f"结局变量「{outcome}」有 {n_levels} 个不同取值、"
            f"平均每组样本量不足 {MIN_AVG_GROUP_SIZE}，疑似连续变量被误用为结局："
            "组数接近样本量时 η² 必然趋近 1、p 值退化。"
            "请选择分类变量（如诊断、分组）作为结局；连续结局请改用回归类分析。"
            "若确为多中心/多分级结局且每组样本充足，请检查数据。")

    rows, pvals = [], []
    for var in candidates:
        if var == outcome or var not in df.columns:
            continue
        vtype = variable_types.get(var, "numeric")
        if vtype not in ("numeric", "categorical"):
            continue
        try:
            r = compare_groups(df, target=var, group=outcome, target_type=vtype)
        except Exception as exc:
            rows.append({"variable": var, "error": str(exc)})
            continue
        if not r.get("valid"):
            rows.append({"variable": var, "error": r.get("note", "无法比较")})
            continue
        p = r["test"]["p"]
        if p is None:
            # 检验退化（如数据全缺失）导致 p 无法计算：跳过该行并说明原因，
            # 不得把 None 混入 FDR 校正
            rows.append({"variable": var, "error": "检验 p 值无法计算（数据退化或全缺失），已跳过"})
            continue
        pvals.append(p)
        rows.append({
            "variable": var,
            "variable_type": vtype,
            "test": r["test"]["name"],
            "statistic": r["test"]["statistic"],
            "p": p,
            "n": r.get("n_total"),
            "effect_name": r.get("effect", {}).get("name"),
            "effect_value": r.get("effect", {}).get("value"),
            "estimate": r.get("estimate"),
            "warnings": r.get("warnings", []),
        })

    corrected = fdr_correct(pvals)
    idx = 0
    for row in rows:
        if "p" in row:
            row["p_fdr"] = corrected["p_fdr"][idx]
            row["significant_raw"] = bool(row["p"] < ALPHA)
            row["significant_fdr"] = bool(corrected["significant"][idx])
            idx += 1
    valid_rows = [r for r in rows if "p" in r]
    valid_rows.sort(key=lambda x: (x["p_fdr"] is None, x["p_fdr"] if x["p_fdr"] is not None else 1))
    return {
        "outcome": outcome,
        "rows": valid_rows,
        "errors": [r for r in rows if "error" in r],
        "n_tests": len(valid_rows),
        "alpha": ALPHA,
        "fdr_method": "Benjamini-Hochberg",
        "note": "结论以 FDR 校正后 p 值为准；单变量筛选结果仅用于描述性报告，"
                "若用于特征选择必须在交叉验证内部完成。",
    }
