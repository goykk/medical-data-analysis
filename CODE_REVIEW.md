# 医疗数据分析平台 — 后端模型代码审查报告

审查时间：2026-10-21
审查范围：`backend/` 目录全部 Python 模块
审查重点：统计方法误用、数据泄漏、训练/验证集混淆、API 参数语义错误、数值稳定性、异常吞没

> **状态更新（2026-10-29，处理完毕）**：以下问题已逐条复核并落地，测试由 203 项增至 322 项。
>
> | 编号 | 结论 |
> |---|---|
> | P0-1 提前 dropna | 已修：只剔除结局缺失，特征缺失交给 Pipeline 内 imputer |
> | P0-2 多分类正类概率 | 已修：改为 one-vs-rest，每类一条 ROC / 校准曲线，DCA 取宏平均 |
> | P0-3 isotonic 小样本 | 已修：训练集 n≥1000 用 isotonic，否则 sigmoid |
> | P1-4 Cohen's d | **误报**：`std_meandiff_pooledvar` 实为均值差的标准误，原换算正确；已补手工核对测试 |
> | P1-5 VIF 含截距 | **误报**：const 在/不在入参结果一致，且与含截距的 1/(1−R²) 相同；已补测试锁定 |
> | P1-6 XGBoost 多分类不平衡 | 已修：改传平衡 `sample_weight`（xgboost 3.4.1 无 class_weight） |
> | P1-7 泄露检测 NaN | 已修：结局水平数先 dropna；类别缺失还原为 NaN |
> | P1-8 数值结局分级 | 已修：含小数→回归；整数≤5 级→分类；6–20 级需每级≥10 例，否则回归，并给出序数提醒 |
> | P2-9 SHAP 异常吞没 | 已修：记录堆栈，回传 `error_type` / `hint`，前端展示 |
> | P2-10 卡方期望频数 | 复核通过：2×2 且期望<5 已自动切 Fisher |
> | P2-11 ID 误判 | 复核通过：唯一值占比 + 名称/整数双重判定 |
> | P2-12 冠军 SHAP 全量 | 复核通过：超过 1000 行随机采样 |
> | P2-13 逐折混淆矩阵 | 已新增：`confusion_matrices` / `confusion_matrix_mean`，看板与智能建模均展示 |
> | P2-14 HL 分组硬编码 | 已修：分组数随样本量自适应（3/5/10/20） |
> | P2-15 FDR NaN 索引错位 | 复核通过：仅有效行参与校正与回写 |


---

## P0 — 严重错误（可能导致错误结论或崩溃）

### 1. `ml_engine.py:269` — 提前 dropna 导致 pipeline 的 imputer 失效

**问题**：训练前 `data = df[features + [target]].dropna()` 删除所有含缺失值的行，之后再进入 pipeline。pipeline 中的 `SimpleImputer` 完全用不上（没有 NaN 可填）。

**影响**：
- 对含缺失的数据集（医疗数据常见），会丢失大量样本
- 与代码注释 "pipeline 负责缺失处理" 的设计意图矛盾
- 若用户期望 "用中位数/众数填充缺失"，实际行为是 "丢弃含缺失的行"

**修复建议**：删除 line 269 的 `dropna()`，让 pipeline 的 imputer 自行处理缺失值。若需要排除完全缺失行，应在数据确认页由用户显式决定。

---

### 2. `ml_engine.py:378` — 多分类正类概率取值写死 `[:, 1]`

**问题**：
```python
prob_pos = y_prob_best[:, 1] if len(classes) == 2 else y_prob_best[:, 1]
```
两个分支完全相同。多分类时 `y_prob` 形状为 `(n_samples, n_classes)`，`[:, 1]` 取第二列（对应 `classes[1]`）作为"正类"概率没有统计学意义。

**影响**：多分类任务的 ROC 曲线、校准曲线、DCA 全部基于 `classes[1]` 的概率计算，结果无意义（其他类的概率被忽略）。

**修复建议**：
- 二分类：保持 `[:, 1]`（对应 `classes[1]`，即正类）
- 多分类：改用 One-vs-Rest，对每个 class 单独计算 ROC/校准/DCA；或明确告知用户 "多分类不支持 ROC/DCA"

---

### 3. `ml_engine.py:375-393` — Isotonic 校准对小样本易过拟合

**问题**：`CalibratedClassifierCV(estimator=pipe, method="isotonic", cv=5)` 使用 isotonic regression 做概率校准。Isotonic 是分段线性插值，在小样本（n < 1000）时极易过拟合训练数据，导致校准曲线在训练集上完美但在新样本上失效。

**影响**：医疗数据集通常样本量 <1000，isotonic 校准给出的概率估计不可靠，影响下游 DCA（决策曲线分析）和阈值选择。

**修复建议**：
- `n < 500` 时用 `method="sigmoid"`（Platt scaling，逻辑回归校准，泛化更好）
- `n >= 500` 时用 `method="isotonic"`
- 在报告中注明校准方法和样本量

---

## P1 — 逻辑错误 / 边界条件

### 4. `analyzer.py:495-499` — Cohen's d 计算公式错误

**问题**：
```python
se = cm.std_meandiff_pooledvar
sp = float(se) / np.sqrt(1.0 / len(a) + 1.0 / len(b))
cohen_d = diff / sp if sp > 0 else None
```
`cm.std_meandiff_pooledvar` 返回的是**合并方差**（pooled variance, $s_p^2$），不是合并标准差（$s_p$）。当前代码把方差当标准差用，再除以 $\sqrt{1/n_1 + 1/n_2}$，量纲错误。

**正确公式**：
$$s_p = \sqrt{s_p^2}, \quad d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$$

**修复建议**：
```python
pooled_var = cm.std_meandiff_pooledvar
sp = float(np.sqrt(pooled_var))
cohen_d = diff / sp if sp > 0 else None
```

---

### 5. `multivariable.py:199-210` — VIF 计算包含截距列

**问题**：
```python
for idx, col in enumerate(Xc.columns):
    if col == "const":
        continue
    v = variance_inflation_factor(Xc, idx)
```
`variance_inflation_factor(Xc, idx)` 在回归中**包含 `const` 列**作为自变量之一。VIF 的定义是 $1/(1 - R^2_j)$，其中 $R^2_j$ 是第 $j$ 个变量对其他变量的回归决定系数。若回归中包含截距，$R^2_j$ 会被截距"吸收"一部分方差，导致 VIF 偏小，漏报多重共线性。

**修复建议**：
```python
Xv = Xc.drop(columns=["const"])
for idx, col in enumerate(Xv.columns):
    v = variance_inflation_factor(Xv, idx)
```

---

### 6. `racer.py:133-134` — XGBoost `scale_pos_weight` 对多分类无效

**问题**：
```python
elif key == "xgboost":
    est.set_params(scale_pos_weight=scale)
```
XGBoost 的 `scale_pos_weight` 参数**只对二分类生效**（通过 `objective="binary:logistic"`）。多分类时（`objective="multi:softprob"`）该参数被忽略，少数类样本得不到加权。

**影响**：多分类任务中少数类被严重低估，模型倾向于预测多数类。

**修复建议**：
- 二分类：保持 `scale_pos_weight`
- 多分类：改用 `class_weight="balanced"`（XGBoost >= 1.6 支持）或手动设置 `sample_weight`

---

### 7. `leakage.py:134` — 二分类标签含 NaN 时误走 NMI 分支

**问题**：
```python
if int(pd.Series(y).nunique()) <= 2:
    score = _auc_binary(y, feat)
```
若 `y` 含 NaN（例如结局列有缺失），`nunique()` 返回 3（0, 1, NaN），走 NMI 分支。NMI 对二分类标签的判别力弱于 AUC，且 NaN 参与 NMI 计算会引入噪声。

**修复建议**：
```python
y_clean = y[~np.isnan(y)]
if int(pd.Series(y_clean).nunique()) <= 2:
    score = _auc_binary(y, feat)  # _auc_binary 内部已排除 NaN
```

---

### 8. `profiler.py:90` — 数值型结局 n_unique ≤ 20 一律视为分类

**问题**：
```python
if pd.api.types.is_numeric_dtype(s):
    return int(s.nunique(dropna=True)) <= MAX_CLASS_LEVELS
```
若数值型结局只有 20 个以下不同值（如 1-15 分的疼痛量表、0-10 的 NRS 评分），会被误判为分类任务，走 Logistic 回归而非线性回归。

**影响**：量表类数据被当作分类处理，丢失"分数高低"的序数信息，模型效能下降。

**修复建议**：
- 增加启发式：若取值是连续整数（如 1, 2, 3, ..., 15），视为序数/连续
- 或在推断后提示用户确认："结局变量有 15 个不同整数值，按分类处理是否正确？"

---

## P2 — 风格 / 小问题

### 9. `ml_engine.py:398-401` — SHAP 计算失败静默吞异常

**问题**：
```python
try:
    payload["shap"] = shap_global(pipe, X_test, task, numeric, categorical)
except Exception as exc:
    payload["shap"] = {"error": f"SHAP 计算失败：{exc}"}
```
SHAP 计算失败时只返回一个错误字符串，用户在前端看不到具体原因（前端只检查 `shap` 是否存在）。

**修复建议**：
- 在 payload 中增加 `shap_error_detail` 字段，包含异常类型和堆栈
- 前端展示错误提示，并建议 "样本量 <100 时 SHAP 可能超时，可跳过"

---

### 10. `analyzer.py:577` — 卡方检验期望频数 <5 时只警告不自动切换

**问题**：
```python
note = ""
if min_expected < 5:
    note = (f"存在期望频数 <5 的格子（最小={min_expected:.2f}），卡方结果不稳定；...")
```
对小样本数据（n < 40 或期望频数 <5），卡方检验结果不可靠，应自动切换 Fisher 精确检验。当前代码只输出警告文本，仍返回卡方结果。

**修复建议**：
- `min_expected < 5` 且 `shape == (2, 2)`：自动用 `scipy.stats.fisher_exact`
- 其他情况：保持卡方但增加醒目提示 "建议使用 Fisher 精确检验"

---

### 11. `validators.py:231-243` — `_is_id_like` 对高唯一性数值列误判

**问题**：
```python
def _is_id_like(s: pd.Series, name: str) -> bool:
    if _ID_PATTERN.search(str(name)):
        return True
    vals = s.dropna()
    if vals.empty or not pd.api.types.is_numeric_dtype(vals):
        return False
    arr = vals.to_numpy(dtype=float)
    return bool(np.all(np.isclose(arr, np.round(arr))))
```
对 WDBC 类数据集，连续测量值（如 `perimeter_mean`）若取值近乎唯一（所有浮点数都是整数倍），会被误判为 ID 列并排除。

**修复建议**：
- 增加方差/标准差判断：若 `std > 0.1`，不太可能是 ID
- 或增加取值分布判断：ID 列通常取值是离散整数且无明显分布规律

---

### 12. `champion.py:193-199` — SHAP 在全量数据上计算

**问题**：
```python
Xt = prep.transform(X)  # X 是完整数据集
...
shap = _shap_top(est, Xt, mapping, top_k) if with_shap else None
```
SHAP 在全量数据上计算，对大数据集（n > 5000）耗时极长（TreeExplainer 复杂度 O(n × tree_size)）。

**修复建议**：
- 支持 `subsample` 参数，默认 `min(n, 1000)`
- 在报告中注明 "SHAP 基于前 1000 个样本计算"

---

### 13. `racer.py:215-312` — 赛马时未记录每折的混淆矩阵

**问题**：赛马只记录 AUC/accuracy/RMSE 等聚合指标，不记录每折的混淆矩阵。对于医疗数据，用户需要看 "假阴性多少、假阳性多少" 来判断模型是否可用。

**修复建议**：在 `RaceRow.metrics` 中增加 `confusion_matrices` 字段（每折一个 2×2 矩阵），前端可展示 "平均混淆矩阵"。

---

### 14. `ml_engine.py:161-198` — Hosmer-Lemeshow 检验分组数硬编码 10

**问题**：
```python
def hosmer_lemeshow(y_true, y_prob, groups: int = 10) -> dict:
```
HL 检验的分组数 `groups` 默认 10，对小样本（n < 100）会导致每组样本量 <10，检验效能极低。文献建议：n < 100 时 groups = 5，n > 500 时 groups = 20。

**修复建议**：
```python
groups = max(5, min(20, int(len(y_true) / 10)))
```

---

### 15. `analyzer.py:305-393` — 单变量筛选 FDR 校正时 p=NaN 被忽略

**问题**：
```python
pvals = [r["p"] for r in rows if r.get("p") is not None]
corrected = multipletests(pvals, alpha=0.05, method="fdr_bh")
```
若某些变量的 p 值为 NaN（例如数据全缺失），这些变量被排除在 FDR 校正之外，但校正后的 p_fdr 仍被写回原 rows，索引可能错位。

**修复建议**：
- 记录哪些行被排除
- 校正后只对被校正的行写回 p_fdr，被排除的行标记 `p_fdr = None` 并注明原因

---

## 总结

| 优先级 | 数量 | 典型问题 |
|--------|------|----------|
| P0     | 3    | 提前 dropna、多分类 ROC 错误、isotonic 过拟合 |
| P1     | 5    | Cohen's d 公式错、VIF 含截距、XGBoost 多分类失效 |
| P2     | 7    | SHAP 异常吞没、卡方不自动切 Fisher、HL 分组硬编码 |

**建议修复顺序**：P0 → P1 → P2。P0 问题直接影响模型结论的正确性，应优先修复并补单元测试。P1 问题影响边界场景的可靠性，建议在下一个迭代中修复。P2 问题影响用户体验和可解释性，可排入 backlog。

**测试建议**：
- 对 P0 问题补单元测试：构造含缺失的数据集，验证 pipeline 是否正确 impute；构造多分类数据，验证 ROC 是否为 OvR
- 对 P1 问题补单元测试：构造已知 Cohen's d 的两样本，验证计算结果；构造已知共线性的设计矩阵，验证 VIF
