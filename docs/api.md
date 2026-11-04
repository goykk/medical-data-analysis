# medical-data-analysis/docs/api.md
# 后端 REST 接口清单（FastAPI，默认 http://127.0.0.1:8002）

统一约定：
- 请求 / 响应均为 `application/json`（上传接口为 `multipart/form-data`）；
- CORS 仅允许 `FRONTEND_ORIGIN` 中配置的前端来源，**不使用 `*`**；
- 成功直接返回业务数据；失败返回 `{code, message, detail}`，前端统一读 `message` 展示；
- 所有统计与建模数值均由后端 scipy / sklearn / statsmodels 计算，前端不做任何计算；
- 若配置了环境变量 `API_TOKEN`，除 `GET /api/health` 外所有 `/api/*` 必须携带
  `X-API-Token: <token>`（或 `Authorization: Bearer <token>`），否则返回 401；
- 重计算接口有进程内滑动窗口限流：`/api/dataset/upload`、`/api/sample/load` 20 次/分钟，
  `/api/ml/train`、`/api/automl/analyze`、`/api/automl/train` 6 次/分钟，`/api/llm/*` 10 次/分钟，超限 429。

---

## 一、基础

### `GET /api/health`
健康检查。
```json
{"status":"ok","mode":"local","version":"0.1.0","datasets_in_memory":1}
```

### `GET /api/config`
返回运行模式、上传限制、允许扩展名、前端来源，供前端初始化界面文案。
```json
{"mode":"local","max_upload_mb":50,
 "allowed_extensions":[".csv",".dta",".json",".sav",".tsv",".txt",".xls",".xlsx"],
 "frontend_origin":"http://127.0.0.1:3001"}
```

### `GET /api/reference-ranges`
返回集中定义的医学参考区间常量（含依据说明）。

---

## 二、数据集

### `POST /api/dataset/upload`（multipart，字段名 `file`）
上传并解析文件。**上传新数据集后，内存中旧模型立即失效。**

响应：
```json
{
  "dataset_id":"3f2c9a1b7e44",
  "name":"medical_sample.csv",
  "n_rows":302, "n_cols":12,
  "parse_meta":{"format":"csv","encoding":"gbk"},
  "validation":{
     "is_medical":true, "match_score":0.67, "matched_fields":8, "total_fields":12,
     "rule":"normal", "strong_clinical_fields":[],
     "message":"…", "options":["terminate","continue"]
  },
  "variables":[
     {"name":"空腹血糖","inferred_type":"numeric","confirmed_type":"numeric",
      "n_unique":286,"missing_rate":0.0497,"sample_values":["5.62","7.10","6.33"],
      "excluded":false,"exclude_reason":null,
      "reference_range":{"low":3.9,"high":6.1,"unit":"mmol/L","label":"空腹血糖","source":"…"}}
  ]
}
```

错误：`400` 解析失败（detail 含具体原因，ARFF/影像/信号矩阵会明确提示先转表格）、
`413` 超过大小限制。

- `validation` 额外返回 `matched_terms`（命中的医学术语）、`name_hint`（文件名中的医学提示词）与
  `rule`（normal / strong_clinical / filename）；字段命中不足时，文件名含医学词可作为补充证据。

- `parse_meta`：`format` / `encoding`；检测到坏行时附 `skipped_rows` 与 `note`；
  检测到**无表头文件**时附 `headerless=true` 与说明，列名按 `col_1…col_N` 命名
  （不再把首行数据当列名，如 `1.0000000e+000`、`+`、DNA 序列）。
- `POST /api/dataset/{id}/outcome-candidates`：按结局词库 + 二值加分排序，最多返回 20 条；
  性别/种族等人口学列、纯数字/符号列名（无表头残留）不会被自动推荐为结局，需用户手动选择。

### `GET /api/dataset/{dataset_id}`
返回数据集元信息、字段清单、清洗记录与备注。

### `POST /api/dataset/{dataset_id}/confirm`
确认字段类型（分析前置步骤）。
```json
{"types":{"性别":"numeric","分组":"categorical"}, "force_continue":false}
```
- 医疗校验未通过且 `force_continue=false` → `422`，前端弹出「终止分析 / 确认继续」；
- 手动放行会在数据集 `notes` 中记录，报告页据此注明。

### `GET /api/dataset/{dataset_id}/cleaning-report`
清洗报告：重复行、缺失值、参考区间 / IQR 异常值、已排除字段。

### `POST /api/dataset/{dataset_id}/clean`
```json
{"drop_duplicates":true,"drop_columns":[],"drop_rows_missing_in":[],
 "outlier_action":"keep"}        // keep | drop_rows
```
响应：`{n_rows, variables, cleaning{n_before,n_after,drop_rate,warning?,actions[]}}`。
- `actions[]` 记录每一步删除行数与占比（如“删除 140 行，剩余 3 行（占 97.9%）”）；
- 总删除比例 ≥20% 时返回 `warning`，提示核对去重/缺失剔除规则，避免静默删数据。

### `GET /api/dataset/{dataset_id}/outcome-candidates`
结局变量智能识别候选（按得分降序，最多 20 条）。性别/种族等人口学列与无表头产生的
数字/符号列名不会被自动推荐。

---

## 三、统计分析

### `POST /api/analysis/describe`
```json
{"dataset_id":"…","fields":null}   // fields 为空表示全部有效变量
```
返回 `numeric[]`（n、均值、标准差、中位数、四分位数、极值、均值 95%CI、偏度、峰度）
与 `categorical[]`（频数、构成比）。

### `POST /api/analysis/normality`
```json
{"dataset_id":"…","fields":["年龄","空腹血糖"]}
```
Shapiro-Wilk；`n>500` 时 `reference_only=true` 并提示结合直方图 / Q-Q 图。

### `POST /api/analysis/univariate`
```json
{"dataset_id":"…","target":"空腹血糖","group":"分组","target_type":"numeric"}
```
自动选检验：正态 + 方差齐 → Student t；方差不齐 → **Welch 校正**；非正态 → Mann-Whitney U；
多组 → ANOVA / **Alexander-Govern**（方差不齐）/ Kruskal-Wallis；分类结局 → 卡方，
任一格期望频数 <5 → **Fisher 精确检验**。

返回：`test{name, statistic, p, assumption_checks[], note}`、
`effect{name, value, interpretation}`、`estimate{label, value, ci}`、`group_summary[]`、`warnings[]`。

### `POST /api/analysis/correlation`
```json
{"dataset_id":"…","fields":["年龄","体质指数","收缩压"]}
```
返回相关系数矩阵 `r`、原始 `p`、**FDR 校正后 `p_fdr`**、每对样本量 `n` 与所用方法 `method`。

### `POST /api/analysis/multivariable`
```json
{"dataset_id":"…","outcome":"结局",
 "predictors":["空腹血糖","吸烟"],"covariates":["年龄","性别"],"task":"auto"}
```
- 分类结局 → Logistic（OR），连续结局 → 线性回归（β）；
- 返回 `coding`（含参照组）、`terms[]`（校正前 / 校正后估计值、95%CI、p）、
  `vif[]`、`confounding[]`（混杂提示）、`warnings[]`（VIF>10、EPV<10、样本量、未收敛）。

### `POST /api/analysis/screen`
```json
{"dataset_id":"…","outcome":"结局","candidates":["年龄","空腹血糖","吸烟"]}
```
单变量筛选，同时给出原始 p 与 **FDR（Benjamini-Hochberg）** 校正 p，结论以校正后为准。

### `POST /api/analysis/histogram` / `boxplot` / `qqplot`
```json
{"dataset_id":"…","field":"年龄","group":"分组"}
```
返回 ECharts 可直接渲染的图表数据（含参考区间线、异常点、理论分位数）。

---

## 四、机器学习

### `POST /api/ml/train`
```json
{"dataset_id":"…","features":["radius_mean","texture_mean",…],
 "target":"diagnosis","task":"classification","with_shap":true}
```
返回：
- `models[]`：逻辑回归 / 随机森林，含交叉验证 `均值±标准差`、测试集指标与**测试集混淆矩阵**
  （`test_metrics.confusion_matrix`，行=真实、列=预测，标签顺序同 `classes`）；
- `best_model{name,label,reason}`：性能差距 ≤0.02 时优先可解释模型；
- 分类任务：`roc`（AUC、约登最佳截断值）、`calibration`（校准曲线 + Hosmer-Lemeshow）、
  `dca`（净获益 + 全部干预 / 全不干预参考线），**全部基于测试集**且注明 `n=测试集样本量`；
  - 校准方法：训练集 `n≥1000` 用 isotonic，否则用 Platt(sigmoid)；校准器直接用同一套
    交叉验证折产生的 OOF 概率拟合（`OOF校准（…，复用 N 折 CV）`），不再额外拟合
    `CalibratedClassifierCV` 的一整套折；多分类按 one-vs-rest 逐类校准后归一化；
  - HL 分组数按测试集样本量自适应（n<50→3，50–99→5，100–500→10，>500→20），见 `calibration.hosmer_lemeshow.groups`；
  - **多分类**：按 one-vs-rest 输出，`roc.multi=true`、`roc.classes[]` 每类一条曲线，
    `roc.macro_auc` 为宏平均 AUC；`calibration.classes[]` 同理；`dca` 为各类净获益的宏平均
    （单类结果见 `dca.classes`）；`calibration.hosmer_lemeshow` 仅二分类存在。
- `shap`：全局蜂群图数据（`TreeExplainer` / `LinearExplainer`，哑变量按原变量聚合，超 1000 点采样）；
  失败时返回 `{error, error_type, hint}`，不影响其他指标；
- `warnings[]`：样本量 <50、EPV<10、测试集 AUC≥0.995（提示排查泄露/过拟合）等；
  严重不足时返回 `skipped=true` 并说明原因。
- `speed{fast_mode,cv_folds,estimator,rf_trees,train_rows,sampled}`：规模自适应实际采用的训练口径；
  训练集 >2 万行进入快速模式：3 折交叉验证，并用 HistGradientBoosting 替代随机森林
  （150 棵 → 直方图提升，实测 15 万行训练 29s → 9s）；HGB 无原生 importance/SHAP，
  特征重要性改用测试集置换重要性，SHAP 返回明确的降级提示；
- `leak_columns[]`：自动剔除的结构性泄露列（单变量判别力 >0.99，如结局副本、结局派生列）；
- `feature_screening{triggered,original,selected,dropped,dropped_count,method,top_scores}`：
  特征数 >50 时在**训练集**上做方差过滤 + 单变量 F 检验 + 共线性（>0.95）去冗，保留 top-40；
  测试集不参与打分，避免用测试标签做特征选择；
- `from_cache`：命中同一数据集版本/同一特征口径的训练缓存时为 `true`。

训练成功即自动 `save_model`（覆盖同数据集旧模型）。

### `POST /api/ml/train/async`（前端默认使用）
请求体同 `/api/ml/train`。大样本训练耗时长，页面改为后台执行 + 轮询，避免 HTTP 长连接超时。
```json
{"task_id":"…","status":"running"}
```

### `GET /api/tasks/{task_id}`
```json
{"task_id":"…","status":"running|done|error","progress":15,"stage":"训练中",
 "error":null,"code":null,"result":{…}}
```
- `status=done` 时 `result` 与同步 `/api/ml/train` 的响应结构完全一致（含 `saved`）；
- `status=error` 时 `error` 为可直接展示的原因，`code` 为 422/500；
- 任务不存在或超过 1 小时过期返回 404；
- 训练任务在进程内线程池执行（最多 2 并发），任务表上限 50 条。


### `GET /api/ml/model/{dataset_id}`
已建模返回模型信息（特征清单、数值字段训练集范围、类别字段取值、测试集指标）；
未建模返回 `{"trained":false,"message":"请先完成当前数据集的建模"}`（前端空状态）。

### `POST /api/ml/predict`
```json
{"dataset_id":"…","sample":{"radius_mean":17.99,"texture_mean":10.38, …}}
```
返回预测类别 / 数值、各类概率、`shap_force_plot`（matplotlib 静态图 base64 PNG）、
`warnings[]`（超出训练集范围 → 外推警示；未见类别 → 未知类别提示）、免责声明。

### `DELETE /api/ml/model/{dataset_id}`
使当前数据集模型失效。

---

## 五、示例数据与大模型

### `GET /api/sample/list`
列出内置示例数据（模拟医疗数据集、WDBC 乳腺癌数据集）与行列数。

### `POST /api/sample/load`
```json
{"key":"wdbc"}     // 或 "medical"
```
与上传接口返回结构一致。

### `GET /api/llm/status`
```json
{"provider":"modelarts_pangu","provider_label":"华为云 ModelArts Studio（盘古）",
 "base_url":"https://cn-north-4.modelarts-infer.com/api/v2","model":"pangu",
 "configured":false,"note":"密钥仅保存在服务端内存中，重启后失效，不会写入任何文件。"}
```

### `POST /api/llm/config`
```json
{"provider":"modelarts_pangu","base_url":"…","model":"pangu","api_key":"…"}
```
`api_key` 为空表示不修改已有密钥；响应**不回传密钥**。

### `POST /api/llm/test`
```json
{"provider":"modelarts_pangu","base_url":"…","model":"pangu","api_key":"…"}
```
成功：`{"connected":true,"latency_ms":312,"message":"连接成功（…）"}`；
失败区分 `not_configured` / `http_error`（密钥错误、模型名不存在）/ `network_error`。

---

## 六、分析历史（本地 SQLite / 云上 RDS）

### `POST /api/history`
保存一次分析记录（前端在分析完成后调用）。
```json
{"dataset_name":"wdbc.csv","n_rows":569,"outcome":"diagnosis",
 "payload":{"best_model":"logistic","auc":0.995}}
```

### `GET /api/history?limit=50`
按时间倒序返回历史记录；`limit` 被钳制在 1–500。
```json
{"mode":"local","items":[{"id":1,"created_at":"…","dataset_name":"wdbc.csv","n_rows":569,
  "outcome":"diagnosis","payload":{"…"}}]}
```

---

## 七、智能建模（`/api/automl/*`）

统一入参：`dataset_id`，可选 `target`（结局）、`time_col` / `event_col`（生存）、`patient_id_col`（分组键）。

### `GET /api/automl/models?task=classification`
模型清单（key、名称、适用任务、默认超参、SHAP 策略、依赖是否可用）。

### `POST /api/automl/standardize`
只做列名标准化（模块零），返回每列的 `raw_name → standard_name`、匹配方式与置信度。

### `POST /api/automl/prepare`
一次拿到画像 + 泄露检测 + 路由预览 + 待确认项（列名映射、泄露列），供前端一次交互确认。
```json
{"dataset_id":"…","target":"diagnosis"}
```

### `POST /api/automl/confirm`
提交列名映射与泄露列处置决定（`{"action":"drop|keep|ask","columns":[…]}`），
用户确认过的映射会写入用户级词典并持久生效。

### `POST /api/automl/analyze`（自动选最优模型，五步流水线）
```json
{"dataset_id":"…","target":"diagnosis","leakage_decision":"ask",
 "confirmed_drop":[],"mapping_decisions":{},"with_shap":true,"leakage_threshold":0.95}
```
`leakage_threshold` 被钳制在 0.5–0.999。返回结构化报告：
`profile`、`standardization`、`leakage`、`routing`、`leaderboard`、`champion`、`limitations`、`disclaimer`。

`leaderboard[]` 每行：
```json
{"model_key":"logistic","label":"逻辑回归","metric_name":"auc","metric_value":0.98,
 "ci_low":0.97,"ci_high":0.99,"fit_seconds":1.2,"status":"ok",
 "metrics":{"auprc":0.97,"accuracy":0.95,"classes":["B","M"],
            "confusion_matrices":[[[71,1],[2,40]] …],
            "confusion_matrix_mean":[[71.0,0.4],[2.0,40.4]]},
 "notes":["…"],"error_message":null}
```
- `confusion_matrices`：逐折混淆矩阵（行=真实、列=预测，标签顺序同 `metrics.classes`）；
- `confusion_matrix_mean`：各折矩阵均值，供看板/报告展示假阴性、假阳性；
- 单模型训练失败只标 `status:"failed"`，不中断整次分析。

### `POST /api/automl/train`（手动指定模型）
```json
{"dataset_id":"…","model_key":"xgboost","target":"diagnosis",
 "params":{"max_depth":4},"with_shap":true}
```
与自动模式共用赛马 / 冠军 / 报告流水线；`params` 走白名单过滤与区间钳制。
少数类占比 <10% 时自动启用类权重；XGBoost 多分类因不支持 `scale_pos_weight`，改传平衡样本权重。

### `GET /api/automl/lexicon/user`
返回用户级列名词典（`{原始列名: 标准名}`），由 `confirm` 写入。

---


## 错误码约定

| code | 含义 | 典型场景 |
|---|---|---|
| 400 | 请求无法处理 | 文件格式不支持、解析失败、文件为空 |
| 404 | 资源不存在 | dataset_id 失效、示例数据文件缺失 |
| 413 | 载荷过大 | 超过 `MAX_UPLOAD_MB` |
| 422 | 业务前提不满足 | 未确认医疗校验、结局非二分类、未建模就预测 |
| 500 | 服务器内部错误 | 未捕获异常（detail 返回异常信息） |
