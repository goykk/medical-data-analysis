# 医疗数据分析平台

> 面向临床科研的医疗数据分析工具：上传数据 → 医疗校验 → 变量确认 → 清洗 → 统计分析 →
> 多因素校正 → 机器学习建模与评估 → 新样本预测 → 报告导出。
>
> 第十一届华为 ICT 大赛创新赛（赛题 1：基于华为云产品打造 AI 创新应用）参赛作品。

**免责声明：本工具分析结果仅供科研参考，不构成临床诊断依据。**

---

## 一、功能清单

### P0（已实现）
| 功能 | 说明 |
|---|---|
| 文件解析 | csv / xlsx / xls / tsv / txt / json / SPSS(.sav) / Stata(.dta)；文本类自动检测编码（UTF-8 / GBK / GB2312），失败自动切换重试 |
| 医疗数据校验 | 中英文医学字段词库匹配 + 强临床指标规则；不足时给出「终止分析 / 确认继续」，手动放行在报告中注明 |
| 变量类型确认 | 自动推断并展示「字段名 + 类型」清单供确认；数字编码分类变量（性别 0/1、分级 1/2/3）按分类处理；自动排除 ID 列与常量列。ID 判定需同时满足「唯一值占比 >90%」与「字段名带编号特征或取值为整数」，带小数的连续测量值不会被误判为 ID |
| 数据清洗 | 重复行、缺失值、医学参考区间 / IQR 异常值报告；剔除与保留由用户确认并记录；缺失值按成对删除，每项分析注明实际纳入样本量 |
| 描述性统计 | 均值、标准差、中位数、四分位数、极值、均值 95%CI；分类变量频数与构成比 |
| 正态性检验 | Shapiro-Wilk；n>500 结果标记「仅作参考」，结合直方图 / Q-Q 图判断 |
| 单变量组间比较 | 按类型、组数、正态性、方差齐性自动选检验：Student t / **Welch 校正** / Mann-Whitney U / ANOVA / **Alexander-Govern** / Kruskal-Wallis / 卡方 / **Fisher 精确检验**；报告效应量（Cohen's d、η²、OR、Cramér's V）、95%CI、统计量、p 值 |
| 相关性分析 | 相关系数矩阵 + 热力图；正态用 Pearson，否则 Spearman；p 值经 **FDR（BH）** 校正 |
| 医学参考区间 | 16 项检验指标参考区间集中定义并注明依据，图形叠加参考线 |
| 多因素分析 | 分类结局 Logistic（调整 OR）、连续结局多元线性回归（β）；哑变量编码注明参照组；VIF>10 共线性提示；**校正前 vs 校正后**对照与混杂提示；EPV<10 与完全分离警示 |
| 可视化 | ECharts 直方图 / 箱线图 / 热力图 / 森林图 / 特征重要性图 / SHAP 蜂群图，每张支持下载 PNG |
| 报告导出 | 固定结构汇总，浏览器打印转 PDF |

### P1（已实现）
| 功能 | 说明 |
|---|---|
| 结局变量识别 | 词库扫描推断候选，用户确认或手动指定；未指定不强行训练 |
| 重要指标筛选 | 同时报告原始 p 与 FDR 校正 p，结论以校正后为准；报告按重要性降序，次要指标折叠到附录 |
| 机器学习建模 | sklearn Pipeline（填补 / 编码 / 标准化）；逻辑回归 + 随机森林（回归任务线性回归 + 随机森林回归）；分层划分 + 5 折交叉验证（均值±标准差）；`class_weight` 处理不平衡；每类召回率；固定随机种子 |
| 评估三件套 | ROC + AUC + 约登最佳截断值；校准曲线 + **Hosmer-Lemeshow 检验**（`CalibratedClassifierCV` 校准后，n≥1000 用 isotonic、小样本用 sigmoid）；**DCA** 净获益 + 全部干预 / 全不干预参考线；多分类按 one-vs-rest 每类一条曲线；全部基于测试集并注明 `n=测试集样本量` |
| 可解释性 SHAP | 树模型 `TreeExplainer`、线性模型 `LinearExplainer`；独热哑变量按原始变量聚合；超 1000 点后端采样；新样本力图用 matplotlib 生成 base64 图片 |
| 新样本预测 | joblib 保存最优 Pipeline / 特征清单 / 训练集数值范围；返回类别或数值、概率、SHAP 力图；超范围返回外推警示；上传新数据集后旧模型立即失效；pytest 验证可复现 |
| 样本量红线 | n<50 或 EPV<10 提示「结果仅供参考」；严重不足跳过建模，仅做统计描述与组间比较 |

### P2-参赛（已实现）
| 功能 | 说明 |
|---|---|
| 盘古大模型接入 | `llm_service.py`，华为云 ModelArts Studio（盘古）Base URL 预填 `https://{region}.modelarts-infer.com/api/v2`，鉴权头 `X-Apig-AppCode`；共用超时 / 重试 / 异常捕获 / 降级逻辑；「测试连接」返回延迟或失败原因；未配置返回「未配置」而非报错 |
| 密钥安全 | 读取优先级：页面配置 > 环境变量 > 未配置；**只存内存**，不写文件、不写日志 |
| 双模式 | `MODE=local` / `MODE=cloud`，业务代码共用（见 `backend/storage.py`）；云上模式页面与 README 明示「数据将上传至华为云，由使用者确认」 |
| 云服务关联 | RDS（历史）、Redis（任务状态 / 缓存）、OBS（数据集 / 模型 / 静态资源 + CDN）、ECS（后端） |
| 部署文档 | `deploy_huawei.md` 全步骤 + `deploy/ecs_bootstrap.sh` 一键脚本 |
| 开源整理 | 无密钥无真实数据，配置走环境变量，`.env.example` 提供本地与华为云两套模板 |

### P2-常规（规划中）
AI 对话助手、AI 报告解读（专业版 + 通俗版）、生存分析（Kaplan-Meier）、配对检验、方案复现、历史持久化页面。

---

## 二、架构

```
┌──────────────────────────────────────────────────────────────┐
│  浏览器（原生 JS + ECharts，不做任何统计计算）                 │
│  9 个页面：上传 / 确认 / 清洗 / 配置 / 智能建模 / 看板 /       │
│  预测 / 报告 / 设置                                            │
└───────────────┬──────────────────────────────────────────────┘
                │ REST（CORS 仅允许固定来源，禁止 "*"）
┌───────────────▼──────────────────────────────────────────────┐
│  Node.js + Express（端口 3001）静态托管                        │
└──────────────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│  Python + FastAPI（端口 8002）                                 │
│  ├ validators.py    医疗校验 / 变量类型推断 / 参考区间异常      │
│  ├ analyzer.py      解析 / 清洗 / 描述 / 正态性 / 组间比较      │
│  ├ multivariable.py Logistic / 线性回归 / VIF / 混杂提示        │
│  ├ ml_engine.py     Pipeline 建模 / 三件套 / SHAP / 预测        │
│  ├ automl/          智能建模：标准化→画像→泄露检测→路由→赛马   │
│  │                  →冠军分析                                   │
│  ├ llm_service.py   盘古等大模型接入（可降级）                  │
│  └ storage.py       双模式存储适配（SQLite|RDS、内存|Redis、    │
│                     本地目录|OBS）                              │
│  统计与建模一律调用 pandas / scipy / statsmodels / sklearn      │
└──────────────────────────────────────────────────────────────┘
```

**双模式差异仅体现在配置与 `storage.py` 适配层**，业务代码不维护两套逻辑：

| 能力 | 本地模式（默认） | 云上模式（MODE=cloud） |
|---|---|---|
| 分析历史 | SQLite 单文件 | 华为云 RDS（MySQL） |
| 任务状态 / 缓存 | 进程内存 | 华为云 DCS Redis |
| 数据集 / 模型产物 | 本地目录 | 华为云 OBS（静态资源配 CDN） |
| 后端运行 | 本机 uvicorn | 华为云 ECS |
| 数据去向 | **数据不出本机** | **数据将上传至华为云，由使用者确认**（页面顶部黄条 + README 明示） |

---

## 三、目录结构

```
medical-data-analysis/
├── frontend/          # Node.js：server.js、package.json、public/（页面与 JS 模块）
├── backend/           # Python：app.py、analyzer.py、ml_engine.py、validators.py、
│                      # multivariable.py、llm_service.py、prompts.py、storage.py、
│                      # config.py、requirements.txt、pytest.ini、tests/
├── sample_data/       # 内置脱敏模拟数据集（GBK csv + xlsx）与 WDBC 导出（csv + xlsx）
├── docs/              # 需求分析.md、系统设计说明书.md、测试报告.md、演示视频脚本.md、api.md
├── deploy/            # ecs_bootstrap.sh 一键部署脚本
├── deploy_huawei.md   # 华为云部署指南
├── .env.example       # 环境变量示例（本地与华为云两套配置模板）
├── .gitignore
└── README.md
```

---

## 四、快速开始

### 环境要求
- Python **≥ 3.10**（推荐 3.10 / 3.11）
- Node.js **≥ 18**

### 启动

```bash
# 1. 后端（默认端口 8002，可用 PORT 环境变量覆盖）
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8002
# 或：python app.py

# 2. 前端（端口固定 3001）
cd frontend
npm install
npm start
```

浏览器访问 **http://127.0.0.1:3001**。

> 未配置任何 API 密钥时项目可正常启动，P0/P1 全部功能可用，AI 功能自动降级为模板解读。

### 跑测试

```bash
cd backend
python -m pytest -q          # 详见 docs/测试报告.md
```

### 示例数据
- `sample_data/medical_sample.csv`（**GBK 编码**，302 例，含缺失值、重复行、超参考区间异常值、ID 列、常量列）
- `sample_data/medical_sample.xlsx`（同一数据集的 Excel 版本）
- `sample_data/wdbc.csv` / `wdbc.xlsx`（WDBC 乳腺癌 569 例，统计与建模全流程基准数据）

前端「使用示例数据体验」可直接加载两者，无需本地文件。

---

## 五、准确性红线（本项目如何遵守）

1. 统计计算只调用 pandas / numpy / scipy / statsmodels；**DCA 与 Hosmer-Lemeshow 是唯二按批准例外手写公式**的指标，二者均配**手工逐步算例**的单元测试（`tests/test_dca_hl.py`）；
2. 模型、评估、交叉验证只用 sklearn 接口，无手写训练逻辑；
3. 前端不做任何统计计算或模型训练，只渲染后端返回结果；
4. 大模型输出的数字必须来自后端计算（提示词附带原始数值并要求一一对应，后端还会程序化提取数字比对，不一致则丢弃并降级）；
5. 大模型不参与检验方法选择、特征筛选结论、模型优劣评估等判断性工作；
6. 每个统计接口配 pytest，期望值由**独立的库调用路径**算出（而非写死数字）；
7. 数据集划分固定 `RANDOM_SEED=42`，建模结果可复现（有专门用例验证两次训练 AUC 完全一致）；
8. 医学参考区间常量集中在 `backend/config.py`，并注明依据文献 / 指南。

---

## 六、操作与快捷键

- 左侧导航按流程推进，完成步骤显示 ✓；
- 顶栏显示当前数据集名称、样本量与变量数徽章、AI 连接状态灯（点击直达设置页）；
- 图表右上角「下载 PNG」导出图片；报告中心「导出 PDF」调用浏览器打印。

---

## 七、华为云部署

见 **`deploy_huawei.md`**（ECS / OBS / CDN / RDS / Redis / 盘古全步骤 + 安全组 + 成本控制 + 部署后自检清单），
一键脚本见 **`deploy/ecs_bootstrap.sh`**。

---

## 八、开源声明

- 本项目代码整理为可开源状态，**不含任何密钥、Token 或真实患者数据**；
- 示例数据全部为程序生成的模拟脱敏数据，WDBC 为公开学术数据集导出；
- 计划托管至 Gitee 仓库（`.gitignore` 已排除 `.env`、`backend/data/`、模型产物与依赖目录）；
- 第三方依赖许可证以其各自仓库为准。

---

## 九、参赛说明

- 赛题：第十一届华为 ICT 大赛创新赛 · 赛题 1（基于华为云产品打造 AI 创新应用）；
- 使用的华为云产品：**ECS、OBS、CDN、RDS for MySQL、DCS Redis、ModelArts Studio（盘古大模型）**；
- 参赛材料：`docs/需求分析.md`、`docs/系统设计说明书.md`、`docs/测试报告.md`、`docs/演示视频脚本.md`；
- 演示主线：以 **WDBC 乳腺癌数据集** 从上传到报告导出一镜到底（脚本见 `docs/演示视频脚本.md`）。
