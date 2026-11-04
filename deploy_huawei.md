# medical-data-analysis/deploy_huawei.md
# 华为云部署指南（ECS + OBS + CDN + RDS + Redis + ModelArts 盘古）

> 适用赛事：第十一届华为 ICT 大赛创新赛（赛题 1：基于华为云产品打造 AI 创新应用）
> 目标：按本文档从零开始，可在华为云环境一次性跑通；评委可复现。
> 成本控制提醒见文末 **第 8 节**，资源用完请立即释放。

---

## 0. 架构与端口一览

| 组件 | 云产品 | 作用 | 本地模式替代 |
|---|---|---|---|
| 前端 | OBS 静态网站托管 + CDN | 页面与 JS/CSS 分发 | Node Express（3001） |
| 后端 | ECS（弹性云服务器） | FastAPI 服务（8002） | 本机 uvicorn |
| 分析历史 | RDS for MySQL | 方案与历史持久化 | SQLite 单文件 |
| 任务状态 / 缓存 | DCS Redis | 长任务状态、预测缓存 | 进程内存 |
| 数据集 / 模型产物 | OBS | 对象存储 | 本地目录 |
| 大模型 | ModelArts Studio（盘古） | AI 解读与意图解析 | 自动降级为模板解读 |

- 后端端口固定 **8002**，前端本地端口固定 **3001**（见 README）。
- 两种模式业务代码共用，仅由环境变量 `MODE` 切换：`MODE=local` / `MODE=cloud`。

---

## 1. 准备：本地自检（先在非云环境跑通）

```bash
# 后端
cd backend
pip install -r requirements.txt
python -m pytest -q                 # 应全部通过
python -m uvicorn app:app --host 127.0.0.1 --port 8002

# 前端（新开终端）
cd frontend
npm install
npm start                            # http://127.0.0.1:3001
```

浏览器访问 `http://127.0.0.1:3001` → 使用示例数据（WDBC）走通「上传 → 报告导出」。

---

## 2. ECS：购买与后端部署

### 2.1 购买
1. 控制台 → 弹性云服务器 ECS → 购买弹性云服务器；
2. 计费模式：**按需计费**（便于赛后释放）；
3. 区域：建议 **华北-北京四（cn-north-4）**，与盘古 / OBS 保持一致；
4. 规格：2 vCPU / 4 GiB（如 `s6.medium.2`）即可；
5. 镜像：**Ubuntu 22.04**；
6. 网络：默认 VPC + 子网，自动分配弹性公网 IP（或后续绑定 EIP）。

### 2.2 安全组（防火墙）
在 ECS 所属安全组添加入方向规则：

| 协议/端口 | 源地址 | 说明 |
|---|---|---|
| TCP 22 | 自己的 IP | SSH 登录 |
| TCP 8002 | 0.0.0.0/0（演示期） | 后端 API；生产建议仅放行前端所在网段 |
| TCP 80 / 443 | 0.0.0.0/0 | 若后端直接对外提供 HTTP |

> 服务器内部若启用 `ufw`：
> ```bash
> sudo ufw allow 22/tcp && sudo ufw allow 8002/tcp && sudo ufw enable
> ```

### 2.3 部署命令（SSH 登录后逐条执行）

```bash
sudo apt update && sudo apt install -y python3.10-venv python3-pip nginx git
git clone <你的 Gitee 仓库地址> medical-data-analysis
cd medical-data-analysis/backend

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 环境变量（云上模式）
cat > .env <<'EOF'
MODE=cloud
HOST=0.0.0.0
PORT=8002
FRONTEND_ORIGIN=https://<你的 CDN 或 OBS 访问域名>
CORS_ORIGINS=https://<你的 CDN 或 OBS 访问域名>
MAX_UPLOAD_MB=50
RANDOM_SEED=42
RDS_HOST=<RDS 内网地址>
RDS_PORT=3306
RDS_USER=<用户名>
RDS_PASSWORD=<密码>
RDS_DATABASE=medical_analysis
REDIS_HOST=<Redis 内网地址>
REDIS_PORT=6379
REDIS_PASSWORD=<密码>
REDIS_DB=0
OBS_AK=<AK>
OBS_SK=<SK>
OBS_SERVER=https://obs.cn-north-4.myhuaweicloud.com
OBS_BUCKET=<桶名>
CDN_DOMAIN=<CDN 域名，可留空>
LLM_PROVIDER=modelarts_pangu
LLM_BASE_URL=https://cn-north-4.modelarts-infer.com/api/v2
LLM_MODEL_NAME=pangu
EOF

# systemd 托管
sudo tee /etc/systemd/system/medical-api.service >/dev/null <<'EOF'
[Unit]
Description=Medical Data Analysis API
After=network.target

[Service]
WorkingDirectory=/root/medical-data-analysis/backend
ExecStart=/root/medical-data-analysis/backend/.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8002
Restart=always
EnvironmentFile=/root/medical-data-analysis/backend/.env

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now medical-api
sudo systemctl status medical-api --no-pager
curl http://127.0.0.1:8002/api/health        # 期望 {"status":"ok","mode":"cloud",...}
```

> 一键脚本：仓库根目录 `deploy/ecs_bootstrap.sh` 已封装以上步骤（见 README）。

---

## 3. OBS：桶创建、静态托管与前端发布

1. 控制台 → 对象存储服务 OBS → 创建桶：
   - 区域：cn-north-4；**存储类别：标准存储**；桶策略：**公共读**（仅静态资源目录）；
2. 桶内建目录：`static/`（前端）、`datasets/`（上传数据）、`models/`（模型产物）；
3. 构建并上传前端：

```bash
cd frontend
# 前端静态资源需要把 API 地址指向 ECS 公网地址
BACKEND_ORIGIN=http://<ECS 公网 IP>:8002 npm start   # 取一次 /config.js
# 或手动生成 config.js 后上传
echo 'window.APP_CONFIG={"apiBase":"http://<ECS 公网 IP>:8002"};' > public/config.js
obsutil cp -r -f public/ obs://<桶名>/static/
```

> `obsutil` 下载：控制台 → OBS → 工具 → obsutil；配置 `obsutil config -i=<AK> -k=<SK> -e=https://obs.cn-north-4.myhuaweicloud.com`

4. 静态网站托管：桶 → 基础配置 → 静态网站托管 → 开启，**默认首页** `index.html`，**默认 404 页面** `index.html`（单页应用）；
   - 若前端使用 `history` 路由，单页应用请配 404 回退到 `index.html`；本项目为 hash-less 单页且由 Express 兜底，静态托管场景直接回退即可。

---

## 4. CDN：静态资源加速

1. 控制台 → 内容分发网络 CDN → 添加域名：
   - 加速域名：你的已备案域名（如 `medical-demo.example.com`）；
   - 业务类型：网站加速；
   - 源站类型：**OBS 桶域名**，源站地址选择第 3 步的桶；
2. 回源配置：开启「静态网站托管回源」；
3. 缓存规则：`*.js;*.css;*.png` 缓存 30 天，`index.html` 缓存 0（不缓存）；
4. 配置 HTTPS：申请免费证书并开启强制跳转；
5. 解析：把加速域名的 CNAME 指向 CDN 分配的 CNAME；
6. 完成后把 `FRONTEND_ORIGIN` / `CORS_ORIGINS` 改成 CDN 域名，重启后端：

```bash
sudo systemctl restart medical-api
```

---

## 5. RDS for MySQL：创建与连接

1. 控制台 → 云数据库 RDS → 购买数据库实例：
   - 引擎 MySQL 8.0；规格：**通用型 2 vCPU / 4 GiB**；存储 40 GB；
   - 与 ECS **同区域、同 VPC**；
2. 创建账号（如 `medical`）与数据库 `medical_analysis`；
3. 白名单 / 安全组：允许 ECS 内网地址访问 3306；
4. 连接测试（在 ECS 上）：

```bash
python - <<'PY'
import pymysql
conn = pymysql.connect(host="<RDS内网地址>", port=3306, user="medical",
                       password="<密码>", database="medical_analysis")
with conn.cursor() as cur:
    cur.execute("SHOW TABLES")
    print(cur.fetchall())
PY
```

5. 表结构由 `backend/storage.py` 首次调用时自动创建（`analysis_history`）。

---

## 6. DCS Redis：创建与连接

1. 控制台 → 分布式缓存服务 DCS → 购买 **Redis 5.0 单机/主备**（测试用单机即可）；
2. 同 VPC，端口 6379，设置密码（或免密，生产不建议）；
3. 连接测试：

```bash
python - <<'PY'
import redis
r = redis.Redis(host="<Redis内网地址>", port=6379, password="<密码>", decode_responses=True)
r.setex("probe", 30, "ok"); print(r.get("probe"))
PY
```

---

## 7. ModelArts Studio（盘古大模型）：接入与验证

1. 控制台 → ModelArts Studio → 大模型即服务平台 → 创建 **API Key**（AppCode）；
2. 开通所需模型（如盘古通用大模型），记录 **模型名称**与 **推理 API 地址**；
3. 平台「设置」页选择 **华为云 ModelArts Studio（盘古）**，
   Base URL 预填 `https://cn-north-4.modelarts-infer.com/api/v2`，鉴权头为 `X-Apig-AppCode`；
4. 点击「测试连接」：成功会显示 **延迟 xx ms**；失败会提示
   *密钥错误 / 模型名不存在 / 网络不通* 之一；
5. 未配置密钥时，AI 功能自动降级为模板解读，P0/P1 全部功能不受影响。

> 盘古接入代码见 `backend/llm_service.py`；密钥只存服务端内存，不写入任何文件或日志。

---

## 8. 环境变量模板与成本控制

复制仓库根目录 `.env.example` 为 `.env` 后修改；云上关键项见第 2.3 节。
完整模板（本地 + 华为云两套）见 `.env.example`，其中：
- 本地：`MODE=local`、`FRONTEND_ORIGIN=http://127.0.0.1:3001`、`LOCAL_SQLITE_PATH=./backend/data/history.db`
- 云上：`MODE=cloud` + RDS / Redis / OBS / CDN / 盘古 配置

**成本控制（重要）**
- 全部资源使用**按需计费**，评测结束后立即释放：ECS → 删除弹性云服务器；RDS / DCS → 删除实例；OBS → 清空并删除桶；CDN → 停用并删除加速域名；
- OBS 只存放必要数据集与模型，避免长期存放大文件；
- 不使用时可**停止** ECS（停止仍收取存储费，释放才彻底停止计费）；
- 盘古大模型按调用量计费，演示结束后删除 API Key。

---

## 9. 部署后自检清单（交付前逐项打勾）

```bash
# 1) 后端健康
curl http://<ECS公网IP>:8002/api/health
# 期望：{"status":"ok","mode":"cloud",...}

# 2) 存储后端确认
curl http://<ECS公网IP>:8002/api/config | python -m json.tool
# 期望 storage.mode = cloud，history=华为云 RDS，cache=华为云 Redis，objects=华为云 OBS

# 3) 前端可访问且拿到后端地址
curl https://<CDN域名>/config.js
# 期望：window.APP_CONFIG = {"apiBase":"http://<ECS公网IP>:8002"};

# 4) 端到端：示例数据 → 建模 → 预测
curl -X POST http://<ECS公网IP>:8002/api/sample/load -H 'Content-Type: application/json' -d '{"key":"wdbc"}'
# 取回 dataset_id 后
curl -X POST http://<ECS公网IP>:8002/api/ml/train -H 'Content-Type: application/json' \
     -d '{"dataset_id":"<id>","features":["radius_mean","texture_mean","perimeter_mean","area_mean","concavity_mean"],"target":"diagnosis","task":"classification"}'
# 期望：roc.auc > 0.9，且 calibration / dca 均存在

# 5) 历史写入 RDS
curl http://<ECS公网IP>:8002/api/history
# 期望返回 items 列表（云上模式来自 RDS）

# 6) 后端单元测试（部署机上再跑一遍）
cd backend && python -m pytest -q
```

---

## 10. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 前端报 CORS 错误 | `FRONTEND_ORIGIN` 与实际访问域名不一致 | 修改 `.env` 后 `systemctl restart medical-api` |
| `curl /api/health` 超时 | 安全组未放通 8002 | 安全组入方向放行 TCP 8002 |
| 历史接口报 `pymysql` 缺失 | 云上模式依赖未安装 | `pip install pymysql redis esdk-obs-python` |
| 盘古测试连接 401/403 | API Key 错误或模型未开通 | 重新创建 AppCode、确认模型已开通 |
| OBS 上传 403 | AK/SK 无该桶权限或桶区域不符 | 检查 IAM 权限与 `OBS_SERVER` 区域 |
| 静态页刷新 404 | 未配置 404 回退 | OBS 静态托管默认 404 页设为 `index.html` |
