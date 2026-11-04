#!/usr/bin/env bash
# medical-data-analysis/deploy/ecs_bootstrap.sh
# 华为云 ECS 一键部署脚本（Ubuntu 22.04）
# 用法（root 用户）：
#   export REPO_URL="https://gitee.com/<你的账号>/medical-data-analysis.git"
#   export FRONTEND_ORIGIN="https://<CDN 或 OBS 访问域名>"
#   export RDS_HOST=... RDS_USER=... RDS_PASSWORD=... REDIS_HOST=... REDIS_PASSWORD=...
#   export OBS_AK=... OBS_SK=... OBS_BUCKET=...
#   bash ecs_bootstrap.sh
# 说明：脚本只写环境变量文件，不写入任何密钥到仓库；执行前请先按需修改。

set -euo pipefail

APP_DIR="${APP_DIR:-/root/medical-data-analysis}"
REPO_URL="${REPO_URL:-}"
FRONTEND_ORIGIN="${FRONTEND_ORIGIN:-http://127.0.0.1:3001}"
PORT="${PORT:-8002}"
OBS_SERVER="${OBS_SERVER:-https://obs.cn-north-4.myhuaweicloud.com}"
# API 访问令牌：公网部署必须启用。未显式提供时自动生成随机串并打印一次（请妥善保存，
# 前端在「设置 → 运行时 → 访问令牌」中填写）。
API_TOKEN="${API_TOKEN:-$(openssl rand -hex 24)}"

echo "==> 1/5 安装系统依赖"
apt update -y
apt install -y python3.10-venv python3-pip git curl

echo "==> 2/5 获取代码"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --rebase || true
else
  if [ -z "$REPO_URL" ]; then echo "请设置 REPO_URL"; exit 1; fi
  git clone "$REPO_URL" "$APP_DIR"
fi

echo "==> 3/5 安装 Python 依赖"
cd "$APP_DIR/backend"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# scikit-survival 会拉取无 py3.13 wheel 的 ecos，直接装会中断整次安装：
# 先剔除该行装其余依赖，再 --no-deps 单独安装（CoxPHSurvivalAnalysis 不依赖 ecos）
grep -v "^scikit-survival" requirements.txt > /tmp/medical_req.txt
pip install -r /tmp/medical_req.txt
pip install --no-deps scikit-survival==0.27.0

echo "==> 4/5 写入环境变量（.env，不提交仓库）"
cat > "$APP_DIR/backend/.env" <<EOF
MODE=cloud
HOST=0.0.0.0
PORT=${PORT}
FRONTEND_ORIGIN=${FRONTEND_ORIGIN}
CORS_ORIGINS=${FRONTEND_ORIGIN}
MAX_UPLOAD_MB=50
RANDOM_SEED=42
API_TOKEN=${API_TOKEN}
RDS_HOST=${RDS_HOST:-}
RDS_PORT=${RDS_PORT:-3306}
RDS_USER=${RDS_USER:-}
RDS_PASSWORD=${RDS_PASSWORD:-}
RDS_DATABASE=${RDS_DATABASE:-medical_analysis}
REDIS_HOST=${REDIS_HOST:-}
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_PASSWORD=${REDIS_PASSWORD:-}
REDIS_DB=0
OBS_AK=${OBS_AK:-}
OBS_SK=${OBS_SK:-}
OBS_SERVER=${OBS_SERVER}
OBS_BUCKET=${OBS_BUCKET:-}
CDN_DOMAIN=${CDN_DOMAIN:-}
LLM_PROVIDER=modelarts_pangu
LLM_BASE_URL=https://cn-north-4.modelarts-infer.com/api/v2
LLM_MODEL_NAME=pangu
EOF
# .env 含 RDS 密码 / OBS SK / API_TOKEN：仅 root 可读
chmod 600 "$APP_DIR/backend/.env"

echo "==> 5/5 注册 systemd 服务"
# 安全提醒：以下以 root 运行仅为最小可用部署。生产环境建议创建专用用户
# （useradd -r medical && chown -R medical /path/to/app）并在 [Service] 中加 User=medical。
cat > /etc/systemd/system/medical-api.service <<EOF
[Unit]
Description=Medical Data Analysis API
After=network.target

[Service]
WorkingDirectory=${APP_DIR}/backend
ExecStart=${APP_DIR}/backend/.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}
Restart=always
EnvironmentFile=${APP_DIR}/backend/.env

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now medical-api
sleep 3
systemctl status medical-api --no-pager || true
curl -s "http://127.0.0.1:${PORT}/api/health" || true
echo
echo "部署完成。若输出 mode=cloud 即表示云上模式已生效。"
echo "API 访问令牌（请妥善保存，仅显示一次）：${API_TOKEN}"
echo "提醒：当前为纯 HTTP 服务，公网环境请在前方加 HTTPS 反向代理（nginx/Caddy）。"
