#!/usr/bin/env bash
# 医疗数据分析平台 - 一键启动（Linux / macOS / Git Bash）
# 用法：bash start.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] Checking backend dependencies..."
if ! python -c "import sklearn, pandas, sksurv, xgboost, lightgbm, catboost, imblearn, shap, pypinyin" >/dev/null 2>&1; then
  echo "  Installing backend dependencies..."
  # scikit-survival 会拉取无 py3.13 wheel 的 ecos，直接装会中断整次安装：
  # 先剔除该行装其余依赖，再 --no-deps 单独安装
  grep -v "^scikit-survival" "$ROOT/backend/requirements.txt" > /tmp/medical_req.txt
  python -m pip install -r /tmp/medical_req.txt
  python -m pip install --no-deps scikit-survival==0.27.0
else
  echo "  Backend dependencies OK."
fi

echo "[2/4] Checking frontend dependencies..."
if [ -d "$ROOT/frontend/node_modules/express" ]; then
  echo "  Frontend dependencies OK."
else
  echo "  Installing frontend dependencies..."
  (cd "$ROOT/frontend" && npm install)
fi

echo "[3/4] Starting backend on http://127.0.0.1:8002 ..."
(cd "$ROOT/backend" && python -m uvicorn app:app --host 127.0.0.1 --port 8002) &
sleep 4

echo "[4/4] Starting frontend on http://127.0.0.1:3001 ..."
(cd "$ROOT/frontend" && npm start) &
sleep 2

echo ""
echo "Done. Frontend: http://127.0.0.1:3001/   Backend: http://127.0.0.1:8002/"
echo "按 Ctrl+C 停止两个服务。"
wait
