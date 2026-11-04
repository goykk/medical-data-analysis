@echo off
rem Medical Data Analysis - one-click start (Windows)
rem Usage: double-click start.bat. Project path must not contain spaces.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "BE_DIR=%~dp0backend"
set "FE_DIR=%~dp0frontend"

echo [1/4] Checking backend dependencies...
python -c "import sklearn, pandas, sksurv, xgboost, lightgbm, catboost, imblearn, shap, pypinyin" >nul 2>&1
if errorlevel 1 (
  echo   Installing backend dependencies...
  rem scikit-survival 会拉取无 py3.13 wheel 的 ecos，直接装会中断整次安装：
  rem 先剔除该行装其余依赖，再 --no-deps 单独安装
  findstr /v /b "scikit-survival" "%BE_DIR%\requirements.txt" > "%TEMP%\medical_req.txt"
  python -m pip install -r "%TEMP%\medical_req.txt" || goto :fail
  python -m pip install --no-deps scikit-survival==0.27.0 || goto :fail
) else (
  echo   Backend dependencies OK.
)

echo [2/4] Checking frontend dependencies...
if exist "%FE_DIR%\node_modules\express" (
  echo   Frontend dependencies OK.
) else (
  echo   Installing frontend dependencies...
  pushd "%FE_DIR%"
  npm install
  if errorlevel 1 (popd & goto :fail)
  popd
)

echo [3/4] Starting backend on http://127.0.0.1:8002 ...
start "medical-backend" cmd /k "cd /d %BE_DIR% && python -m uvicorn app:app --host 127.0.0.1 --port 8002"
timeout /t 4 /nobreak >nul

echo [4/4] Starting frontend on http://127.0.0.1:3001 ...
start "medical-frontend" cmd /k "cd /d %FE_DIR% && npm start"
timeout /t 2 /nobreak >nul

start "" "http://127.0.0.1:3001/"
echo.
echo Done. Frontend: http://127.0.0.1:3001/   Backend: http://127.0.0.1:8002/
echo Close the two server windows to stop.
goto :eof

:fail
echo.
echo ERROR: setup failed. Check the message above.
pause
exit /b 1
