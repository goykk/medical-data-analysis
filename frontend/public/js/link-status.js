// medical-data-analysis/frontend/public/js/link-status.js
// 全局链路状态区：侧栏底部三盏常驻状态灯（后端心跳 / 模型 / AI）。
// 约定：所有取数都在这个单例里完成，三盏灯只负责按 state 渲染，不各自发请求。
// 所有请求失败一律静默：不弹窗、不 toast、不往控制台刷错误。
//
// 说明：这里刻意不用 API.get —— api.js 的 request 会在失败时弹 toast，
// 与"离线期间无任何干扰"的要求冲突，所以心跳与模型探测走原生 fetch。

(function () {
  'use strict';

  const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.apiBase) || 'http://127.0.0.1:8002';
  const POLL_MS = 30000;      // 心跳轮询间隔（固定 30 秒，不做指数退避）
  const TIMEOUT_MS = 3000;    // 单次心跳超时
  const LS_MODEL_KEY = 'mdp.model.latest';
  const PROVIDER_NAME = {
    deepseek: 'DeepSeek', qwen: '通义千问', glm: 'GLM-4',
    modelarts_pangu: '盘古', custom: '自定义',
  };

  const state = {
    backend: { online: false, latency: null },
    model: { trained: false, algorithm: '', fingerprint: '', datasetId: '' },
    ai: { configured: false, name: 'DeepSeek' },
  };

  const $ = (id) => document.getElementById(id);

  // ---------------- 工具 ----------------
  function algoShort(label) {
    // 后端 label 形如"逻辑回归（Logistic Regression）"，侧栏只取中文名
    return String(label || '').split('（')[0].trim() || '模型';
  }
  function fingerprint(datasetId) {
    return window.Theme ? Theme.fingerprint(datasetId) : String(datasetId || '').slice(0, 8);
  }
  function readSnapshot() {
    try { return JSON.parse(localStorage.getItem(LS_MODEL_KEY) || 'null'); } catch (e) { return null; }
  }
  function writeSnapshot(s) {
    try { localStorage.setItem(LS_MODEL_KEY, JSON.stringify(s)); } catch (e) { /* 隐私模式忽略 */ }
  }
  function clearSnapshot() {
    try { localStorage.removeItem(LS_MODEL_KEY); } catch (e) { /* ignore */ }
  }
  /** 访问令牌头：后端启用 API_TOKEN 时，除 /api/health 外所有接口都需要 */
  function authHeaders() {
    const tk = (window.API && API.getToken) ? API.getToken() : (localStorage.getItem('mdp.apiToken') || '');
    return tk ? { 'X-API-Token': tk } : {};
  }
  function emptyModel(datasetId) {
    state.model = { trained: false, algorithm: '', fingerprint: '', datasetId: datasetId || '' };
  }

  // ---------------- 1. 后端心跳 ----------------
  function probeHealth() {
    const ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const started = performance.now();
    const timer = setTimeout(() => { if (ctrl) ctrl.abort(); }, TIMEOUT_MS);
    const opts = { cache: 'no-store' };
    if (ctrl) opts.signal = ctrl.signal;

    fetch(API_BASE + '/api/health', opts)
      .then((r) => { if (!r.ok) throw new Error('unhealthy'); return r.json(); })
      .then(() => {
        state.backend.online = true;
        state.backend.latency = Math.round(performance.now() - started);
      })
      .catch(() => { state.backend.online = false; state.backend.latency = null; })
      .finally(() => { clearTimeout(timer); renderBackend(); });
  }

  let pollTimer = null;
  function startPoll() {
    stopPoll();
    probeHealth();
    pollTimer = setInterval(() => { if (!document.hidden) probeHealth(); }, POLL_MS);
  }
  function stopPoll() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function renderBackend() {
    const dot = $('lsBackendDot'), txt = $('lsBackendText'), row = $('lsBackend');
    if (!dot || !txt || !row) return;
    if (state.backend.online) {
      dot.className = 'ls-dot ok';
      txt.textContent = `在线 · ${state.backend.latency}ms`;
      row.title = `后端心跳：每 30 秒轮询 /api/health，当前延迟 ${state.backend.latency}ms`;
    } else {
      dot.className = 'ls-dot err';
      txt.textContent = '后端离线';
      row.title = '后端心跳：每 30 秒轮询 /api/health，最近一次请求失败或超时（3s）';
    }
  }

  // ---------------- 2. 模型状态 ----------------
  /** 优先向后端确认当前数据集是否已有模型；后端不在内存时退回本地快照 */
  function probeModel(datasetId) {
    if (!datasetId) return;
    fetch(`${API_BASE}/api/ml/model/${datasetId}`, { cache: 'no-store', headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('none'))))
      .then((info) => {
        if (info && info.trained) setModel(info.best_model, datasetId);
        else { emptyModel(datasetId); clearSnapshot(); renderModel(); }
      })
      .catch(() => { /* 静默：保持当前状态 */ });
  }

  function setModel(bestModel, datasetId) {
    if (!bestModel) { emptyModel(datasetId); clearSnapshot(); renderModel(); return; }
    state.model = {
      trained: true,
      algorithm: algoShort(bestModel.label),
      fingerprint: fingerprint(datasetId),
      datasetId: datasetId || '',
    };
    writeSnapshot(state.model);
    renderModel();
  }

  /** 建模完成：由分析配置页调用 */
  function setModelFromMl(ml, datasetId) {
    if (!ml || ml.skipped || !ml.best_model) { emptyModel(datasetId); clearSnapshot(); renderModel(); return; }
    setModel(ml.best_model, datasetId);
  }

  /** 切换数据集：旧模型随之失效，回到灰显 */
  function onDatasetChanged(datasetId) {
    const snap = readSnapshot();
    if (snap && snap.datasetId && (!datasetId || snap.datasetId === datasetId)) {
      state.model = snap;
      renderModel();
      return;
    }
    emptyModel(datasetId);
    clearSnapshot();
    renderModel();
    if (datasetId) probeModel(datasetId);
  }

  function renderModel() {
    const dot = $('lsModelDot'), txt = $('lsModelText'), row = $('lsModel');
    if (!dot || !txt || !row) return;
    if (state.model.trained) {
      dot.className = 'ls-dot on';   // 已建模用主色点，区别于"在线"的绿点
      txt.textContent = `${state.model.algorithm} · ${state.model.fingerprint}`;
      row.title = `当前最优模型：${state.model.algorithm}（数据集指纹 ${state.model.fingerprint}）`;
    } else {
      dot.className = 'ls-dot idle';
      txt.textContent = '未建模';
      row.title = '尚未完成建模；建模完成后自动点亮，刷新后状态保留';
    }
    // 智能建模不在七步流程里（可选侧路），勾选与否直接跟模型灯走同源状态
    if (window.APP && APP.markDone) APP.markDone('automl', state.model.trained);
  }

  // ---------------- 3. AI 模型 ----------------
  /** 以"是否已配置"为准，不做主动连通性探测 */
  function refreshAi() {
    fetch(API_BASE + '/api/llm/status', { cache: 'no-store', headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('bad'))))
      .then((s) => {
        state.ai.configured = !!s.configured;
        state.ai.name = currentProviderName(s);
      })
      .catch(() => { /* 静默：保留上一次已知状态 */ })
      .finally(renderAi);
  }

  /** 名称跟随设置页当前选中的服务商，避免与下拉框显示不一致 */
  function currentProviderName(s) {
    const sel = $('llmProvider');
    if (sel && sel.value && PROVIDER_NAME[sel.value]) return PROVIDER_NAME[sel.value];
    if (s && s.model) return s.model;
    if (s && s.provider_label) return s.provider_label;
    return 'DeepSeek';
  }

  function renderAi() {
    const dot = $('lsAiDot'), txt = $('lsAiText'), row = $('lsAi');
    if (!dot || !txt || !row) return;
    if (state.ai.configured) {
      dot.className = 'ls-dot ok';
      txt.textContent = `${state.ai.name} · 已连接`;
      row.title = `AI 模型：${state.ai.name} 已配置（以配置状态为准，不做主动探测）`;
    } else {
      dot.className = 'ls-dot idle';
      txt.textContent = `${state.ai.name}未配置 · 模板降级`;
      row.title = `AI 模型：${state.ai.name} 未配置，解读走模板降级；点击前往设置`;
    }
  }

  // ---------------- 跳转（沿用现有 APP.go 路由） ----------------
  function goto(page, anchorId) {
    if (window.APP && typeof APP.go === 'function') APP.go(page);
    if (!anchorId) return;
    requestAnimationFrame(() => {
      const el = $(anchorId);
      if (el && el.scrollIntoView) el.scrollIntoView({ block: 'start' });
    });
  }

  function bindClicks() {
    const b = $('lsBackend'), m = $('lsModel'), a = $('lsAi');
    if (b) b.addEventListener('click', () => goto('settings', 'runtime'));
    if (m) m.addEventListener('click', () => goto('prediction', null));
    if (a) a.addEventListener('click', () => goto('settings', 'ai-config'));
  }

  // ---------------- 初始化 ----------------
  function init() {
    bindClicks();
    onDatasetChanged(null);   // 载入本地快照，刷新后保持
    renderAi();
    refreshAi();
    startPoll();              // 心跳：立即一次 + 每 30 秒
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) stopPoll(); else startPoll();
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

  window.LinkStatus = {
    state, setModelFromMl, onDatasetChanged, refreshAi,
    renderModel, renderBackend, renderAi,
  };
})();
