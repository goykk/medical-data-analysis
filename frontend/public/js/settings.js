// medical-data-analysis/frontend/public/js/settings.js
// 设置页：大模型接入（含华为云盘古）+ 测试连接耗时。密钥只在服务端内存中。

(function () {
  'use strict';
  window.Pages = window.Pages || {};

  const PRESET = {
    deepseek: { url: 'https://api.deepseek.com/v1', model: 'deepseek-chat', label: 'DeepSeek' },
    qwen: { url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: 'qwen-plus', label: '通义千问' },
    glm: { url: 'https://open.bigmodel.cn/api/paas/v4', model: 'glm-4-flash', label: '智谱 GLM' },
    modelarts_pangu: {
      url: 'https://cn-north-4.modelarts-infer.com/api/v2', model: 'pangu',
      label: '华为云 ModelArts Studio（盘古）',
    },
    custom: { url: '', model: '', label: '自定义' },
  };

  let savedAt = null;      // 会话内记录的保存时间
  let savedMask = '';      // 密钥掩码（仅会话内显示）

  function maskOf(key) {
    if (!key) return '';
    if (key.length <= 8) return '••••••••';
    return key.slice(0, 4) + '••••••••' + key.slice(-4);
  }

  function renderKeyState(configured) {
    const box = document.getElementById('keyMask');
    if (!box) return;
    if (savedAt) {
      box.innerHTML = `<span class="mono">${esc(savedMask)}</span>
        <span class="tag green">已保存</span>
        <span class="mono" style="color:var(--text-mute)">${esc(savedAt)}</span>`;
    } else if (configured) {
      box.innerHTML = `<span class="mono">••••••••</span>
        <span class="tag green">已配置</span>
        <span class="mono" style="color:var(--text-mute)">服务端内存</span>`;
    } else {
      box.innerHTML = '<span class="tag gray">未配置</span>';
    }
  }

  function applyPreset(key, force) {
    const p = PRESET[key] || PRESET.custom;
    const url = document.getElementById('llmBaseUrl');
    const model = document.getElementById('llmModel');
    if (force || !url.value) url.value = p.url;
    if (force || !model.value) model.value = p.model;
    const hint = document.getElementById('llmHint');
    if (key === 'modelarts_pangu') {
      hint.innerHTML = '盘古大模型使用 API Key 鉴权（请求头 <code>X-Apig-AppCode</code>）。'
        + '如部署区域不同，请把 Base URL 中的 <code>cn-north-4</code> 替换为实际区域。';
    } else {
      hint.textContent = '';
    }
  }

  function init() {
    const sel = document.getElementById('llmProvider');

    sel.addEventListener('change', () => {
      applyPreset(sel.value, true);
      if (window.LinkStatus) LinkStatus.refreshAi();
    });

    document.getElementById('toggleKey').addEventListener('click', () => {
      const k = document.getElementById('llmKey');
      const btn = document.getElementById('toggleKey');
      if (k.type === 'password') { k.type = 'text'; btn.textContent = '隐藏'; }
      else { k.type = 'password'; btn.textContent = '显示'; }
    });

    document.getElementById('testConn').addEventListener('click', () => {
      const btn = document.getElementById('testConn');
      const hint = document.getElementById('llmHint');
      const started = performance.now();
      btn.disabled = true; btn.textContent = '测试中…';
      hint.dataset.preset = sel.value;
      API.post('/api/llm/test', {
        provider: sel.value,
        base_url: document.getElementById('llmBaseUrl').value,
        model: document.getElementById('llmModel').value,
        api_key: document.getElementById('llmKey').value,
      }).then((r) => {
        const local = Math.round(performance.now() - started);
        const ms = (r.latency_ms !== null && r.latency_ms !== undefined) ? r.latency_ms : local;
        const line = `${esc(r.message)}　<span class="mono">耗时 ${ms} ms</span>`;
        hint.innerHTML = `<span class="${r.connected ? 'tag green' : 'tag red'}">${r.connected ? '连接成功' : '连接失败'}</span> ${line}`;
        toast(r.message, r.connected ? 'ok' : 'err');
      }).catch(() => {
        const local = Math.round(performance.now() - started);
        hint.innerHTML = `<span class="tag red">连接失败</span> <span class="mono">耗时 ${local} ms</span>`;
      }).finally(() => { btn.disabled = false; btn.textContent = '测试连接'; });
    });

    document.getElementById('saveLlm').addEventListener('click', () => {
      const key = document.getElementById('llmKey').value;
      API.post('/api/llm/config', {
        provider: sel.value,
        base_url: document.getElementById('llmBaseUrl').value,
        model: document.getElementById('llmModel').value,
        api_key: key,
      }).then(() => {
        document.getElementById('llmKey').value = '';
        if (key) { savedMask = maskOf(key); savedAt = Theme.stampText(); }
        renderKeyState(true);
        toast('配置已保存（密钥仅存服务端内存）', 'ok');
        APP.refreshAiStatus();
      });
    });

    // 访问令牌：仅存本机 localStorage，由 api.js 随请求头发送
    const tkInput = document.getElementById('apiToken');
    if (tkInput) {
      tkInput.value = API.getToken();
      document.getElementById('saveToken').addEventListener('click', () => {
        API.setToken(tkInput.value.trim());
        toast(tkInput.value.trim() ? '访问令牌已保存到本机浏览器' : '已清除本机访问令牌', 'ok');
      });
    }

    initStorage();
    refreshStatus();
  }

  function selEl() { return document.getElementById('llmProvider'); }

  /**
   * 已保存配置时沿用用户选择；未配置时默认 DeepSeek。
   * 华为云盘古保留在下拉列表中作为可选能力。
   */
  function applyStatus(s) {
    const sel = selEl();
    if (s.configured) {
      if (s.provider) sel.value = s.provider;
    } else {
      sel.value = 'deepseek';
    }
    if (s.base_url) document.getElementById('llmBaseUrl').value = s.base_url;
    if (s.model) document.getElementById('llmModel').value = s.model;
    if (!s.configured) applyPreset(sel.value, true);
    renderKeyState(s.configured);
    // 侧栏 AI 灯跟随设置页的实际选择
    if (window.LinkStatus) LinkStatus.refreshAi();
  }

  function refreshStatus() {
    API.get('/api/llm/status').then(applyStatus).catch(() => {
      applyPreset('deepseek', true);
      renderKeyState(false);
    });
  }

  // ---------------- 数据与存储 ----------------
  const KEEP_KEY = 'mdp.keepRecent';
  const RECENT_KEY = 'mdp.recent.v1';

  function readRecent() {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]'); } catch (e) { return []; }
  }

  function renderRecentCount() {
    const el = document.getElementById('recentCount');
    if (el) el.textContent = `当前 ${readRecent().length} 条`;
  }

  function loadHistoryCount() {
    const el = document.getElementById('histCount');
    if (!el) return;
    API.get('/api/history').then((r) => {
      el.textContent = String(((r || {}).items || []).length);
    }).catch(() => { el.textContent = '读取失败'; });
  }

  function initStorage() {
    const sel = document.getElementById('keepRecent');
    if (sel) {
      sel.value = String(Number(localStorage.getItem(KEEP_KEY) || 20));
      sel.addEventListener('change', () => {
        localStorage.setItem(KEEP_KEY, sel.value);
        toast(`本地最近文件保留上限设为 ${sel.value} 条`, 'ok');
      });
    }
    const clr = document.getElementById('clearRecent');
    if (clr) clr.addEventListener('click', () => {
      const n = readRecent().length;
      if (!n) { toast('本地最近文件记录为空', 'warn'); return; }
      localStorage.removeItem(RECENT_KEY);
      toast(`已清空 ${n} 条本地记录`, 'ok');
      renderRecentCount();
      if (window.Pages && window.Pages.upload && window.Pages.upload.refresh) window.Pages.upload.refresh();
    });
    renderRecentCount();
  }

  window.Pages.settings = {
    init,
    onShow() { refreshStatus(); renderRecentCount(); loadHistoryCount(); },
  };
})();
