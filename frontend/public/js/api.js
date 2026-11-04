// medical-data-analysis/frontend/public/js/api.js
// fetch 封装：统一加载态与耗时计时、统一错误提示（读 {code, message, detail} 的 message）、XSS 转义工具。

(function () {
  'use strict';

  const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.apiBase) || 'http://127.0.0.1:8002';

  let loadingCount = 0;
  let busyLabel = '处理中';
  const loadingEl = () => document.getElementById('loading');

  /** 设定下一轮忙态的文案，如「分析中」「预测中」，耗时由底部状态栏统一计时 */
  function setBusyLabel(t) { busyLabel = t || '处理中'; }

  function setLoading(on) {
    const before = loadingCount;
    loadingCount = Math.max(0, loadingCount + (on ? 1 : -1));
    const el = loadingEl();
    const T = window.Theme;
    if (T) {
      if (loadingCount > 0 && before === 0) T.beginBusy(busyLabel);
      else if (loadingCount === 0 && before > 0) T.endBusy('完成');
    }
    if (el) el.hidden = loadingCount === 0;
  }

  function toast(msg, type) {
    const box = document.getElementById('toastBox');
    if (!box) return;
    const d = document.createElement('div');
    d.className = 'toast' + (type ? ' ' + type : '');
    d.textContent = msg;
    box.appendChild(d);
    setTimeout(() => d.remove(), 3600);
    while (box.children.length > 4) box.removeChild(box.firstChild);
  }

  /** HTML 转义：所有数据集文本字段与 AI 返回内容渲染前必须经过此函数 */
  function esc(v) {
    if (v === null || v === undefined) return '';
    return String(v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  /** 访问令牌（后端 API_TOKEN 启用时必需）：仅存本机 localStorage，设置页可改 */
  function getToken() { return localStorage.getItem('mdp.apiToken') || ''; }
  function setToken(t) {
    if (t) localStorage.setItem('mdp.apiToken', t);
    else localStorage.removeItem('mdp.apiToken');
  }

  async function request(path, options) {
    const opts = Object.assign({ headers: {} }, options || {});
    const isForm = opts.body instanceof FormData;
    if (opts.body && !isForm) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(opts.body);
    }
    const tk = getToken();
    if (tk) opts.headers['X-API-Token'] = tk;
    setLoading(true);
    try {
      const res = await fetch(API_BASE + path, opts);
      let data = {};
      try { data = await res.json(); } catch (e) { /* 非 JSON 响应 */ }
      if (!res.ok) {
        let msg = data.message || (res.status === 413 ? '文件超过大小限制' : '请求失败');
        // 后端对可修正错误会附带 suggestion（如“补齐事件状态列”），一并展示，
        // 否则用户只看到原因，不知道该改哪里。
        if (data.suggestion) msg += '；建议：' + data.suggestion;
        toast(msg, 'err');
        const err = new Error(msg);
        err.payload = data;
        err.status = res.status;
        throw err;
      }
      return data;
    } catch (e) {
      if (window.Theme) window.Theme.failBusy(e.message || '请求失败');
      if (e instanceof TypeError) {
        toast('无法连接后端服务：' + API_BASE +
          '（若页面用 localhost 打开，请改用 127.0.0.1；或后端未允许当前来源）', 'err');
      }
      throw e;
    } finally {
      setLoading(false);
    }
  }

  const get = (p) => request(p, { method: 'GET' });
  const post = (p, body) => request(p, { method: 'POST', body: body });
  const del = (p) => request(p, { method: 'DELETE' });

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /**
   * 轮询任务状态。刻意不走 request()：轮询每 800ms 一次，若复用全局加载态会让
   * 底部忙碌条反复闪烁、计时器反复重置。
   */
  async function pollTask(taskId) {
    const headers = {};
    const tk = getToken();
    if (tk) headers['X-API-Token'] = tk;
    const res = await fetch(API_BASE + '/api/tasks/' + encodeURIComponent(taskId), { headers });
    let data = {};
    try { data = await res.json(); } catch (e) { /* 非 JSON 响应 */ }
    if (!res.ok) {
      const err = new Error(data.message || '任务查询失败');
      err.status = res.status;
      throw err;
    }
    return data;
  }

  /**
   * 提交异步任务并轮询到完成（训练耗时长，避免 HTTP 长连接超时）。
   * 返回任务 result；running 期间保持底部忙碌态与计时，error 时抛出后端原因。
   */
  async function postTask(path, body, onProgress) {
    const submit = await request(path, { method: 'POST', body: body });
    const taskId = submit && submit.task_id;
    if (!taskId) return submit;                 // 后端未启用异步时按同步响应兜底
    const startedAt = Date.now();
    setBusyLabel('模型训练中');
    setLoading(true);
    try {
      for (;;) {
        await sleep(800);
        const st = await pollTask(taskId);
        if (onProgress) onProgress(st);
        if (st.status === 'done') return st.result;
        if (st.status === 'error') {
          const err = new Error(st.error || '训练失败');
          err.payload = st;
          err.status = st.code || 500;
          toast(err.message, 'err');
          throw err;
        }
        if (Date.now() - startedAt > 45 * 60 * 1000) {
          throw new Error('训练任务超时（超过 45 分钟），请稍后重试');
        }
      }
    } finally {
      setLoading(false);
    }
  }

  /** 上传文件（带进度条回调） */
  function upload(path, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', API_BASE + path);
      const tk = getToken();
      if (tk) xhr.setRequestHeader('X-API-Token', tk);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else { toast(data.message || '上传失败', 'err'); reject(new Error(data.message || '上传失败')); }
      };
      xhr.onerror = () => { toast('无法连接后端服务：' + API_BASE, 'err'); reject(new Error('网络错误')); };
      xhr.send(formData);
    });
  }

  /**
   * 页内确认对话框（Promise<boolean>）。
   * 预览面板等内嵌 WebView 会静默拦截原生 window.confirm（不弹窗、直接返回假值），
   * 因此涉及「继续/取消」的确认一律用此 DOM 弹窗。
   */
  function confirmDlg(message) {
    return new Promise((resolve) => {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'position:fixed;inset:0;background:rgba(15,23,42,.45);z-index:9999;' +
        'display:flex;align-items:center;justify-content:center;';
      const box = document.createElement('div');
      box.style.cssText = 'max-width:440px;width:86%;background:var(--bg);border:1px solid var(--line);' +
        'border-radius:10px;padding:18px 20px;box-shadow:0 10px 32px rgba(0,0,0,.28);' +
        'color:var(--text);font-size:13px;line-height:1.8;white-space:pre-line;';
      const msg = document.createElement('div');
      msg.textContent = message;
      const btns = document.createElement('div');
      btns.style.cssText = 'display:flex;justify-content:flex-end;gap:8px;margin-top:14px;';
      const cancel = document.createElement('button');
      cancel.className = 'btn'; cancel.textContent = '取消';
      const ok = document.createElement('button');
      ok.className = 'btn primary'; ok.textContent = '继续';
      btns.appendChild(cancel); btns.appendChild(ok);
      box.appendChild(msg); box.appendChild(btns); wrap.appendChild(box);
      document.body.appendChild(wrap);
      const done = (v) => { wrap.remove(); resolve(v); };
      ok.addEventListener('click', () => done(true));
      cancel.addEventListener('click', () => done(false));
      wrap.addEventListener('click', (e) => { if (e.target === wrap) done(false); });
    });
  }

  // p 值展示：null/非数 → '-'；p<0.001 → 两位有效数字的科学计数（如 1.1×10⁻⁷⁸）；其余三位小数。
  // 科学计数用 Unicode 上标，纯文本环境下（表格/报告/导出）均可安全显示。
  const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function fmtP(p) {
    if (p === null || p === undefined || !isFinite(Number(p))) return '-';
    const v = Number(p);
    if (v === 0) return '0.000';
    if (v < 0.001) {
      const parts = v.toExponential(1).split('e');     // ["1.1", "-78"]
      const exp = String(parts[1]).split('').map(c => SUP[c] || c).join('');
      return `${parts[0]}×10${exp}`;
    }
    return v.toFixed(3);
  }
  // 估计值：点估计 (95%CI 下限–上限)
  function fmtEst(v, ci, nd) {
    const n = nd === undefined ? 3 : nd;
    if (v === null || v === undefined) return '-';
    const base = Number(v).toFixed(n);
    if (ci && ci[0] !== null && ci[1] !== null) {
      return `${base} (${Number(ci[0]).toFixed(n)}–${Number(ci[1]).toFixed(n)})`;
    }
    return base;
  }
  function fmtNum(v, nd) {
    if (v === null || v === undefined) return '-';
    return Number(v).toFixed(nd === undefined ? 3 : nd);
  }

  /**
   * 后端 reason 常以 label 开头（如"逻辑回归（Logistic Regression） 在测试集上指标最优…"），
   * 直接拼"推荐模型：label（reason）"会把 label 重复两遍；此处去重。
   */
  function dedupeLabel(label, reason) {
    const l = String(label || '').trim(), r = String(reason || '').trim();
    let out = !l ? r : (!r ? l : (r.startsWith(l) ? r : `${l}（${r}）`));
    // 调用方会统一补句号，去掉 reason 自带的，避免出现"。。"
    return out.replace(/。+$/, '');
  }

  window.API = { API_BASE, get, post, del, upload, request, postTask, pollTask,
                 toast, esc, fmtP, fmtEst, fmtNum,
                 dedupeLabel, setLoading, setBusyLabel, confirmDlg, getToken, setToken };
  window.esc = esc;
  window.toast = toast;
})();
