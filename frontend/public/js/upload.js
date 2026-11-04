// medical-data-analysis/frontend/public/js/upload.js
// 数据上传页：紧凑拖拽条 + 最近文件表格（内置示例与用户文件同等呈现）+ 解析结果元信息。

(function () {
  'use strict';
  window.Pages = window.Pages || {};

  const LS_KEY = 'mdp.recent.v1';
  const TYPE_LABEL = {
    numeric: '数值型', categorical: '分类型', datetime: '日期型',
    text: '文本型', id: 'ID', constant: '常量',
  };
  const TYPE_COLOR = {
    numeric: 'var(--primary)', categorical: '#7A9DB0', datetime: '#6B7280',
    text: '#8A6A3B', id: '#C3CCCF', constant: '#C3CCCF',
  };

  let samples = [];   // 内置示例数据（后端）
  let recent = [];    // 用户文件（本地记录）

  /** 本地记录保留条数：来自设置页，默认 20 条 */
  function keepLimit() {
    const n = Number(localStorage.getItem('mdp.keepRecent') || 20);
    return (n > 0 && n <= 50) ? n : 20;
  }

  function loadRecent() {
    try { recent = JSON.parse(localStorage.getItem(LS_KEY) || '[]'); } catch (e) { recent = []; }
  }
  function saveRecent() {
    recent = recent.slice(0, keepLimit());
    try { localStorage.setItem(LS_KEY, JSON.stringify(recent)); } catch (e) { /* 隐私模式下忽略 */ }
  }

  const stamp = (ts) => Theme.stampText(new Date(ts));

  // ---------------- 上传 ----------------
  function doUpload(file) {
    const bar = document.getElementById('progressBar');
    const wrap = document.getElementById('progress');
    wrap.hidden = false; bar.style.width = '0%';
    const fd = new FormData();
    fd.append('file', file);
    API.setBusyLabel('解析中');
    API.upload('/api/dataset/upload', fd, (r) => { bar.style.width = Math.round(r * 90) + '%'; })
      .then((data) => {
        bar.style.width = '100%';
        setTimeout(() => { wrap.hidden = true; }, 300);
        onLoaded(data, { kind: 'upload', name: data.name });
      })
      .catch(() => { wrap.hidden = true; });
  }

  function onLoaded(payload, meta) {
    APP.setDataset(payload);
    upsert(Object.assign({
      kind: meta && meta.kind ? meta.kind : 'upload',
      key: meta && meta.key, id: payload.dataset_id,
      name: payload.name, rows: payload.n_rows, cols: payload.n_cols,
      time: Date.now(),
    }, {}));
    toast(`已加载 ${payload.name}：${payload.n_rows} 行 × ${payload.n_cols} 列`, 'ok');
    renderMeta();
    APP.go('confirm');
  }

  /** 记录最近文件：同名覆盖旧条目 */
  function upsert(item) {
    const i = recent.findIndex(r => (item.id && r.id === item.id) || (item.key && r.key === item.key) || r.name === item.name);
    if (i >= 0) recent.splice(i, 1, Object.assign({}, recent[i], item));
    else recent.unshift(item);
    saveRecent();
    renderTable();
  }

  // ---------------- 最近文件表格 ----------------
  function loadSampleByKey(key) {
    API.setBusyLabel('载入中');
    API.post('/api/sample/load', { key })
      .then(res => onLoaded(res, { kind: 'sample', key }))
      .catch(() => {});
  }

  function reloadRow(idx) {
    const r = recent[idx];
    if (!r) return;
    if (r.kind === 'sample' && r.key) { loadSampleByKey(r.key); return; }
    if (!r.id) { toast('该文件需重新选择本地文件', 'warn'); document.getElementById('fileInput').click(); return; }
    // 后端仍在内存中则直接复用；会话重启后数据集已失效
    API.setBusyLabel('读取中');
    API.get(`/api/dataset/${r.id}`)
      .then((res) => {
        if (!res || !res.variables) { toast('会话已失效，请重新上传该文件', 'warn'); return; }
        APP.setDataset(res);
        r.time = Date.now(); saveRecent(); renderTable(); renderMeta();
        toast(`已重新载入 ${res.name}`, 'ok');
        APP.go('confirm');
      })
      .catch(() => { toast('会话已失效，请重新上传该文件', 'warn'); });
  }

  function removeRow(idx) {
    const btn = document.querySelector(`[data-remove="${idx}"]`);
    // 两步确认：第一次点击进入待命态（按钮变红），3 秒内再点才真删，避免误触
    if (!btn || btn.dataset.armed !== '1') {
      if (btn) {
        btn.dataset.armed = '1';
        btn.classList.add('danger');
        btn.textContent = '确认删除';
        setTimeout(() => {
          if (btn.dataset.armed === '1') {
            btn.dataset.armed = '';
            btn.classList.remove('danger');
            btn.textContent = '删除';
          }
        }, 3000);
      }
      return;
    }
    recent.splice(idx, 1);
    saveRecent();
    renderTable();
  }

  function renderTable() {
    const tbl = document.getElementById('recentTable');
    const cnt = document.getElementById('fileCount');
    if (!tbl) return;

    const sampleRows = samples.map(s => `
      <tr>
        <td class="mono">${esc(s.name)}</td>
        <td class="num">${s.rows !== null && s.rows !== undefined ? `${s.rows} × ${s.cols}` : '—'}</td>
        <td class="num">—</td>
        <td>${s.available ? '<span class="tag blue">内置示例</span>' : '<span class="tag gray">文件缺失</span>'}</td>
        <td>${s.available
          ? `<button class="btn link" data-load-sample="${esc(s.key)}">载入分析</button>`
          : '<span class="tag gray">不可用</span>'}</td>
      </tr>`).join('');

    const userRows = recent.map((r, i) => `
      <tr>
        <td class="mono">${esc(r.name)}</td>
        <td class="num">${r.rows} × ${r.cols}</td>
        <td class="num">${esc(stamp(r.time))}</td>
        <td>${r.kind === 'sample' ? '<span class="tag blue">内置示例</span>' : '<span class="tag gray">已上传</span>'}</td>
        <td class="act">
          <button class="btn link" data-reload="${i}">重新分析</button>
          <button class="btn link" data-remove="${i}">删除</button>
        </td>
      </tr>`).join('');

    const rows = sampleRows + userRows || `<tr><td colspan="5" class="empty-cell">还没有本地文件记录，上传或载入内置示例后会按时间显示在这里。</td></tr>`;
    tbl.innerHTML = `
      <thead><tr>
        <th>文件名</th><th class="num">行 × 列</th><th class="num">上传时间</th>
        <th>来源</th><th class="act">操作</th>
      </tr></thead>
      <tbody>${rows}</tbody>`;

    if (cnt) cnt.textContent = `内置 ${samples.length} 项 · 本地记录 ${recent.length} 项`;

    tbl.querySelectorAll('[data-load-sample]').forEach(b => {
      b.addEventListener('click', () => loadSampleByKey(b.dataset.loadSample));
    });
    tbl.querySelectorAll('[data-reload]').forEach(b => {
      b.addEventListener('click', () => reloadRow(Number(b.dataset.reload)));
    });
    tbl.querySelectorAll('[data-remove]').forEach(b => {
      b.addEventListener('click', () => removeRow(Number(b.dataset.remove)));
    });
  }

  // ---------------- 解析结果元信息 ----------------
  /** 最近分析记录：取自后端 /api/history，读不到就如实说明 */
  function recentAnalysisHtml() {
    API.get('/api/history').then((res) => {
      const items = (res && res.items) || [];
      const box = document.getElementById('recentAnalysis');
      if (!box) return;
      if (!items.length) {
        box.innerHTML = '<div class="hint">后端暂无分析记录。完成一次分析后，此处按时间倒序列出。</div>';
        return;
      }
      box.innerHTML = `<div class="table-wrap"><table class="grid">
        <thead><tr><th>数据集</th><th class="num">样本量</th><th>结局变量</th><th class="num">记录时间</th></tr></thead>
        <tbody>${items.slice(0, 8).map(it => `<tr>
          <td class="mono">${esc(it.dataset_name || '—')}</td>
          <td class="num">${it.n_rows === null || it.n_rows === undefined ? '—' : it.n_rows}</td>
          <td class="mono">${esc(it.outcome || '—')}</td>
          <td class="num">${esc(it.created_at || '—')}</td>
        </tr>`).join('')}</tbody></table></div>`;
    }).catch(() => {
      const box = document.getElementById('recentAnalysis');
      if (box) box.innerHTML = '<div class="hint">分析记录读取失败。</div>';
    });
    return '<div id="recentAnalysis"><div class="hint">读取分析记录…</div></div>';
  }

  /** 空态：不放大片留白，填「快速开始」三步指引 + 最近分析记录 */
  function emptyMetaHtml() {
    const steps = [
      ['01', '上传数据', '拖入或选择 csv / xlsx / sav / dta 文件，文本类自动检测编码。'],
      ['02', '确认字段', '核对每个字段的类型并确认；ID 列与常量列会自动排除。'],
      ['03', '执行分析', '指定结局变量后运行，得到统计检验、模型评估与可导出报告。'],
    ].map(([n, t, d]) => `
      <div class="qs-step">
        <div class="qs-n">${n}</div><div class="qs-t">${t}</div>
        <div class="qs-d">${d}</div>
      </div>`).join('');

    return `
      <div class="card">
        <div class="qs">${steps}</div>
        <div class="hint">内置示例数据可直接载入，无需准备文件。</div>
      </div>
      <div class="gt">最近分析记录<span class="gt-note">来自后端历史</span></div>
      ${recentAnalysisHtml()}`;
  }

  function renderMeta() {
    const box = document.getElementById('metaBox');
    if (!box) return;
    const d = APP.dataset;
    if (!d) { box.innerHTML = emptyMetaHtml(); return; }
    const pm = d.parse_meta || {};
    const vars = d.variables || [];
    const counts = {};
    vars.forEach(v => {
      const t = v.confirmed_type || 'unknown';
      counts[t] = (counts[t] || 0) + 1;
    });
    const total = vars.length || 1;
    const order = Object.keys(TYPE_LABEL).filter(k => counts[k]);

    const bar = order.map(t =>
      `<div style="width:${(counts[t] / total) * 100}%;background:${TYPE_COLOR[t] || '#9CA3AF'};height:100%"></div>`).join('');
    const legend = order.map(t =>
      `<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px">
        <span style="width:8px;height:8px;display:inline-block;background:${TYPE_COLOR[t] || '#9CA3AF'}"></span>
        ${TYPE_LABEL[t]} <span class="mono">${counts[t]}</span></span>`).join('');

    const info = [
      ['格式', pm.format || '—'],
      ['编码', pm.encoding || '—'],
      ['有效字段', vars.filter(v => !v.excluded).length],
      ['排除字段', vars.filter(v => v.excluded).length],
    ].map(([k, v]) => `
      <div class="stat-cell">
        <div class="sv" style="font-size:15px">${esc(String(v))}</div>
        <div class="sl">${esc(k)}</div>
      </div>`).join('');

    box.innerHTML = `
      <div class="stat-strip">${info}</div>
      <div class="gt">列类型分布<span class="gt-note">共 ${vars.length} 个字段</span></div>
      <div style="height:6px;display:flex;border:1px solid var(--line);border-radius:2px;overflow:hidden;background:#fff">${bar}</div>
      <div style="margin-top:6px;font-size:12px;color:var(--text-dim)">${legend}</div>
      ${pm.encoding === 'binary'
        ? '<div class="hint">二进制表格文件（xlsx / xls / sav / dta）按原格式解析，不经过字符编码检测。</div>'
        : ''}
      ${pm.skipped_rows
        ? `<div class="alert warn">${esc(pm.note || `检测到 ${pm.skipped_rows} 行字段数与表头不一致，已跳过`)}</div>`
        : ''}`;
  }

  function loadSamples() {
    API.get('/api/sample/list').then((res) => {
      samples = res.samples || [];
      renderTable();
    }).catch(() => { samples = []; renderTable(); });
  }

  window.Pages.upload = {
    init() {
      const dz = document.getElementById('dropzone');
      const fi = document.getElementById('fileInput');
      loadRecent();

      dz.addEventListener('click', () => fi.click());
      fi.addEventListener('change', () => { if (fi.files[0]) doUpload(fi.files[0]); fi.value = ''; });
      ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, (e) => {
        e.preventDefault(); e.stopPropagation(); dz.classList.add('over');
      }));
      ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, (e) => {
        e.preventDefault(); e.stopPropagation(); dz.classList.remove('over');
      }));
      dz.addEventListener('drop', (e) => {
        const f = e.dataTransfer && e.dataTransfer.files[0];
        if (f) doUpload(f);
      });

      API.get('/api/config').then(c => {
        const exts = c.allowed_extensions || [];
        document.getElementById('dzHint').textContent =
          `支持 ${exts.length} 种格式：${exts.join(' ')} ｜ 单个文件 ≤ ${c.max_upload_mb} MB ｜ 文本类自动检测编码`;
        if (exts.length) fi.setAttribute('accept', exts.join(','));
      }).catch(() => {});

      loadSamples();
      renderMeta();
    },
    onShow() { loadSamples(); renderMeta(); },
    // 供设置页清空记录后调用，避免列表停留在旧状态
    refresh() { loadRecent(); renderTable(); },
  };
})();
