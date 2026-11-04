// 侧栏上下文：当前数据集 / 字段概览 / 最近运行。只读，不发分析请求。
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const TYPE = { numeric: '数值', categorical: '分类', datetime: '日期', text: '文本', id: 'ID', constant: '常量' };

  function fmtTime(ts) {
    const m = /^\d{4}-(\d{2})-(\d{2})\s+(\d{2}:\d{2})/.exec(String(ts || ''));
    return m ? `${m[1]}-${m[2]} ${m[3]}` : (ts || '—');
  }

  function stageTags() {
    const st = (window.APP && APP.steps) || {};
    const tags = [];
    if (st.confirm) tags.push('<span class="tag green">已确认</span>');
    if (st.cleaning) tags.push('<span class="tag green">已清洗</span>');
    if (window.APP && APP.model) tags.push('<span class="tag blue">已建模</span>');
    return tags.join('') || '<span class="tag gray">待确认</span>';
  }

  function renderDataset() {
    const d = window.APP && APP.dataset;
    const name = $('dsName'), meta = $('dsMeta'), tags = $('dsTags');
    if (!name) return;
    const body = $('sideBody');
    if (body) body.classList.toggle('no-data', !d);
    if (!d) {
      name.textContent = '尚未载入';
      name.classList.add('none');
      meta.textContent = '拖入 CSV / Excel / SAV / DTA，或载入内置示例';
      tags.innerHTML = '';
      return;
    }
    name.textContent = d.name;
    name.classList.remove('none');
    const active = (d.variables || []).filter(v => !v.excluded).length;
    meta.innerHTML = `<span class="mono">${d.n_rows} × ${d.n_cols}</span> · 有效字段 ${active}`;
    tags.innerHTML = stageTags();
  }

  let samples = null;

  function loadSamples() {
    if (samples) return Promise.resolve(samples);
    return API.get('/api/sample/list')
      .then((r) => (samples = (r && r.samples) || []))
      .catch(() => (samples = []));
  }

  function loadSample(key) {
    API.setBusyLabel('载入中');
    API.post('/api/sample/load', { key })
      .then((res) => {
        APP.setDataset(res);
        if (window.Pages && Pages.upload && Pages.upload.refresh) Pages.upload.refresh();
        toast(`已载入 ${res.name}`, 'ok');
        APP.go('confirm');
      })
      .catch(() => toast('示例载入失败', 'err'));
  }

  function renderFields() {
    const d = window.APP && APP.dataset;
    const box = $('fieldList'), cnt = $('fieldCount'), title = $('fieldTitle');
    if (!box) return;
    const vars = (d && d.variables) || [];
    const active = vars.filter(v => !v.excluded);
    if (cnt) cnt.textContent = d ? `${active.length}/${vars.length}` : '';
    if (!d) {
      if (title) title.textContent = '内置示例';
      loadSamples().then((list) => {
        if (window.APP && APP.dataset) return;
        box.innerHTML = list.length
          ? list.map((s) => `<button type="button" class="sample-row" data-sample="${esc(s.key)}">` +
              `<span class="sr-name">${esc(s.name)}</span>` +
              `<span class="sr-meta">${s.rows}×${s.cols}</span></button>`).join('')
          : '<span class="side-empty">拖入文件开始，或载入内置示例</span>';
        box.querySelectorAll('[data-sample]').forEach((b) => {
          b.addEventListener('click', () => loadSample(b.dataset.sample));
        });
      });
      return;
    }
    if (title) title.textContent = '字段';
    if (!active.length) {
      box.innerHTML = '<span class="side-empty">没有可用字段</span>';
      return;
    }
    box.innerHTML = active.map(v => {
      const t = TYPE[v.confirmed_type] || TYPE[v.inferred_type] || '—';
      const miss = typeof v.missing_rate === 'number' ? v.missing_rate : 0;
      return `<div class="fl-row"><span class="fl-name">${esc(v.name)}</span>` +
        `<span class="fl-meta">${t}${miss > 0 ? ` <b>${(miss * 100).toFixed(1)}%</b>` : ''}</span></div>`;
    }).join('');
  }

  function renderRuns() {
    const box = $('runsBody'), cnt = $('runsCount');
    if (!box) return;
    API.get('/api/history').then((res) => {
      const items = (res && res.items) || [];
      if (cnt) cnt.textContent = items.length ? `${items.length} 条` : '';
      if (!items.length) {
        box.innerHTML = '<span class="side-empty">还没有运行记录<br>在分析配置页运行一次后显示</span>';
        return;
      }
      box.innerHTML = items.slice(0, 3).map((it) => {
        const p = it.payload || {};
        const algo = p.algorithm ? String(p.algorithm).split('（')[0].trim() : '';
        const metric = typeof p.auc === 'number' ? `AUC ${p.auc.toFixed(2)}`
          : (typeof p.r2 === 'number' ? `R² ${p.r2.toFixed(2)}` : '');
        const line = [algo, metric].filter(Boolean).join(' · ') || (it.outcome || '分析');
        return `<div class="runs-item">
          <div class="runs-name">${esc(it.dataset_name || '—')}</div>
          <div class="runs-meta">${esc(line)} · ${esc(fmtTime(it.created_at))}</div>
        </div>`;
      }).join('');
    }).catch(() => { box.innerHTML = '<span class="side-empty">运行记录读取失败。</span>'; });
  }

  function refresh() { renderDataset(); renderFields(); }
  function refreshRuns() { renderRuns(); }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { refresh(); refreshRuns(); });
  } else { refresh(); refreshRuns(); }

  window.Sidebar = { refresh, refreshRuns };
})();
