// medical-data-analysis/frontend/public/js/automl.js
// 智能建模页：接 /api/automl/* 后端引擎。
// 自动选最优模型（analyze）+ 手动指定模型训练（train），结果统一呈现排行榜 + 冠军详解。
(function () {
  'use strict';

  let MODELS = [];        // 后端模型清单缓存
  let lastReport = null;  // 最近一次报告，供导出/复用

  const esc = (v) => API.esc(v === null || v === undefined ? '' : String(v));

  function variables() {
    const d = APP.dataset;
    if (!d || !d.variables) return [];
    return d.variables.filter(v => !v.excluded &&
      (v.confirmed_type === 'categorical' || v.confirmed_type === 'numeric'));
  }

  async function ensureModels() {
    if (MODELS.length) return MODELS;
    try {
      const r = await API.get('/api/automl/models');
      MODELS = (r && r.models) || [];
    } catch (e) { MODELS = []; }
    return MODELS;
  }

  function render() {
    const host = document.getElementById('automlBody');
    if (!host) return;
    if (!APP.dataset) {
      host.innerHTML = '<div class="empty">请先上传并确认数据集</div>';
      return;
    }
    const vs = variables();
    const targetOpts = ['<option value="">（不选 → 无标签聚类）</option>']
      .concat(vs.map(v => `<option value="${esc(v.name)}">${esc(v.name)}（${v.confirmed_type === 'categorical' ? '分类' : '数值'}）</option>`))
      .join('');

    host.innerHTML = `
      <div class="gt">智能建模<span class="gt-note">自动选最优 / 手动选模型 · 共享同一流水线</span></div>
      <div class="card">
        <div class="frm">
          <div class="frm-row">
            <div class="lb">结局变量</div>
            <div class="ct"><select id="amlTarget">${targetOpts}</select></div>
          </div>
          <div class="frm-row">
            <div class="lb">模型（手动）</div>
            <div class="ct"><select id="amlModel"><option value="">加载中…</option></select></div>
          </div>
        </div>
        <div class="form-actions">
          <button class="btn primary" id="amlAuto">自动选最优模型</button>
          <button class="btn" id="amlManual">用所选模型训练</button>
        </div>
        <div class="hint">泄露列自动剔除；患者 ID 自动作 GroupKFold 分组键；「随访时间 + 事件」结构自动按生存分析处理；结果可复现（统一随机种子）。</div>
      </div>
      <div id="amlResult"></div>`;

    document.getElementById('amlAuto').addEventListener('click', runAuto);
    document.getElementById('amlManual').addEventListener('click', runManual);
    fillModels();
  }

  async function fillModels() {
    const sel = document.getElementById('amlModel');
    const ms = await ensureModels();
    sel.innerHTML = '<option value="">请选择模型…</option>' + ms.map(m =>
      `<option value="${esc(m.key)}"${m.available === false ? ' disabled' : ''}>` +
      `${esc(m.label)}${m.available === false ? '（依赖缺失）' : ''}</option>`).join('');
  }

  function targetValue() {
    return document.getElementById('amlTarget').value;
  }

  function setBusy(busy) {
    document.getElementById('amlAuto').disabled = busy;
    document.getElementById('amlManual').disabled = busy;
  }

  async function runAuto() {
    setBusy(true);
    try {
      const report = await API.post('/api/automl/analyze', {
        dataset_id: APP.dataset.id, target: targetValue() || null,
      });
      lastReport = report;
      renderResult(report);
    } catch (e) { /* API 已 toast */ }
    setBusy(false);
  }

  async function runManual() {
    const key = document.getElementById('amlModel').value;
    if (!key) { API.toast('请先选择要训练的模型', 'warn'); return; }
    setBusy(true);
    try {
      const report = await API.post('/api/automl/train', {
        dataset_id: APP.dataset.id, target: targetValue() || null, model_key: key,
      });
      lastReport = report;
      renderResult(report);
    } catch (e) { /* API 已 toast */ }
    setBusy(false);
  }

  function taskLabel(t) {
    return ({ classification: '分类', regression: '回归', clustering: '聚类', survival: '生存分析' }[t] || t);
  }

  function renderResult(r) {
    const host = document.getElementById('amlResult');
    if (!host) return;
    const p = r.profile || {};
    const lb = r.leaderboard || [];
    const ch = r.champion;
    const lims = r.limitations || [];

    const rows = lb.map((m, i) => `
      <tr>
        <td class="num">${i + 1}</td>
        <td class="mono">${esc(m.model_key)}</td>
        <td>${esc(m.label)}</td>
        <td class="num">${m.status === 'ok' ? esc(m.metric_value) + (m.ci_low != null ? `<span class="dim"> [${esc(m.ci_low)}, ${esc(m.ci_high)}]</span>` : '') : '<span class="tag red">失败</span>'}</td>
        <td class="num">${m.status === 'ok' ? esc(m.fit_seconds) + 's' : '—'}</td>
      </tr>`).join('');

    const impRows = ch && ch.importance_top15 ? ch.importance_top15.map(t => `
      <tr><td class="mono">${esc(t.variable)}</td><td class="num">${esc(t.importance)}</td></tr>`).join('') : '';

    // 推荐模型的交叉验证平均混淆矩阵：赛马时逐折记录，均值矩阵便于看假阴性/假阳性
    const top = lb[0];
    const topMetrics = top && top.status === 'ok' ? (top.metrics || {}) : {};
    const cmMean = topMetrics.confusion_matrix_mean;
    const cmLabels = topMetrics.classes || [];
    const cmTable = p.task_type === 'classification' && cmMean && cmMean.length && cmMean.length <= 10
      ? `<div class="gt">推荐模型混淆矩阵<span class="gt-note">${esc(top.label)} · 交叉验证各折均值（行=真实，列=预测）</span></div>
         <div class="card"><div class="table-wrap"><table class="grid">
           <thead><tr><th>真实 \\ 预测</th>${cmLabels.map(l => `<th class="num">${esc(l)}</th>`).join('')}</tr></thead>
           <tbody>${cmMean.map((row, i) => `<tr><td class="mono">${esc(cmLabels[i] || i)}</td>
             ${row.map(v => `<td class="num">${(+v).toFixed(1)}</td>`).join('')}</tr>`).join('')}</tbody>
         </table></div>
         <div class="hint">对角线为正确分类；纵向看漏判（假阴性），横向看误判（假阳性）。逐折矩阵见 metrics.confusion_matrices。</div>
         </div>`
      : '';

    host.innerHTML = `
      <div class="gt">任务<span class="gt-note">${esc(taskLabel(p.task_type))} · ${esc(p.task_reason || '')}</span></div>
      <div class="card">
        <div class="cc-head"><div><div class="cc-title">模型排行榜</div></div></div>
        <div class="table-wrap"><table class="grid">
          <thead><tr><th class="num">#</th><th>模型</th><th>名称</th><th class="num">指标${p.task_type === 'regression' ? '（RMSE）' : p.task_type === 'survival' ? '（C-index）' : p.task_type === 'clustering' ? '（轮廓系数）' : '（AUC）'}</th><th class="num">耗时</th></tr></thead>
          <tbody>${rows}</tbody>
        </table></div>
      </div>
      ${cmTable}
      ${ch ? `
      <div class="gt">冠军模型<span class="gt-note">${esc(ch.label)}</span></div>
      <div class="card">
        <p class="interp">${esc(ch.interpretation)}</p>
        ${ch.shap ? `<div class="hint">SHAP：${esc(ch.shap.method)}${ch.shap.warning ? ' · ' + esc(ch.shap.warning) : ''}</div>` : ''}
        ${impRows ? `<div class="gt">特征重要性 Top${ch.importance_top15.length}</div>
          <div class="table-wrap"><table class="grid">
            <thead><tr><th>特征</th><th class="num">归一化重要性</th></tr></thead>
            <tbody>${impRows}</tbody></table></div>` : ''}
      </div>` : '<div class="card"><div class="empty">未产出冠军模型（候选全部失败或无冠军分析）</div></div>'}
      ${lims.length ? `<div class="gt">局限性与说明</div><div class="card">${lims.map(x => `<div class="alert warn">${esc(x)}</div>`).join('')}</div>` : ''}
      <div class="alert warn"><b>免责声明：</b>${esc(r.disclaimer || '本结果仅供研究参考与辅助决策，不能替代临床医生判断。')}</div>`;
  }

  window.Pages = window.Pages || {};
  window.Pages.automl = { onShow: render };
})();
