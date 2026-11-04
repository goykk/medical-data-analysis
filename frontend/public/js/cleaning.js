// medical-data-analysis/frontend/public/js/cleaning.js
// 数据清洗页：三列紧凑统计（竖分割线）→ 明细表格 → 底部决策栏。

(function () {
  'use strict';
  window.Pages = window.Pages || {};

  let report = null;

  const cell = (cls, val, label, note) => `
    <div class="stat-cell ${cls || ''}">
      <div class="sv">${val}</div>
      <div class="sl">${esc(label)}</div>
      ${note ? `<div class="sn">${esc(note)}</div>` : ''}
    </div>`;

  function render() {
    const body = document.getElementById('cleaningBody');
    const d = APP.dataset;
    if (!d) { body.innerHTML = '<div class="empty">请先完成数据确认</div>'; return; }

    API.setBusyLabel('核查中');
    API.get(`/api/dataset/${d.id}/cleaning-report`).then((rep) => {
      report = rep;
      body.innerHTML = `
        <div class="gt">清洗概览<span class="gt-note">缺失值默认按成对删除处理</span></div>
        <div class="stat-strip bare">
          ${cell(rep.n_duplicated_rows ? 'warn' : 'good', rep.n_duplicated_rows, '重复行', '建议剔除（保留首行）')}
          ${cell(rep.total_missing_rate > 0.05 ? 'warn' : 'good', (rep.total_missing_rate * 100).toFixed(1) + '%', '总体缺失率', '按成对删除处理')}
          ${cell((rep.outliers || []).length ? 'warn' : 'good', (rep.outliers || []).length, '异常字段数', '参考区间 / 1.5×IQR 判定')}
        </div>

        ${missingSection(rep)}
        ${outlierSection(rep)}
        ${excludedSection(rep)}
        ${decisionBar(rep)}
      `;
      bind();
    }).catch(() => {
      body.innerHTML = '<div class="empty">清洗报告获取失败，请返回数据确认页重试。</div>';
    });
  }

  function missingSection(rep) {
    const rows = (rep.missing || []);
    const table = rows.length ? `
      <div class="table-wrap"><table class="grid">
        <thead><tr><th>字段</th><th class="num">缺失数</th><th class="num">缺失率</th></tr></thead>
        <tbody>${rows.map(m => `
          <tr><td class="mono">${esc(m.field)}</td>
              <td class="num">${m.n_missing}</td>
              <td class="num">${(m.missing_rate * 100).toFixed(1)}%</td></tr>`).join('')}</tbody>
      </table></div>` : '<div class="hint">未检测到缺失值。</div>';
    return `
      <div class="gt">缺失值明细<span class="gt-note">${rows.length} 个字段存在缺失　·　${esc(rep.note || '')}</span></div>
      ${table}`;
  }

  function outlierSection(rep) {
    const rows = (rep.outliers || []);
    const table = rows.length ? `
      <div class="table-wrap"><table class="grid">
        <thead><tr><th>字段</th><th class="num">IQR 离群</th>
          <th class="num">低于 / 高于参考区间</th><th>参考区间</th></tr></thead>
        <tbody>${rows.map(o => `
          <tr><td class="mono">${esc(o.field)}</td>
              <td class="num">${o.n_iqr_outlier}</td>
              <td class="num">${o.reference_range ? `${o.n_below_reference} / ${o.n_above_reference}` : '—'}</td>
              <td>${o.reference_range
                ? `<span class="tag gray">${esc(o.reference_range.label)} <span class="mono">${o.reference_range.low}–${o.reference_range.high}</span> ${esc(o.reference_range.unit)}</span>`
                : '<span class="tag gray">无参考区间</span>'}</td></tr>`).join('')}</tbody>
      </table></div>` : '<div class="hint">未检测到超出医学参考区间或 1.5×IQR 的异常值。</div>';
    return `
      <div class="gt">异常值明细<span class="gt-note">医学参考区间与 1.5×IQR 双重判定</span></div>
      ${table}`;
  }

  function excludedSection(rep) {
    const list = rep.excluded_fields || [];
    return `
      <div class="gt">自动排除字段<span class="gt-note">${list.length} 个</span></div>
      ${list.length
        ? `<div class="table-wrap"><table class="grid">
            <thead><tr><th>字段</th><th>排除原因</th></tr></thead>
            <tbody>${list.map(f => `<tr><td class="mono">${esc(f.field)}</td><td>${esc(f.reason || '')}</td></tr>`).join('')}</tbody>
          </table></div>`
        : '<div class="hint">无字段被排除。</div>'}`;
  }

  function decisionBar(rep) {
    const dupIdx = (rep.duplicated_row_index || []).slice(0, 8);
    return `
      <div class="gt">清洗决策<span class="gt-note">${rep.n_duplicated_rows
        ? `重复行号示例：<span class="mono">${dupIdx.join(', ')}</span>`
        : '未检出重复行'}</span></div>
      <div class="card">
        <div class="frm">
          <div class="frm-row">
            <div class="lb">重复行</div>
            <div class="ct">
              <select id="dupAction">
                <option value="1" selected>剔除重复行（保留首行）</option>
                <option value="0">保留全部行</option>
              </select>
            </div>
          </div>
          <div class="frm-row">
            <div class="lb">超参考区间样本</div>
            <div class="ct">
              <select id="outlierAction">
                <option value="keep" selected>保留（仅标注）</option>
                <option value="drop_rows">剔除所在行</option>
              </select>
            </div>
          </div>
        </div>
        <div class="form-actions">
          <button class="btn primary" id="cleanBtn">确认清洗并进入分析配置</button>
          <span class="hint" style="margin:0">清洗后将重新计算各字段缺失率</span>
        </div>
      </div>`;
  }

  function bind() {
    const btn = document.getElementById('cleanBtn');
    if (btn) btn.addEventListener('click', submit);
  }

  function submit() {
    const d = APP.dataset;
    const btn = document.getElementById('cleanBtn');
    if (btn) btn.disabled = true;      // 防双击重复提交：第二次清洗会在已清洗数据上重复执行
    API.setBusyLabel('清洗中');
    API.post(`/api/dataset/${d.id}/clean`, {
      drop_duplicates: document.getElementById('dupAction').value === '1',
      drop_columns: [], drop_rows_missing_in: [],
      outlier_action: document.getElementById('outlierAction').value,
    }).then((res) => {
      APP.cleaning = res.cleaning;
      APP.dataset.variables = res.variables;
      APP.dataset.n_rows = res.n_rows;
      APP.refreshTopbar();
      const acts = (res.cleaning.actions || []).map(a => a.detail);
      toast(acts.length ? '清洗完成：' + acts.join('；') : '清洗完成：未执行额外处理', 'ok');
      APP.markDone('cleaning', true);
      APP.go('config');
    }).catch(() => { if (btn) btn.disabled = false; });
  }

  window.Pages.cleaning = { onShow: render };
})();
