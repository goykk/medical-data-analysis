// medical-data-analysis/frontend/public/js/confirm.js
// 数据确认页：字段清单表格（类型下拉内嵌），顶部单行校验结论。

(function () {
  'use strict';
  window.Pages = window.Pages || {};

  const TYPE_LABEL = {
    numeric: '数值型', categorical: '分类型', datetime: '日期型',
    text: '文本型', id: 'ID', constant: '常量',
  };

  function renderAlert() {
    const box = document.getElementById('confirmAlert');
    const v = (APP.dataset && APP.dataset.validation) || {};
    if (v.is_medical) {
      box.innerHTML = `<div class="vline ok"><span>✓</span><span>医疗数据校验通过　${esc(v.message)}</span></div>`;
    } else {
      box.innerHTML = `<div class="vline warn"><span>▲</span><span>${esc(v.message)}</span>
        <span class="acts">
          <button class="btn link" id="vTerminate">终止分析</button>
          <button class="btn link" id="vContinue">确认继续</button>
        </span></div>`;
      const t = document.getElementById('vTerminate'), c = document.getElementById('vContinue');
      if (t) t.addEventListener('click', terminate);
      if (c) c.addEventListener('click', () => submit(true));
    }
  }

  function renderTable() {
    const d = APP.dataset;
    const tbl = document.getElementById('varTable');
    const cnt = document.getElementById('varCount');

    const rows = d.variables.map((f, i) => `
      <tr>
        <td class="num" style="color:var(--text-mute)">${i + 1}</td>
        <td class="mono">${esc(f.name)}</td>
        <td class="mini editable">
          <select class="sel-inline" data-idx="${i}" ${f.excluded ? 'disabled' : ''}>
            ${Object.keys(TYPE_LABEL).map(t =>
              `<option value="${t}"${f.confirmed_type === t ? ' selected' : ''}>${TYPE_LABEL[t]}</option>`).join('')}
          </select>
        </td>
        <td class="num">${f.n_unique}</td>
        <td class="num">${(f.missing_rate * 100).toFixed(1)}%</td>
        <td>${esc((f.sample_values || []).join(' / '))}</td>
        <td>${f.reference_range
          ? `<span class="tag blue">${esc(f.reference_range.label)} <span class="mono">${f.reference_range.low}–${f.reference_range.high}</span> ${esc(f.reference_range.unit)}</span>`
          : '<span class="tag gray">无</span>'}</td>
        <td>${/* 标签只标例外：纳入且缺失正常的字段留空；仅"已排除""高缺失"显示灰色标签 */
          f.excluded
            ? `<span class="tag gray" title="${esc(f.exclude_reason || '后端未给出排除原因')}">已排除</span>`
            : (f.high_missing
              ? `<span class="tag gray" title="缺失 ${f.n_missing} 个值，缺失率 ${(f.missing_rate * 100).toFixed(1)}%，是否纳入由你决定">高缺失</span>`
              : '')}</td>
      </tr>`).join('');

    tbl.innerHTML = `
      <thead><tr>
        <th class="num">#</th><th>字段名</th><th>类型</th><th class="num">唯一值数</th>
        <th class="num">缺失率</th><th>示例值</th><th>医学参考区间</th><th>状态</th>
      </tr></thead><tbody>${rows}</tbody>`;

    tbl.querySelectorAll('select').forEach(sel => {
      sel.addEventListener('change', () => {
        APP.dataset.variables[Number(sel.dataset.idx)].confirmed_type = sel.value;
      });
    });

    const total = d.variables.length;
    const keep = d.variables.filter(v => !v.excluded).length;
    if (cnt) cnt.textContent = `共 ${total} 个字段 · 纳入 ${keep} · 排除 ${total - keep}`;
  }

  function render() {
    const d = APP.dataset;
    if (!d) { document.getElementById('varTable').innerHTML = ''; return; }
    renderAlert();
    renderTable();
  }

  function showModal(message) {
    document.getElementById('validateText').textContent = message;
    document.getElementById('validateModal').classList.add('on');
  }
  function hideModal() { document.getElementById('validateModal').classList.remove('on'); }

  function terminate() {
    hideModal();
    toast('已终止分析，请上传医疗数据集', 'warn');
    APP.go('upload');
  }

  function submit(force) {
    const d = APP.dataset;
    if (!d) { toast('请先上传数据集', 'warn'); return; }
    const btn = document.getElementById('confirmBtn');
    if (btn) btn.disabled = true;
    const types = {};
    d.variables.forEach(f => { if (!f.excluded) types[f.name] = f.confirmed_type; });
    API.setBusyLabel('校验中');
    API.post(`/api/dataset/${d.id}/confirm`, { types, force_continue: !!force })
      .then((res) => {
        APP.dataset.variables = res.variables;
        APP.forceContinue = !!force;
        hideModal();
        toast('字段类型已确认', 'ok');
        APP.markDone('confirm', true);
        APP.go('cleaning');
      }).catch(() => { if (btn) btn.disabled = false; });
  }

  window.Pages.confirm = {
    init() {
      document.getElementById('confirmBtn').addEventListener('click', () => {
        const v = APP.dataset && APP.dataset.validation;
        if (v && !v.is_medical && !APP.forceContinue) { showModal(v.message); return; }
        submit(APP.forceContinue);
      });
      document.getElementById('terminateBtn').addEventListener('click', terminate);
      document.getElementById('continueBtn').addEventListener('click', () => submit(true));
    },
    onShow() {
      if (!APP.dataset) { APP.go('upload'); return; }
      render();
    },
  };
})();
