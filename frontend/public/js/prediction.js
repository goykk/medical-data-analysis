// medical-data-analysis/frontend/public/js/prediction.js
// 新样本预测页：表单左 / 结果右（竖分割线），结果区大号等宽数字、10px 概率条、SHAP 内嵌。

(function () {
  'use strict';
  window.Pages = window.Pages || {};

  const kv = (label, val) => `
    <div class="stat-cell">
      <div class="sv" style="font-size:15px">${esc(String(val))}</div>
      <div class="sl">${esc(label)}</div>
    </div>`;

  function render() {
    const body = document.getElementById('predBody');
    if (!APP.dataset) { body.innerHTML = '<div class="empty">请先上传并完成建模</div>'; return; }
    const did = APP.dataset.id;

    API.setBusyLabel('读取模型');
    API.get(`/api/ml/model/${did}`).then((info) => {
      if (!info.trained) {
        const disc = info.discarded;
        const meta = disc
          ? `<div style="margin-top:6px;color:var(--text-dim)">训练时间 ${esc(disc.time || '')}${disc.n_total ? ` · 样本 n=${esc(String(disc.n_total))}` : ''}</div>`
          : '';
        body.innerHTML = `<div class="empty">
          <div class="alert warn" style="text-align:left">${esc(info.message || '请先完成当前数据集的建模')}${meta}</div>
          <div style="margin-top:10px"><button class="btn primary" id="gotoConfig">前往分析配置</button></div></div>`;
        const b = document.getElementById('gotoConfig');
        if (b) b.addEventListener('click', () => APP.go('config'));
        return;
      }
      APP.model = info;
      const noPredict = info.predictive === false;
      body.innerHTML = `
        ${noPredict ? `<div class="alert warn">该模型测试集 R²<0（不如直接用均值预测），已禁用新样本预测。请更换特征列或改用其他模型重新训练。</div>` : ''}
        <div class="gt">模型信息<span class="gt-note">指纹 ${esc(Theme.fingerprint(did))}</span></div>
        <div class="stat-strip">
          ${kv('最优模型', info.best_model.label)}
          ${kv('任务', info.task === 'classification' ? '分类' : '回归')}
          ${kv('结局变量', info.target)}
          ${kv('训练集 n', info.n_train)}
          ${kv('测试集 n', info.n_test)}
          ${kv('特征数', (info.features || []).length)}
          ${kv('建模时间', info.created_at)}
        </div>

        <div class="gt">录入新病例指标<span class="gt-note">留空由 Pipeline 填补</span></div>
        <div class="card">
          <div class="pred-split">
            <div class="pred-left">
              <div class="form-scroll"><div class="form-grid" id="predForm"></div></div>
              <div class="form-actions">
                <button class="btn" id="fillDemo">填入示例值</button>
                <button class="btn primary" id="predictBtn">开始预测</button>
              </div>
            </div>
            <div class="pred-right">
              <div id="predResult"><div class="empty">录入指标后点击「开始预测」，结果在此显示。</div></div>
            </div>
          </div>
        </div>`;
      buildForm(info);
      if (noPredict) {
        const b = document.getElementById('predictBtn');
        if (b) { b.disabled = true; b.textContent = '模型不具备预测能力'; }
      }
    }).catch(() => {
      body.innerHTML = '<div class="empty">模型信息读取失败，请先完成建模。</div>';
    });
  }

  function buildForm(info) {
    const form = document.getElementById('predForm');
    form.innerHTML = (info.features || []).map(f => {
      const isNum = (info.numeric_features || []).indexOf(f) >= 0;
      if (isNum) {
        const rg = (info.numeric_ranges || {})[f];
        const hint = rg ? `训练集范围 ${rg[0]} – ${rg[1]}` : '可留空（由 Pipeline 填补）';
        return `<label>${esc(f)}
          <input type="number" step="any" data-f="${esc(f)}" placeholder="${esc(hint)}" /></label>`;
      }
      const opts = ((info.categories || {})[f] || []).map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
      return `<label>${esc(f)}
        <select data-f="${esc(f)}"><option value="">（留空）</option>${opts}</select></label>`;
    }).join('');

    document.getElementById('fillDemo').addEventListener('click', () => {
      API.post('/api/analysis/describe', { dataset_id: APP.dataset.id }).then((d) => {
        (d.numeric || []).forEach(x => {
          const el = form.querySelector(`[data-f="${CSS.escape(x.field)}"]`);
          if (el && x.mean !== null) el.value = x.mean;
        });
        (d.categorical || []).forEach(x => {
          const el = form.querySelector(`[data-f="${CSS.escape(x.field)}"]`);
          if (el && x.levels && x.levels.length) el.value = x.levels[0].level;
        });
        toast('已填入各指标的均值 / 常见水平', 'ok');
      });
    });

    document.getElementById('predictBtn').addEventListener('click', submit);
  }

  function submit() {
    const form = document.getElementById('predForm');
    const sample = {};
    form.querySelectorAll('[data-f]').forEach(el => {
      const v = el.value;
      if (v === '' || v === null) return;
      sample[el.dataset.f] = (el.tagName === 'SELECT' || isNaN(Number(v))) ? v : Number(v);
    });
    const btn = document.getElementById('predictBtn');
    btn.disabled = true; btn.textContent = '预测中…';
    API.setBusyLabel('预测中');
    API.post('/api/ml/predict', { dataset_id: APP.dataset.id, sample })
      .then((r) => { showResult(r); btn.disabled = false; btn.textContent = '开始预测'; })
      .catch(() => { btn.disabled = false; btn.textContent = '开始预测'; });
  }

  function showResult(r) {
    // 预测真正跑出结果才算完成本步（只打开页面不算）
    APP.markDone('prediction', true);
    const box = document.getElementById('predResult');
    const warns = (r.warnings || []).map(w => `<div class="alert warn">${esc(w)}</div>`).join('');
    let main = '';

    if (r.task === 'classification') {
      const pos = r.positive_probability !== null && r.positive_probability !== undefined
        ? r.positive_probability : 0;
      const classes = Object.keys(r.probabilities || {});
      const isPos = r.predicted_class === r.positive_class;
      main = `
        <div class="res-grid">
          <div>
            <div class="big-num ${isPos ? 'bad' : 'good'}">${esc(r.predicted_class)}</div>
            <div class="sl" style="font-size:12px;color:var(--text-dim)">预测类别 · ${esc(r.target)}</div>
          </div>
          <div>
            <div class="big-num">${(pos * 100).toFixed(1)}%</div>
            <div class="sl" style="font-size:12px;color:var(--text-dim)">阳性概率（${esc(r.positive_class)}）</div>
          </div>
        </div>
        <div class="prob-bar">
          <div class="prob-neg" style="width:${(1 - pos) * 100}%"></div>
          <div class="prob-pos" style="width:${pos * 100}%"></div>
        </div>
        <div class="prob-legend">
          <span>${esc(classes[0] || '')} ${((1 - pos) * 100).toFixed(1)}%</span>
          <span>${esc(classes[1] || '')} ${(pos * 100).toFixed(1)}%</span>
        </div>`;
    } else {
      main = `
        <div class="res-grid">
          <div>
            <div class="big-num">${esc(r.predicted_value)}</div>
            <div class="sl" style="font-size:12px;color:var(--text-dim)">预测值 · ${esc(r.target)}</div>
          </div>
        </div>`;
    }

    // SHAP 图是后端返回的 base64：先做字符白名单校验，防止属性逃逸注入
    const b64 = (typeof r.shap_force_plot === 'string' && /^[A-Za-z0-9+/=\r\n]+$/.test(r.shap_force_plot))
      ? r.shap_force_plot : null;
    box.innerHTML = `
      ${warns}
      ${main}
      ${b64
        ? `<div class="gt" style="margin-top:12px">单样本归因<span class="gt-note">后端 matplotlib 生成</span></div>
           <img class="shap-img" src="data:image/png;base64,${b64}" alt="SHAP 力图" />`
        : (r.shap_error ? `<div class="alert err" style="margin-top:8px">${esc(r.shap_error)}</div>` : '')}
      <div class="alert warn" style="margin-top:10px">${esc(r.disclaimer || '科研演示用途，不构成临床诊断依据。')}</div>`;
  }

  window.Pages.prediction = { onShow: render };
})();
