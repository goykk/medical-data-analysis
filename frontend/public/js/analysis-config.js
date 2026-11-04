// medical-data-analysis/frontend/public/js/analysis-config.js
// 分析配置页：表单式变量指派 + 单行样本量/EPV 摘要（超限时转红并给出建议）。

(function () {
  'use strict';
  window.Pages = window.Pages || {};

  function activeVars() {
    return (APP.dataset.variables || []).filter(v => !v.excluded);
  }

  function render() {
    const body = document.getElementById('configBody');
    if (!APP.dataset) { body.innerHTML = '<div class="empty">请先上传数据集</div>'; return; }
    const vars = activeVars();

    API.setBusyLabel('识别中');
    API.get(`/api/dataset/${APP.dataset.id}/outcome-candidates`).then((res) => {
      const cands = res.candidates || [];
      const suggested = cands.length ? cands[0].field : (vars[0] ? vars[0].name : '');
      if (!APP.analysis.outcome) APP.analysis.outcome = suggested;

      body.innerHTML = `
        <div class="gt">结局变量<span class="gt-note">未指定时不强行训练模型</span></div>
        <div class="card">
          <div class="frm">
            <div class="frm-row">
              <div class="lb">结局变量</div>
              <div class="ct">
                <select id="outcomeSel">
                  ${vars.map(v => `<option value="${esc(v.name)}"${v.name === APP.analysis.outcome ? ' selected' : ''}>${esc(v.name)}（${v.confirmed_type === 'categorical' ? '分类' : '数值'}）</option>`).join('')}
                </select>
              </div>
            </div>
            <div class="frm-row">
              <div class="lb">任务类型</div>
              <div class="ct">
                <select id="taskSel">
                  <option value="auto"${APP.analysis.task === 'auto' ? ' selected' : ''}>自动判定</option>
                  <option value="classification"${APP.analysis.task === 'classification' ? ' selected' : ''}>分类</option>
                  <option value="regression"${APP.analysis.task === 'regression' ? ' selected' : ''}>回归（连续结局）</option>
                </select>
              </div>
            </div>
          </div>
          <div id="outcomeTypeWarn" class="alert warn" hidden></div>
          <div class="hint">智能识别结果：${cands.length
            ? cands.slice(0, 3).map(c => `<span class="tag blue">${esc(c.field)}</span>`).join(' ')
            : '<span class="tag gray">未识别到明显结局字段，请手动指定</span>'}</div>
        </div>

        <div class="gt">变量指派<span class="gt-note">点击变量池中的字段加入 / 移除</span></div>
        <div class="card">
          <div class="frm">
            <div class="frm-row">
              <div class="lb">目标自变量</div>
              <div class="ct"><div style="flex:1"><div class="chip-box" id="predBox"></div><div class="var-pool" id="predPool"></div></div></div>
            </div>
            <div class="frm-row">
              <div class="lb">待校正协变量</div>
              <div class="ct"><div style="flex:1"><div class="chip-box" id="covBox"></div><div class="var-pool" id="covPool"></div></div></div>
            </div>
            <div class="frm-row">
              <div class="lb">建模特征</div>
              <div class="ct"><div style="flex:1"><div class="chip-box" id="featBox"></div><div class="var-pool" id="featPool"></div></div></div>
            </div>
          </div>
        </div>

        <div class="gt">样本量与 EPV</div>
        <div id="sampleAlert"></div>

        <div class="form-actions">
          <button class="btn" id="autoSelBtn">按 FDR 显著自动填入目标自变量</button>
          <button class="btn primary" id="runBtn">开始分析</button>
        </div>`;

      bind(vars);
      refreshChips(vars);
      refreshSummary();
      refreshOutcomeWarn(vars);
    }).catch(() => {
      body.innerHTML = '<div class="empty">结局变量候选获取失败，请返回上一步重试。</div>';
    });
  }

  function bind(vars) {
    document.getElementById('outcomeSel').addEventListener('change', (e) => {
      APP.analysis.outcome = e.target.value; refreshChips(vars); refreshSummary(); refreshOutcomeWarn(vars);
    });
    document.getElementById('taskSel').addEventListener('change', (e) => {
      APP.analysis.task = e.target.value;
    });

    ['pred', 'cov', 'feat'].forEach(kind => {
      const pool = document.getElementById(kind + 'Pool');
      pool.innerHTML = vars.map(v =>
        `<span class="v" data-name="${esc(v.name)}">${esc(v.name)}</span>`).join('');
      pool.querySelectorAll('.v').forEach(el => {
        el.addEventListener('click', () => {
          const name = el.dataset.name;
          const list = kind === 'pred' ? APP.analysis.predictors
            : kind === 'cov' ? APP.analysis.covariates : (APP.analysis.features = APP.analysis.features || []);
          const i = list.indexOf(name);
          if (i >= 0) list.splice(i, 1); else list.push(name);
          refreshChips(vars); refreshSummary();
        });
      });
    });

    document.getElementById('autoSelBtn').addEventListener('click', () => {
      const outcome = APP.analysis.outcome;
      const ov = vars.find(x => x.name === outcome);
      if (ov && ov.confirmed_type === 'numeric') {
        toast('结局变量为数值型，不能作为分组进行组间比较筛选；请选择分类变量作为结局', 'err');
        return;
      }
      API.setBusyLabel('筛选中');
      API.post('/api/analysis/screen', {
        dataset_id: APP.dataset.id, outcome,
        candidates: vars.map(v => v.name).filter(n => n !== outcome),
      }).then((res) => {
        APP.analysis.predictors = (res.rows || []).filter(r => r.significant_fdr).map(r => r.variable);
        if (!APP.analysis.predictors.length) {
          APP.analysis.predictors = (res.rows || []).slice(0, 3).map(r => r.variable);
          toast('无 FDR 显著变量，已按 p 值取前 3 个', 'warn');
        } else {
          toast(`已填入 ${APP.analysis.predictors.length} 个 FDR 显著变量`, 'ok');
        }
        APP.results.screen = res;
        refreshChips(vars); refreshSummary();
      });
    });

    document.getElementById('runBtn').addEventListener('click', run);
  }

  function chipsHtml(list, kind, vars) {
    return list.map(n => {
      const v = vars.find(x => x.name === n);
      return `<span class="chip">${esc(n)}${v ? `<span class="tag gray">${v.confirmed_type === 'categorical' ? '分类' : '数值'}</span>` : ''}<i data-rm="${kind}" data-name="${esc(n)}">✕</i></span>`;
    }).join('');
  }

  function refreshChips(vars) {
    const outcome = APP.analysis.outcome;
    APP.analysis.predictors = APP.analysis.predictors.filter(n => n !== outcome);
    APP.analysis.covariates = APP.analysis.covariates.filter(n => n !== outcome && APP.analysis.predictors.indexOf(n) < 0);
    APP.analysis.features = APP.analysis.features && APP.analysis.features.length
      ? APP.analysis.features.filter(n => n !== outcome)
      : vars.map(v => v.name).filter(n => n !== outcome);

    document.getElementById('predBox').innerHTML = chipsHtml(APP.analysis.predictors, 'pred', vars);
    document.getElementById('covBox').innerHTML = chipsHtml(APP.analysis.covariates, 'cov', vars);
    document.getElementById('featBox').innerHTML = chipsHtml(APP.analysis.features, 'feat', vars);
    document.querySelectorAll('[data-rm]').forEach(i => {
      i.addEventListener('click', () => {
        const n = i.dataset.name, k = i.dataset.rm;
        const list = k === 'pred' ? APP.analysis.predictors : k === 'cov' ? APP.analysis.covariates : APP.analysis.features;
        const idx = list.indexOf(n); if (idx >= 0) list.splice(idx, 1);
        refreshChips(vars); refreshSummary();
      });
    });
  }

  /** 单行指标摘要：样本量 · 变量数 · EPV；超限转红并附「查看建议」 */
  function refreshSummary() {
    const box = document.getElementById('sampleAlert');
    if (!box) return;
    const n = APP.dataset.n_rows;
    const k = (APP.analysis.predictors.length + APP.analysis.covariates.length) || 1;
    const epv = Math.floor(n / k);
    const maxVar = Math.max(1, Math.floor(n / 10));
    const lowN = n < 50, lowEpv = epv < 10;
    const over = lowN || lowEpv;

    const core = `样本量 <span class="mono">${n}</span> · 纳入变量 <span class="mono">${k}</span> · EPV <span class="mono">${epv}</span>`;
    const flag = over ? ` · ${lowN ? '样本量不足' : 'EPV 低于 10'}` : '';

    let tip = '';
    if (lowN) tip += `样本量 ${n} < 50，检验效能不足，模型结果仅供参考，建议以描述性统计与组间比较为主。`;
    if (lowEpv) {
      tip += (tip ? '　' : '') +
        `当前 EPV=${epv}，低于经验阈值 10，存在过拟合风险。建议把纳入变量减少至 ≤ ${maxVar} 个，` +
        `或改用惩罚回归（LASSO）并仅报告单变量结果。`;
    }

    box.innerHTML = `
      <div class="vline ${over ? 'over' : 'ok'}" style="border-bottom:1px solid var(--line-soft);padding-bottom:5px">
        <span>${over ? '▲' : '✓'}</span>
        <span>${core}${flag}</span>
        ${over ? '<span class="acts"><button class="btn link" id="epvAdvice">查看建议</button></span>' : ''}
      </div>
      ${tip ? `<div class="hint" id="epvTip" hidden>${esc(tip)}</div>` : ''}`;

    const ab = document.getElementById('epvAdvice'), tt = document.getElementById('epvTip');
    if (ab && tt) ab.addEventListener('click', () => { tt.hidden = !tt.hidden; });
  }

  /** 数值型结局警示条：连续变量作结局不能用于组间比较筛选，只适用回归类分析 */
  function refreshOutcomeWarn(vars) {
    const box = document.getElementById('outcomeTypeWarn');
    if (!box) return;
    const v = vars.find(x => x.name === APP.analysis.outcome);
    const numeric = !!(v && v.confirmed_type === 'numeric');
    box.hidden = !numeric;
    if (numeric) {
      box.textContent = `「${v.name}」为数值型（连续）变量：不能作为分组进行单变量筛选，` +
        `分析时将按回归类分析处理（多因素线性回归 / 机器学习回归）。` +
        `若本意是比较组间差异（如良性 vs 恶性），请改选分类变量（如诊断、分组）作为结局变量。`;
    }
  }

  function run() {
    const btn = document.getElementById('runBtn');
    const vars = activeVars();
    const outcomeVar = vars.find(v => v.name === APP.analysis.outcome);
    const outcomeNumeric = !!(outcomeVar && outcomeVar.confirmed_type === 'numeric');
    if (outcomeNumeric) {
      // 必须用页内对话框：预览环境 WebView 会静默拦截原生 window.confirm
      API.confirmDlg(
        `结局变量「${APP.analysis.outcome}」为数值型（连续变量）：\n\n` +
        `· 单变量筛选（组间比较）将被跳过；\n` +
        `· 多因素分析将按线性回归处理；\n` +
        `· 机器学习将按回归任务处理。\n\n` +
        `若本意是「按分组比较指标差异」（如良性 vs 恶性），请点「取消」并改选分类变量（如 diagnosis）作为结局。`
      ).then((ok) => {
        if (ok) { runJobs(btn, vars, true); }
        else { toast('已取消：请改选分类变量（如 diagnosis）作为结局后再分析', 'warn'); }
      });
      return;
    }
    runJobs(btn, vars, false);
  }

  function runJobs(btn, vars, outcomeNumeric) {
    // 每次运行前重置结果：否则重复运行会叠加旧卡片，换结局后旧结论残留
    APP.results = { describe: null, screen: null, univariate: [], multivariable: null, ml: null };
    btn.disabled = true; btn.textContent = '分析中…';
    const d = APP.dataset.id;
    const outcome = APP.analysis.outcome;
    const predictors = APP.analysis.predictors.slice();
    const covariates = APP.analysis.covariates.slice();
    const features = APP.analysis.features.slice();
    const isCls = !outcomeNumeric && APP.analysis.task !== 'regression';

    API.setBusyLabel('分析中');
    const jobs = [
      API.post('/api/analysis/describe', { dataset_id: d }).then(r => { APP.results.describe = r; }),
    ];
    if (outcomeNumeric) {
      // 连续结局：组间比较筛选不适用，明确标记为跳过并在看板/报告中说明
      APP.results.screen = {
        skipped: true,
        note: '结局变量为数值型（连续结局），单变量组间比较筛选不适用，已跳过；请参考多因素线性回归与机器学习回归结果。',
      };
    } else {
      jobs.push(API.post('/api/analysis/screen', { dataset_id: d, outcome,
        candidates: activeVars().map(v => v.name).filter(n => n !== outcome) })
        .then(r => { APP.results.screen = r; }));
    }
    if (predictors.length) {
      jobs.push(      // 失败时把后端原因（如"Logistic 回归要求二分类结局"）持久化进结果，
      // 否则看板/报告会整段消失而用户无从知晓原因——toast 转瞬即逝，不够
      API.post('/api/analysis/multivariable', {
        dataset_id: d, outcome, predictors, covariates,
        task: APP.analysis.task === 'auto' ? 'auto' : (isCls ? 'logistic' : 'linear'),
      }).then(r => { APP.results.multivariable = r; })
        .catch(e => { APP.results.multivariable = { skipped: true, note: e.message }; }));
      if (!outcomeNumeric) {
        // 组间比较卡片依赖分类结局，连续结局下全部跳过（避免逐条 422 报错刷屏）
        jobs.push(Promise.all(predictors.map(p => {
          const v = activeVars().find(x => x.name === p);
          return API.post('/api/analysis/univariate', {
            dataset_id: d, target: p, group: outcome,
            target_type: v && v.confirmed_type === 'categorical' ? 'categorical' : 'numeric',
          }).then(r => { APP.results.univariate.push({ variable: p, result: r }); })
            .catch(e => {
              console.warn(`Univariate analysis failed for ${p}:`, e);
              APP.results.univariate.push({ variable: p, result: { error: e.message } });
            });
        })));
      }
    }
    jobs.push(API.postTask('/api/ml/train/async', {
      dataset_id: d, features, target: outcome,
      task: isCls ? 'classification' : 'regression',
    }).then(r => {
      APP.results.ml = r;
      // 建模完成：点亮侧栏模型灯（算法名 + 数据集指纹）
      if (window.LinkStatus) LinkStatus.setModelFromMl(r, d);
    }).catch(e => {
      console.error('ML training failed:', e);
      APP.results.ml = {
        skipped: true,
        reason: '模型训练失败：' + (e.message || '未知错误'),
        models: []
      };
    }));

    Promise.all(jobs).then(() => {
      btn.disabled = false; btn.textContent = '开始分析';
      APP.markDone('config', true);
      APP.markDone('dashboard', true);
      APP.go('dashboard');
      // 写一条后端分析历史（前台静默，失败不打扰），侧栏运行记录与上传页历史表都从这里取数
      const ml = APP.results.ml;
      const bm = (ml && ml.best_model) || null;
      const bmInfo = ((ml && ml.models) || []).find(m => m.name === (bm && bm.name)) || null;
      const tm = (bmInfo && bmInfo.test_metrics) || {};
      const payload = { task: (ml && ml.task) || null, algorithm: (bm && bm.label) || null };
      if (tm.roc_auc != null) payload.auc = tm.roc_auc;
      if (tm.r2 != null) payload.r2 = tm.r2;
      fetch(API.API_BASE + '/api/history', {
        method: 'POST',
        // 绕过 API.request 的静默写历史也要带访问令牌，否则后端启用 API_TOKEN 时会 401
        headers: Object.assign({ 'Content-Type': 'application/json' },
                               API.getToken() ? { 'X-API-Token': API.getToken() } : {}),
        body: JSON.stringify({
          dataset_name: APP.dataset.name,
          n_rows: APP.dataset.n_rows,
          outcome: APP.analysis.outcome || null,
          payload,
        }),
      }).catch(() => {}).finally(() => { if (window.Runs) Runs.refresh(); });
    }).catch(() => { btn.disabled = false; btn.textContent = '开始分析'; });
  }

  window.Pages['config'] = { onShow: render };
})();
