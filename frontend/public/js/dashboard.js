// medical-data-analysis/frontend/public/js/dashboard.js
// 分析看板：内联统计条 → 重要指标 → 单变量 → 多因素森林图 → 模型评估三件套 → SHAP。
// 图表统一采用学术图号（图 1 / 图 2 …），每张图右下角标注样本量与生成时间。

(function () {
  'use strict';
  window.Pages = window.Pages || {};

  const A = () => APP.analysis;

  // ---------------- 模板解读 ----------------
  function interpUnivariate(variable, r) {
    if (!r || !r.valid) {
      // 无效结果时优先展示后端给出的具体原因（如分组变量水平数超限），而不是笼统话术
      return (r && r.note) ? r.note
        : '样本量不足或前提不满足，无法给出可靠检验结果，请仅参考描述性统计。';
    }
    const t = r.test || {}, e = r.effect || {}, est = r.estimate || {};
    const sig = t.p !== null && t.p !== undefined && t.p < 0.05;
    const checks = (t.assumption_checks || []).map(c => `${c.name}${c.passed ? '通过' : '未通过'}`).join('，');
    let s = `采用${t.name}（${checks}）。`;
    s += `统计量 = ${t.statistic === null ? '—' : t.statistic}，p = ${API.fmtP(t.p)}，${sig ? '差异具有统计学意义' : '差异无统计学意义'}（α=0.05）。`;
    if (est.value !== null && est.value !== undefined) {
      s += ` ${est.label || '估计值'} = ${est.value}`;
      if (est.ci) s += `（95% CI ${est.ci[0]}–${est.ci[1]}）`;
      s += '。';
    }
    if (e.name && e.value !== null) s += ` 效应量 ${e.name} = ${e.value}（${e.interpretation}）。`;
    if (t.note) s += ' ' + t.note;
    if (r.warnings && r.warnings.length) s += ' 注意：' + r.warnings.join('；') + '。';
    return s;
  }

  function interpMultivariable(mv) {
    if (!mv) return '';
    const sigs = (mv.terms || []).filter(t => t.significant);
    let s = `多因素${mv.task === 'logistic' ? ' Logistic 回归' : '线性回归'}共纳入 ${mv.n_used} 例完整病例`;
    s += mv.n_events !== null && mv.n_events !== undefined ? `（事件数 ${mv.n_events}，EPV=${mv.epv}）` : '';
    s += '。';
    s += sigs.length
      ? `校正后具有统计学意义的变量：${sigs.map(t => t.variable + (t.level ? `·${t.level}` : '')).join('、')}。`
      : '校正后无变量具有统计学意义。';
    if (mv.warnings && mv.warnings.length) s += ' 警示：' + mv.warnings.join('；') + '。';
    return s;
  }

  function interpMl(ml) {
    if (!ml) return '';
    if (ml.skipped) return ml.reason;
    const best = (ml.models || []).find(m => m.name === ml.best_model.name);
    let s = `推荐模型：${API.dedupeLabel(ml.best_model.label, ml.best_model.reason)}。`;
    if (ml.task === 'classification') {
      const tm = best.test_metrics;
      // 敏感度/特异度仅二分类有定义，多分类下后端返回 null，这里如实省略
      const binaryPart = tm.sensitivity !== null && tm.sensitivity !== undefined
        ? `，敏感度 ${tm.sensitivity}，特异度 ${tm.specificity}` : '';
      s += ` 测试集 n=${ml.n_test}：准确率 ${tm.accuracy}，AUC ${tm.roc_auc}${binaryPart}。`;
      // 三件套：二分类为单条曲线；多分类按 one-vs-rest 每类一条（roc.multi / calibration.multi）
      if (ml.roc && ml.roc.multi) {
        s += ` 多分类 ROC 宏平均 AUC = ${ml.roc.macro_auc}（各类按 one-vs-rest 计算）。`;
      } else if (ml.roc) {
        s += ` ROC 最佳截断值（约登指数）= ${ml.roc.youden_threshold}，对应敏感度 ${ml.roc.sensitivity_at_best}、特异度 ${ml.roc.specificity_at_best}。`;
      }
      if (ml.calibration && ml.calibration.hosmer_lemeshow) {
        const hl = ml.calibration.hosmer_lemeshow;
        s += ` Hosmer-Lemeshow 检验 p=${API.fmtP(hl.p)}，${hl.p > 0.05 ? '校准良好' : '校准欠佳'}。`;
      }
      if (ml.dca) {
        s += ` DCA 在阈值 ${ml.dca.best_threshold} 处净获益最大（${ml.dca.best_net_benefit}）${ml.dca.multi ? '，为各类 OvR 宏平均' : ''}。`;
      }
    } else {
      s += ` 测试集 n=${ml.n_test}：RMSE ${best.test_metrics.rmse}，R² ${best.test_metrics.r2}。`;
    }
    if (ml.warnings && ml.warnings.length) s += ' 警示：' + ml.warnings.join('；') + '。';
    return s;
  }

  // ---------------- 内联统计条 ----------------
  const cell = (cls, val, label, sub) => `
    <div class="stat-cell ${cls || ''}">
      <div class="sv">${val}</div><div class="sl">${esc(label)}</div>
      ${sub ? `<div class="sn">${esc(sub)}</div>` : ''}
    </div>`;

  function render() {
    const body = document.getElementById('dashBody');
    const R = APP.results;
    if (!APP.dataset) { body.innerHTML = '<div class="empty">请先上传数据集</div>'; return; }
    if (!R.screen && !R.multivariable && !R.ml) {
      body.innerHTML = '<div class="empty">请先在「分析配置」页点击开始分析</div>'; return;
    }

    const screen = R.screen || { rows: [] };
    const important = (screen.rows || []).filter(r => r.significant_fdr);
    const minor = (screen.rows || []).filter(r => !r.significant_fdr);
    const ml = R.ml;
    const bestMl = ml && !ml.skipped ? (ml.models || []).find(m => m.name === ml.best_model.name) : null;
    const desc = R.describe || { numeric: [], categorical: [] };
    const totalMissing = desc.numeric.reduce((s, x) => s + (x.n_missing || 0), 0)
      + desc.categorical.reduce((s, x) => s + (x.n_missing || 0), 0);
    const missRate = APP.dataset.n_rows
      ? totalMissing / (APP.dataset.n_rows * Math.max(1, desc.numeric.length + desc.categorical.length)) : 0;

    Charts.resetFig();
    // 在重建 DOM 前销毁所有旧的 ECharts 实例，避免内存泄漏和图表冲突
    Charts.disposeAll();

    body.innerHTML = `
      <div class="gt">概览<span class="gt-note">清洗后数据集</span></div>
      <div class="stat-strip">
        ${cell('', APP.dataset.n_rows, '样本量', '清洗后')}
        ${cell('', (APP.dataset.variables || []).filter(v => !v.excluded).length, '有效变量', '已排除 ID / 常量列')}
        ${cell(missRate > 0.05 ? 'warn' : 'good', (missRate * 100).toFixed(1) + '%', '缺失率', '成对删除')}
        ${cell('', important.length, '重要指标', 'FDR 校正后显著')}
        ${cell('good', bestMl
          ? (ml.task === 'classification' ? bestMl.test_metrics.roc_auc : bestMl.test_metrics.r2) : '—',
          '最优模型 ' + (ml && ml.task === 'classification' ? 'AUC' : 'R²'),
          ml && ml.best_model ? ml.best_model.label : '未建模')}
      </div>

      ${sectionCorrelation()}
      ${sectionImportant(important, screen)}
      ${sectionUniv(R.univariate)}
      ${sectionMulti(R.multivariable)}
      ${sectionMl(ml)}
      ${sectionShap(ml)}
      ${sectionAppendix(minor)}
    `;

    // 延迟一帧确保 DOM 完全渲染后再初始化 ECharts
    requestAnimationFrame(drawAll);
  }

  function sectionImportant(rows, screen) {
    if (screen && screen.skipped) {
      return `<div class="gt">重要指标<span class="gt-note">单变量筛选 · 已跳过</span></div>
        <div class="alert warn">${esc(screen.note || '结局变量为数值型，单变量组间比较筛选不适用。')}</div>`;
    }
    const body = rows.length ? `
      <div class="table-wrap"><table class="grid">
        <thead><tr><th>指标</th><th>检验方法</th><th class="num">原始 p</th><th class="num">FDR 校正 p</th>
        <th>效应量</th><th>结论</th></tr></thead><tbody>
        ${rows.map(r => `<tr>
          <td class="mono">${esc(r.variable)}</td><td>${esc(r.test)}</td>
          <td class="num">${API.fmtP(r.p)}</td>
          <td class="num"><b>${API.fmtP(r.p_fdr)}</b></td>
          <td>${r.effect_name ? `${esc(r.effect_name)} = <span class="mono">${esc(r.effect_value)}</span>` : '—'}</td>
          <td><span class="tag ${r.significant_fdr ? 'red' : 'gray'}">${r.significant_fdr ? '显著' : '不显著'}</span>
              ${r.significant_raw && !r.significant_fdr ? '<span class="tag amber">校正后转阴</span>' : ''}</td>
        </tr>`).join('')}
        </tbody></table></div>
      <div class="interpret">结论以 <b>FDR（Benjamini-Hochberg）校正后 p 值</b>为准。单变量筛选结果仅用于描述性报告；
        若用于特征选择，必须在交叉验证内部完成。</div>`
      : '<div class="hint">未筛选出 FDR 校正后显著的指标。</div>';
    return `<div class="gt">重要指标<span class="gt-note">单变量筛选 · ${rows.length} 项</span></div>${body}`;
  }

  function sectionCorrelation() {
    const vars = (APP.dataset.variables || []).filter(v => !v.excluded && v.confirmed_type === 'numeric');
    // 只在变量数量适中时显示热力图（2-15个），太少没意义，太多难以阅读
    if (vars.length < 2 || vars.length > 15) return '';
    return `<div class="gt">相关性分析<span class="gt-note">${vars.length} 个数值变量</span></div>
      <div class="chart-grid">
        ${Charts.cardHtml('corrChart', { title: '相关系数矩阵', sub: 'Pearson（正态）/ Spearman（非正态）', note: Charts.note(APP.dataset.n_rows), tall: true })}
      </div>`;
  }

  function sectionUniv(list) {
    if (!list || !list.length) return '';
    const items = list.map((it, i) => {
      const r = it.result;
      // Handle error case
      if (r && r.error) {
        return `<div class="card" style="border-left:2px solid var(--line)">
          <div class="card-head" style="margin-bottom:0">
            <div style="font-size:13px;font-weight:600">
              <span class="mono">${esc(it.variable)}</span> × <span class="mono">${esc(A().outcome || '')}</span>
              <span class="tag gray">分析失败</span>
            </div>
          </div>
          <div class="alert warn">${esc(r.error)}</div>
        </div>`;
      }
      const sig = r && r.test && r.test.p !== null && r.test.p < 0.05;
      return `<div class="card" style="border-left:2px solid ${sig ? 'var(--bad)' : 'var(--line)'}">
        <div class="card-head" style="margin-bottom:0">
          <div style="font-size:13px;font-weight:600">
            <span class="mono">${esc(it.variable)}</span> × <span class="mono">${esc(A().outcome || '')}</span>
            <span class="tag ${sig ? 'red' : 'gray'}">${sig ? '显著' : '不显著'}</span>
          </div>
          <div style="font-size:12px;color:var(--text-mute)">${esc((r && r.test && r.test.name) || '—')}</div>
        </div>
        <div class="chart-grid" style="margin-top:8px">
          ${Charts.cardHtml('u_hist_' + i, { title: it.variable + ' 分布', sub: '直方图 · 含医学参考区间', note: Charts.note(APP.dataset.n_rows) })}
          ${Charts.cardHtml('u_box_' + i, { title: it.variable + ' 箱线图', sub: '按 ' + (A().outcome || '') + ' 分组', note: Charts.note(APP.dataset.n_rows) })}
        </div>
        <div class="interpret">${esc(interpUnivariate(it.variable, r))}</div>
      </div>`;
    }).join('');
    return `<div class="gt">单变量分析<span class="gt-note">${list.length} 个目标自变量</span></div>${items}`;
  }

  function sectionMulti(mv) {
    if (!mv) return '';
    // 后端拒绝建模（如结局非二分类）时，把原因留在原位，而不是让整段无声消失
    if (mv.skipped) {
      return `<div class="gt">多因素校正</div><div class="alert warn">${esc(mv.note || '多因素分析未执行')}</div>`;
    }
    const rows = (mv.terms || []).map(t => `
      <tr><td class="mono">${esc(t.variable)}${t.level ? ` · ${esc(t.level)}` : ''}</td>
        <td>${esc(t.reference ? `参照：${t.reference}` : '—')}</td>
        <td class="num">${API.fmtEst(t.crude_estimate, t.crude_ci, 3)}</td>
        <td class="num"><b>${API.fmtEst(t.estimate, t.ci, 3)}</b></td>
        <td class="num">${API.fmtP(t.crude_p)}</td>
        <td class="num"><b>${API.fmtP(t.p)}</b></td>
        <td>${t.significant ? '<span class="tag red">显著</span>' : '<span class="tag gray">不显著</span>'}</td></tr>`).join('');
    const warnings = (mv.warnings || []).map(w => `<div class="alert warn">${esc(w)}</div>`).join('');
    const conf = (mv.confounding || []).map(c => `<div class="alert warn">${esc(c.message)}</div>`).join('');
    const vifHigh = (mv.vif || []).filter(v => v.vif && v.vif > 10);

    const estName = mv.task === 'logistic' ? 'OR' : 'β';

    return `
      <div class="gt">多因素校正<span class="gt-note">${mv.task === 'logistic' ? 'Logistic 回归 · OR' : '线性回归 · β'} · n=${mv.n_used}${mv.n_events !== null && mv.n_events !== undefined ? ` · 事件数 ${mv.n_events}` : ''}</span></div>
      ${warnings}${conf}
      <div class="card">
        <div class="chart-card">
          <div class="cc-head">
            <div>
              <div class="cc-title"><span class="fig">图 ${Charts.nextFig()}</span> · 多因素森林图（校正后）</div>
              <div class="cc-sub">参考线 1；圆点为点估计，横线为 95% CI（灰色表示不显著）</div>
            </div>
          </div>
          <div class="cc-body"><div id="forestChart" class="chart"></div></div>
          <div class="cc-foot">${Charts.note(mv.n_used)}</div>
        </div>
      </div>
      <div class="table-wrap"><table class="grid">
        <thead><tr><th>变量</th><th>编码</th><th class="num">校正前 ${estName}</th><th class="num">校正后 ${estName}</th>
        <th class="num">校正前 p</th><th class="num">校正后 p</th><th>结论</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <div class="interpret">${esc(interpMultivariable(mv))}
        ${vifHigh.length ? ` VIF>10 的变量：${vifHigh.map(v => esc(v.term)).join('、')}，提示共线性风险。` : ''}</div>`;
  }

  function sectionMl(ml) {
    if (!ml) return '';
    if (ml.skipped) return `<div class="gt">模型表现与评估</div>
      <div class="alert warn">${esc(ml.reason)}</div>
      <div class="interpret">已跳过机器学习建模，请仅参考统计描述与组间比较结果。</div>`;

    const cmp = (ml.models || []).map(m => `
      <tr><td>${esc(m.label)}</td><td class="num">${esc(m.cv_text)}</td>
      <td class="num">${m.test_metrics.roc_auc !== undefined ? m.test_metrics.roc_auc : m.test_metrics.r2}</td>
      <td class="num">${m.test_metrics.accuracy !== undefined ? m.test_metrics.accuracy : m.test_metrics.rmse}</td>
      <td>${m.name === ml.best_model.name ? '<span class="tag green">推荐</span>' : ''}</td></tr>`).join('');

    // 评估三件套按后端实际产出的曲线渲染：二分类为单条曲线，
    // 多分类按 one-vs-rest 每类一条（roc.multi / calibration.multi），缺哪张就不渲染哪张
    const evalCards = [];
    if (ml.roc) {
      evalCards.push(Charts.cardHtml('rocChart', { title: 'ROC 曲线', sub: ml.roc.title,
        note: Charts.note(ml.n_test) }));
    }
    if (ml.calibration) {
      evalCards.push(Charts.cardHtml('calChart', { title: '校准曲线', sub: ml.calibration.title,
        note: Charts.note(ml.n_test) }));
    }
    if (ml.dca) {
      evalCards.push(Charts.cardHtml('dcaChart', { title: '决策曲线 DCA', sub: ml.dca.title,
        note: Charts.note(ml.n_test) }));
    }
    const eval3 = evalCards.length ? `<div class="chart-grid">${evalCards.join('')}</div>` : '';

    const best = ml.models.find(m => m.name === ml.best_model.name);
    const clsRows = ml.task === 'classification' && best
      ? Object.entries(best.test_metrics.per_class || {})
        .map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="num">${v.precision}</td>
          <td class="num">${v.recall}</td><td class="num">${v.f1}</td><td class="num">${v.support}</td></tr>`).join('')
      : '';

    // 测试集混淆矩阵（行=真实、列=预测）：医疗场景要看假阴性/假阳性，而不只是聚合 AUC
    const cmMat = ml.task === 'classification' && best ? best.test_metrics.confusion_matrix : null;
    const clsLabels = (ml.classes || []).map(String);
    const cmTable = cmMat && cmMat.length && cmMat.length <= 10
      ? `<div class="table-wrap" style="margin-top:10px"><table class="grid">
          <thead><tr><th>真实 \\ 预测</th>${clsLabels.map(l => `<th class="num">${esc(l)}</th>`).join('')}</tr></thead>
          <tbody>${cmMat.map((row, i) => `<tr><td class="mono">${esc(clsLabels[i] || i)}</td>
            ${row.map(v => `<td class="num">${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table></div>
        <div class="hint">混淆矩阵（测试集 n=${ml.n_test}）：对角线为正确分类；行内非对角为漏判（假阴性），列内非对角为误判（假阳性）。</div>` : '';

    return `
      <div class="gt">模型表现与评估<span class="gt-note">训练集 n=${ml.n_train} · 测试集 n=${ml.n_test} · 随机种子 ${ml.random_seed}</span></div>
      <div class="table-wrap"><table class="grid">
        <thead><tr><th>模型</th><th class="num">交叉验证（${esc((ml.models[0] || {}).cv_metric || '')}）</th>
        <th class="num">测试集 ${ml.task === 'classification' ? 'AUC' : 'R²'}</th>
        <th class="num">${ml.task === 'classification' ? '准确率' : 'RMSE'}</th><th></th></tr></thead>
        <tbody>${cmp}</tbody></table></div>
      ${clsRows ? `<div class="table-wrap" style="margin-top:10px"><table class="grid">
        <thead><tr><th>类别</th><th class="num">精确率</th><th class="num">召回率</th><th class="num">F1</th><th class="num">样本数</th></tr></thead>
        <tbody>${clsRows}</tbody></table></div>` : ''}
      ${cmTable}
      ${eval3}
      <div class="interpret">${esc(interpMl(ml))}</div>
      ${ml.calibration ? `<div class="hint">${ml.calibration.multi ? '各类 OvR 校准曲线' : '校准曲线与 HL 检验'}基于 ${esc(ml.calibration_model || 'CalibratedClassifierCV')} 校准后的测试集概率。</div>` : ''}`;
  }

  function sectionShap(ml) {
    if (!ml || ml.skipped || !ml.shap || ml.shap.error) {
      return ml && ml.shap && ml.shap.error
        ? `<div class="gt">特征重要性与 SHAP</div>
           <div class="alert err">${esc(ml.shap.error)}</div>
           ${ml.shap.hint ? `<div class="hint">${esc(ml.shap.hint)}</div>` : ''}` : '';
    }
    return `
      <div class="gt">特征重要性与 SHAP<span class="gt-note">${esc(ml.shap.method)} · 解释样本 n=${ml.shap.n_sample}</span></div>
      <div class="chart-grid">
        ${Charts.cardHtml('impChart', { title: '特征重要性', sub: '按原始变量聚合', note: Charts.note(ml.shap.n_sample) })}
        ${Charts.cardHtml('shapChart', { title: 'SHAP 蜂群图', sub: '哑变量已按原始变量聚合', note: Charts.note(ml.shap.n_sample), tall: true })}
      </div>
      <div class="interpret">SHAP 值表示该特征对单个样本预测输出的贡献方向与幅度；
        图中按平均 |SHAP| 降序排列，排名靠前者为模型主要依赖的指标。</div>`;
  }

  function sectionAppendix(minor) {
    if (!minor.length) return '';
    return `<div class="gt">附录<span class="gt-note">次要指标 ${minor.length} 项 · FDR 校正后不显著</span></div>
      <div class="card appendix">
        <details><summary>次要指标清单</summary>
        <div class="table-wrap" style="margin-top:8px"><table class="grid">
          <thead><tr><th>指标</th><th>检验方法</th><th class="num">原始 p</th><th class="num">FDR 校正 p</th></tr></thead>
          <tbody>${minor.map(r => `<tr><td class="mono">${esc(r.variable)}</td><td>${esc(r.test)}</td>
            <td class="num">${API.fmtP(r.p)}</td><td class="num">${API.fmtP(r.p_fdr)}</td></tr>`).join('')}</tbody>
        </table></div></details></div>`;
  }

  // ---------------- 绘图 ----------------
  function drawAll() {
    const R = APP.results;
    const d = APP.dataset.id;

    // 相关性热力图：仅在 2-15 个数值变量时生成
    const numericVars = (APP.dataset.variables || []).filter(v => !v.excluded && v.confirmed_type === 'numeric').map(v => v.name);
    if (numericVars.length >= 2 && numericVars.length <= 15) {
      API.post('/api/analysis/correlation', { dataset_id: d, fields: numericVars })
        .then(corr => { Charts.heatmap(corr, 'corrChart'); Charts.setNote('corrChart', corr.n); })
        .catch(e => console.warn('Correlation analysis failed:', e));
    }

    (R.univariate || []).forEach((it, i) => {
      // Skip chart rendering for failed analyses
      if (it.result && it.result.error) return;
      
      API.post('/api/analysis/histogram', { dataset_id: d, field: it.variable })
        .then(h => { Charts.histogram(h, 'u_hist_' + i); Charts.setNote('u_hist_' + i, h.n); }).catch(() => {});
      API.post('/api/analysis/boxplot', { dataset_id: d, field: it.variable, group: A().outcome })
        .then(b => { Charts.boxplot(b, 'u_box_' + i); Charts.setNote('u_box_' + i, b.n); }).catch(() => {});
    });

    const mv = R.multivariable;
    if (mv && document.getElementById('forestChart')) {
      Charts.forest(mv.terms || [], 'forestChart',
        { fmt: (t) => `${API.fmtEst(t.estimate, t.ci, 3)}  p=${API.fmtP(t.p)}` });
    }

    const ml = R.ml;
    if (ml && !ml.skipped) {
      if (ml.task === 'classification') {
        if (document.getElementById('rocChart')) { Charts.roc(ml.roc, 'rocChart'); Charts.setNote('rocChart', ml.n_test); }
        if (document.getElementById('calChart')) { Charts.calibration(ml.calibration, 'calChart'); Charts.setNote('calChart', ml.n_test); }
        if (document.getElementById('dcaChart')) { Charts.dca(ml.dca, 'dcaChart'); Charts.setNote('dcaChart', ml.n_test); }
      }
      if (ml.feature_importance && document.getElementById('impChart')) {
        Charts.importance(ml.feature_importance, 'impChart');
        Charts.setNote('impChart', ml.shap ? ml.shap.n_sample : ml.n_test);
      }
      if (ml.shap && !ml.shap.error && document.getElementById('shapChart')) {
        Charts.beeswarm(ml.shap, 'shapChart');
        Charts.setNote('shapChart', ml.shap.n_sample);
      }
    }
  }

  window.Pages.dashboard = { onShow: render };
})();
