// medical-data-analysis/frontend/public/js/report.js
// 报告中心：学术文档式排版 —— 表 N 编号（表题在上）+ 图 N 编号（图题在下）+ 侧边目录锚点。

(function () {
  'use strict';
  window.Pages = window.Pages || {};

  let figN = 0, tblN = 0;

  /** 表题置于表格上方 */
  function table(caption, html) {
    return `<div class="cap"><span class="no">表 ${++tblN}</span> · ${esc(caption)}</div>${html}`;
  }

  /**
   * 图题置于图形下方；图形在离屏容器中渲染为静态 PNG，保证打印一致。
   * hostWidth 应尽量接近实际显示宽度，避免缩放后坐标轴文字过小。
   */
  function figure(caption, drawFn, height, hostWidth) {
    const id = 'repfig_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7);
    const host = document.createElement('div');
    host.id = id;
    host.className = 'offscreen';
    host.style.height = (height || 300) + 'px';
    if (hostWidth) host.style.width = hostWidth + 'px';
    document.body.appendChild(host);
    // 关键：离屏渲染后立刻导出 PNG，若开着入场动画会抓到第 0 帧（轴在、序列空），
    // 这正是"看板有图、报告没图"的原因
    Charts.setNoAnim(true);
    try {
      drawFn(id);
      const inst = echarts.getInstanceByDom(host);
      if (!inst) throw new Error('no chart');
      const url = inst.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
      inst.dispose();
      host.remove();
      return `<figure>
        <img src="${url}" alt="${esc(caption)}"
             style="width:100%;border:1px solid var(--line);border-radius:3px;background:#fff;display:block" />
        <div class="cap"><span class="no">图 ${++figN}</span> · ${esc(caption)}</div>
      </figure>`;
    } catch (e) {
      host.remove();
      return '';
    } finally {
      Charts.setNoAnim(false);
    }
  }

  /**
   * 必需章节是否有数据：数据集 + 至少一类分析结果。
   * 缺任一核心结果时不渲染"每节都是否定句"的报告，直接给空态与引导。
   */
  function hasContent(d, R) {
    if (!d) return false;
    return !!(R.screen || R.multivariable || R.ml || ((R.univariate || []).length));
  }

  function renderEmpty(reason) {
    const body = document.getElementById('reportBody');
    const btn = document.getElementById('exportPdf');
    if (btn) btn.hidden = true;
    body.innerHTML = `
      <div class="card">
        <div class="empty">报告尚未生成</div>
        <div class="form-actions" style="justify-content:center">
          <button class="btn primary" id="gotoConfigFromReport">前往分析配置</button>
        </div>
        <div class="hint">${esc(reason)}</div>
      </div>`;
    const g = document.getElementById('gotoConfigFromReport');
    if (g) g.addEventListener('click', () => APP.go('config'));
  }

  function render() {
    const body = document.getElementById('reportBody');
    const d = APP.dataset;
    const R = APP.results;

    if (!hasContent(d, R)) {
      renderEmpty(!d ? '当前没有已加载的数据集。' : '尚未运行分析，报告无内容可汇总。');
      return;
    }
    const btn = document.getElementById('exportPdf');
    if (btn) btn.hidden = false;
    // 报告真正生成（有可汇总的分析结果）才算完成本步；空态不算
    APP.markDone('report', true);

    figN = 0; tblN = 0;

    const A = APP.analysis;
    const ml = R.ml;
    const bestMl = ml && !ml.skipped ? (ml.models || []).find(m => m.name === ml.best_model.name) : null;
    const mv = R.multivariable;
    const important = ((R.screen || {}).rows || []).filter(r => r.significant_fdr);
    const minor = ((R.screen || {}).rows || []).filter(r => !r.significant_fdr);
    const excluded = (d.variables || []).filter(v => v.excluded);
    const now = Theme.stampText();

    const toc = [
      ['r1', '一、数据概况'], ['r2', '二、结局变量说明'], ['r3', '三、重要指标清单'],
      ['r4', '四、单变量分析'], ['r5', '五、多因素校正'], ['r6', '六、模型表现与评估'],
      ['r7', '七、特征重要性与 SHAP'], ['r8', '八、局限性与建议'],
    ];
    const tocHtml = toc.map(([id, t]) => `<a href="#${id}">${t}</a>`).join('')
      + (minor.length ? '<a href="#r9">附录　次要指标</a>' : '');

    // 图形先占位：此刻 .report-paper 尚未插入文档，量不到真实宽度；
    // 插入后由 fillFigures() 按实测列宽补渲 PNG
    const figureHtml = FIGURES_SLOT;

    body.innerHTML = `<div class="report-wrap">
      <nav class="report-toc"><div class="toc-t">目录</div>${tocHtml}</nav>
      <div class="report-paper">
        <h1>医疗数据分析报告</h1>
        <div class="hint">
          数据集 <span class="mono">${esc(d.name)}</span> · 指纹 <span class="mono">${esc(Theme.fingerprint(d.id))}</span> ·
          生成时间 <span class="mono">${esc(now)}</span> · 本地模式
        </div>
        <div class="alert warn">本工具分析结果仅供科研参考，不构成临床诊断依据。</div>

        <h2 id="r1">一、数据概况</h2>
        <p>共纳入 <b class="mono">${d.n_rows}</b> 例样本，有效变量 <b class="mono">${(d.variables || []).filter(v => !v.excluded).length}</b> 个。
           解析信息：格式 <span class="mono">${esc((d.parse_meta || {}).format || '—')}</span>，
           编码 <span class="mono">${esc((d.parse_meta || {}).encoding || '—')}</span>。</p>
        ${(d.parse_meta || {}).skipped_rows
          ? `<p class="alert warn">${esc(d.parse_meta.note)}</p>` : ''}

        <h3>排除字段</h3>
        <p>${excluded.length
          ? excluded.map(v => `<span class="tag gray">${esc(v.name)}（${esc(v.exclude_reason || '')}）</span>`).join(' ')
          : '无'}</p>

        <h3>清洗决策</h3>
        <p>${APP.cleaning
          ? ((APP.cleaning.actions || []).length
            ? (APP.cleaning.actions || []).map(a => `${esc(a.action)}：${esc(a.detail)}`).join('<br/>')
            : '未执行额外清洗操作。')
          : '未执行额外清洗（缺失值按成对删除处理）。'}</p>
        ${d.validation && !d.validation.is_medical
          ? `<div class="alert warn">该数据集未通过医疗数据校验，由使用者手动放行继续分析（force_continue）。</div>` : ''}

        <h2 id="r2">二、结局变量说明</h2>
        <p>结局变量：<b class="mono">${esc(A.outcome || '未指定')}</b>${A.task && A.task !== 'auto'
          ? `（任务类型：${A.task === 'regression' ? '回归' : '分类'}）` : ''}。
           未指定结局变量时不强行训练模型。</p>

        <h2 id="r3">三、重要指标清单</h2>
        ${R.screen && R.screen.skipped
          ? `<div class="alert warn">${esc(R.screen.note || '结局变量为数值型，单变量组间比较筛选不适用，已跳过。')}</div>`
          : important.length
          ? table('重要指标清单（FDR 校正后显著）', `<table class="grid">
              <thead><tr><th>指标</th><th>检验方法</th><th class="num">原始 p</th>
              <th class="num">FDR 校正 p</th><th>效应量</th></tr></thead>
              <tbody>${important.map(r => `<tr><td class="mono">${esc(r.variable)}</td><td>${esc(r.test)}</td>
                <td class="num">${API.fmtP(r.p)}</td><td class="num">${API.fmtP(r.p_fdr)}</td>
                <td>${r.effect_name ? `${esc(r.effect_name)} = <span class="mono">${r.effect_value}</span>` : '—'}</td>
              </tr>`).join('')}</tbody></table>`)
          : '<p>未筛选出 FDR 校正后显著的指标。</p>'}

        <h2 id="r4">四、单变量分析结果</h2>
        ${(R.univariate || []).length
          ? table('单变量分析结果', `<table class="grid">
              <thead><tr><th>指标</th><th>检验方法</th><th class="num">统计量</th>
              <th class="num">p</th><th>效应量</th><th>估计值 (95% CI)</th></tr></thead>
              <tbody>${R.univariate.map(it => {
                const r = it.result || {}, t = r.test || {}, e = r.effect || {}, est = r.estimate || {};
                return `<tr><td class="mono">${esc(it.variable)}</td><td>${esc(t.name || '—')}</td>
                  <td class="num">${t.statistic === null || t.statistic === undefined ? '—' : t.statistic}</td>
                  <td class="num">${API.fmtP(t.p)}</td>
                  <td>${e.name ? `${esc(e.name)} = <span class="mono">${e.value}</span>` : '—'}</td>
                  <td class="num">${est.value !== null && est.value !== undefined ? API.fmtEst(est.value, est.ci, 3) : '—'}</td>
                </tr>`; }).join('')}</tbody></table>`)
          : '<p>未执行单变量分析。</p>'}

        <h2 id="r5">五、多因素校正结果</h2>
        ${mv && mv.skipped
          ? `<div class="alert warn">${esc(mv.note || '多因素分析未执行')}</div>`
          : mv
          ? `<p>模型：${mv.task === 'logistic' ? '多因素 Logistic 回归（OR）' : '多元线性回归（β）'}，
             完整病例 n=${mv.n_used}${mv.n_events !== null && mv.n_events !== undefined
               ? `，事件数 ${mv.n_events}，EPV=${mv.epv}` : ''}。</p>
             ${table('多因素校正结果（校正前后对照）', `<table class="grid">
              <thead><tr><th>变量</th><th>参照组</th><th class="num">校正前</th>
              <th class="num">校正后</th><th class="num">校正前 p</th><th class="num">校正后 p</th></tr></thead>
              <tbody>${(mv.terms || []).map(t => `<tr>
                <td class="mono">${esc(t.variable)}${t.level ? ` · ${esc(t.level)}` : ''}</td>
                <td>${esc(t.reference || '—')}</td>
                <td class="num">${API.fmtEst(t.crude_estimate, t.crude_ci, 3)}</td>
                <td class="num"><b>${API.fmtEst(t.estimate, t.ci, 3)}</b></td>
                <td class="num">${API.fmtP(t.crude_p)}</td>
                <td class="num"><b>${API.fmtP(t.p)}</b></td></tr>`).join('')}</tbody></table>`)}
             ${(mv.confounding || []).map(c => `<div class="alert warn"><b>混杂提示：</b>${esc(c.message)}</div>`).join('')}
             ${(mv.warnings || []).map(w => `<div class="alert warn">${esc(w)}</div>`).join('')}`
          : '<p>未执行多因素分析。</p>'}

        <h2 id="r6">六、模型表现与评估</h2>
        ${ml && !ml.skipped
          ? `<p>推荐模型：<b>${esc(API.dedupeLabel(ml.best_model.label, ml.best_model.reason))}</b>。
             训练集 n=${ml.n_train}，测试集 n=${ml.n_test}，随机种子 ${ml.random_seed}。</p>
             ${figureHtml}
             ${table('候选模型对比（测试集）', `<table class="grid">
              <thead><tr><th>模型</th><th class="num">交叉验证</th>
              <th class="num">测试集 ${ml.task === 'classification' ? 'AUC' : 'R²'}</th>
              <th class="num">${ml.task === 'classification' ? '准确率' : 'RMSE'}</th></tr></thead>
              <tbody>${(ml.models || []).map(m => `<tr><td>${esc(m.label)}</td><td class="num">${m.cv_text}</td>
                <td class="num">${m.test_metrics.roc_auc !== undefined ? m.test_metrics.roc_auc : m.test_metrics.r2}</td>
                <td class="num">${m.test_metrics.accuracy !== undefined ? m.test_metrics.accuracy : m.test_metrics.rmse}</td>
              </tr>`).join('')}</tbody></table>`)}
             ${interpModelMetrics(ml, bestMl)}
             ${(ml.warnings || []).map(w => `<div class="alert warn">${esc(w)}</div>`).join('')}`
          : `<p>${ml && ml.skipped ? esc(ml.reason) : '未执行机器学习建模。'}</p>`}

        <h2 id="r7">七、特征重要性与 SHAP 解读</h2>
        ${ml && ml.shap && !ml.shap.error
          ? `<p>解释方法：${esc(ml.shap.method)}，样本量 ${ml.shap.n_sample}（独热哑变量已按原始变量聚合）。</p>
             ${table('SHAP 特征贡献排序', `<table class="grid">
               <thead><tr><th>变量</th><th class="num">平均 |SHAP|</th></tr></thead>
               <tbody>${(ml.shap.features || []).slice(0, 15).map(f => `<tr>
                 <td class="mono">${esc(f.variable)}</td><td class="num">${f.mean_abs_shap}</td></tr>`).join('')}
               </tbody></table>`)}`
          : '<p>未生成 SHAP 结果。</p>'}

        <h2 id="r8">八、局限性与建议</h2>
        <ul>
          <li>本分析为观察性数据的统计关联分析，不能据此推断因果关系。</li>
          <li>缺失值采用成对删除处理，各分析实际纳入样本量可能不同，请以单项结果标注的样本量为准。</li>
          <li>单变量筛选结果仅用于描述性报告；结论以 FDR 校正后 p 值为准。</li>
          ${mv && mv.epv !== null && mv.epv !== undefined && mv.epv < 10
            ? '<li>多因素模型 EPV&lt;10，存在过拟合风险，建议扩充样本或减少自变量。</li>' : ''}
          ${d.n_rows < 50 ? '<li>样本量不足 50，模型与检验结果仅供参考。</li>' : ''}
          <li>本报告由程序自动生成，数值全部来自后端 scipy / sklearn / statsmodels 计算结果。</li>
        </ul>

        ${minor.length ? `<h2 id="r9">附录　次要指标（FDR 校正后不显著）</h2>
          ${table('次要指标清单', `<table class="grid">
            <thead><tr><th>指标</th><th>检验方法</th><th class="num">原始 p</th><th class="num">FDR 校正 p</th></tr></thead>
            <tbody>${minor.map(r => `<tr><td class="mono">${esc(r.variable)}</td><td>${esc(r.test)}</td>
              <td class="num">${API.fmtP(r.p)}</td><td class="num">${API.fmtP(r.p_fdr)}</td></tr>`).join('')}
            </tbody></table>`)}` : ''}
      </div>
    </div>`;

    // 占位区此时已有真实布局宽度，按实测列宽补渲图形
    fillFigures(ml, bestMl);
  }

  /**
   * 第六节导读文字：只引用后端真正产出的指标。
   * 二分类下三件套为单条曲线；多分类下 ROC / 校准按每类 OvR 一条曲线（roc.multi），
   * DCA 取各类净获益宏平均。缺哪段就说哪段，绝不引用 undefined 字段。
   */
  function interpModelMetrics(ml, bestMl) {
    const metrics = [];
    if (ml.task === 'classification') {
      metrics.push(`ROC AUC = <span class="mono">${bestMl.test_metrics.roc_auc}</span>，准确率 <span class="mono">${bestMl.test_metrics.accuracy}</span>`);
      if (ml.roc && !ml.roc.multi) {
        metrics.push(`约登最佳截断值 <span class="mono">${ml.roc.youden_threshold}</span>`);
      }
      if (ml.roc && ml.roc.multi) {
        metrics.push(`多分类 ROC 宏平均 AUC = <span class="mono">${ml.roc.macro_auc}</span>（各类按 one-vs-rest 计算）`);
      }
      if (ml.calibration && ml.calibration.hosmer_lemeshow) {
        const hl = ml.calibration.hosmer_lemeshow;
        metrics.push(`Hosmer-Lemeshow 检验 p = <span class="mono">${API.fmtP(hl.p)}</span>（${hl.p > 0.05 ? '校准良好' : '校准欠佳'}）`);
      }
      if (ml.dca) {
        metrics.push(`DCA 最大净获益 <span class="mono">${ml.dca.best_net_benefit}</span>（阈值 <span class="mono">${ml.dca.best_threshold}</span>）`);
      }
    } else {
      metrics.push(`RMSE <span class="mono">${bestMl.test_metrics.rmse}</span>，R² <span class="mono">${bestMl.test_metrics.r2}</span>`);
    }
    return `<p>${metrics.join('；')}。全部评估指标均基于测试集 n=${ml.n_test} 计算。</p>`;
  }

  /** 第六节附图：ROC 与特征重要性（有结果才出图，保证图号连续） */
  function buildFigures(ml, bestMl, hostWidth) {
    if (!ml || ml.skipped) return [];
    // 离屏渲染宽度必须与图形最终显示宽度一致：PNG 是位图，按别的宽度渲染出来再拉伸，
    // 一是文字字号失真，二是当时被裁掉的文字会永久固定在图片里。
    const W = Math.max(280, Math.round(hostWidth || 520));
    const parts = [];
    if (ml.task === 'classification' && ml.roc) {
      const rocTitle = ml.roc.multi
        ? `ROC 曲线（多分类 OvR，宏平均 AUC=${ml.roc.macro_auc}，测试集 n=${ml.n_test}）`
        : `ROC 曲线（测试集 n=${ml.n_test}，AUC=${ml.roc.auc}）`;
      parts.push(figure(rocTitle, (id) => Charts.roc(ml.roc, id), 300, W));
      if (ml.calibration) {
        const hlP = ml.calibration.hosmer_lemeshow
          ? `Hosmer-Lemeshow p=${API.fmtP(ml.calibration.hosmer_lemeshow.p)}`
          : '各类 OvR 曲线';
        parts.push(figure(`校准曲线（${hlP}）`,
          (id) => Charts.calibration(ml.calibration, id), 300, W));
      }
      if (ml.dca) {
        const dcaNote = ml.dca.multi ? '各类 OvR 净获益宏平均，' : '';
        parts.push(figure(`决策曲线 DCA（${dcaNote}最大净获益 ${ml.dca.best_net_benefit}，阈值 ${ml.dca.best_threshold}）`,
          (id) => Charts.dca(ml.dca, id), 300, W));
      }
    }
    if (ml.feature_importance && ml.feature_importance.length) {
      parts.push(figure('特征重要性（按原始变量聚合）',
        (id) => Charts.importance(ml.feature_importance, id), 320, W));
    }
    return parts;
  }

  /**
   * 图形容器占位 → 插入 DOM 后补渲。
   * 生成报告时 .report-paper 还没插入文档，量不到真实宽度，只能先留占位；
   * 插入后再按实测列宽渲染 PNG，避免"按 520px 渲染、按 723px 显示"的字号失真。
   */
  // 占位符本身就是一个空的 .fig-grid，这样补渲时能读到网格实际排出的列宽
  const FIGURES_SLOT = '<div class="fig-grid" id="reportFigures"></div>';

  function fillFigures(ml, bestMl) {
    const host = document.getElementById('reportFigures');
    if (!host) return;
    if (!ml || ml.skipped) { host.remove(); return; }
    // 列数随视口变（≤1280px 为单列，更宽为双列），所以直接读网格算出的实际列宽，
    // 不能写死"/2"——否则单列布局下会按半宽渲染，再被拉到全宽，字号失真、笔划发虚
    const colW = parseFloat(getComputedStyle(host).gridTemplateColumns) || 520;
    const parts = buildFigures(ml, bestMl, colW);
    host.outerHTML = parts.length ? `<div class="fig-grid">${parts.join('')}</div>` : '';
  }

  window.Pages.report = {
    init() {
      document.getElementById('exportPdf').addEventListener('click', () => window.print());
    },
    onShow: render,
  };
})();
