// medical-data-analysis/frontend/public/js/charts.js
// ECharts 统一封装：统一主题（配色 / 字体 / 网格）、学术图号卡片、PNG 下载。

(function () {
  'use strict';

  const instances = {};
  // 报告中心离屏导出用：关掉入场动画，否则 getDataURL 会捕捉到动画起始帧（序列为空）
  let NO_ANIM = false;
  function setNoAnim(v) { NO_ANIM = !!v; }

  const C = () => (window.Theme ? Theme.COLORS : {
    primary: '#0E5E6F', ok: '#2F855A', bad: '#C0392B', amber: '#B45309',
    gray: '#9CA3AF', primaryWeak: '#E6F2F4', text: '#111827', dim: '#4B5563',
  });

  // 容器尺寸变化（结果渲染后出现滚动条、侧栏收起、窗口缩放等）后必须 resize：
  // ECharts 画布宽度在 init 时固定，容器变窄后右缘刻度/图例会被卡片裁掉
  let ro = null;
  function ensureObserver() {
    if (ro || typeof ResizeObserver === 'undefined') return ro;
    ro = new ResizeObserver(() => {
      Object.values(instances).forEach(c => {
        if (c.isDisposed()) return;
        // 必须走 getDom()：读私有 c.dom 会得到 undefined，回调永远 return，
        // 图表尺寸就停留在 init 那一刻，窗口/侧栏变化后右缘被裁
        const el = domOf(c);
        if (!el || !el.isConnected) return;
        try { c.resize(); } catch (e) { /* 实例销毁竞态，忽略 */ }
      });
    });
    return ro;
  }
  /** 取实例容器：只走公开 API getDom()；读私有属性会拿到 undefined，传给
   *  ResizeObserver.unobserve() 会抛 TypeError，进而中断 render() 导致图表空白 */
  function domOf(chart) {
    try { return chart && typeof chart.getDom === 'function' ? chart.getDom() : null; }
    catch (e) { return null; }
  }
  function observe(el) {
    const o = ensureObserver();
    if (o && el instanceof Element) o.observe(el);
  }
  function unobserve(el) {
    if (!ro || !(el instanceof Element)) return;
    try { ro.unobserve(el); } catch (e) { /* 容器已从文档移除，忽略 */ }
  }

  function render(id, option, height) {
    const el = document.getElementById(id);
    if (!el) return null;
    if (height) el.style.height = height + 'px';
    if (instances[id]) {
      unobserve(domOf(instances[id]));
      try { instances[id].dispose(); } catch (e) { /* 已销毁，忽略 */ }
      delete instances[id];
    }
    const chart = echarts.init(el);
    const base = Theme.echartsBase();
    if (NO_ANIM) base.animation = false;
    chart.setOption(Object.assign(base, option));
    instances[id] = chart;
    observe(el);
    return chart;
  }

  /**
   * 直角坐标网格：一律 containLabel，由 ECharts 按真实字体度量自动留白。
   *
   * 不能写死边距：变量名长度随数据集变（`fractal_dimension_worst` 127px、
   * `perimeter_and_area_composite_index` 233px），刻度位数随模型变（同一份数据的 |SHAP|
   * 最大值 Logistic 为 6.18、随机森林分类 0.378、随机森林回归 206），固定像素必然顾此失彼。
   *
   * containLabel 只兜刻度标签，不兜轴名 —— 轴名一律用 Theme.axisName()（置于轴中点，
   * 落在下面的 bottom / left 留白里），否则 nameLocation 默认值 'end' 会把轴名推出画布。
   *
   * @param o 额外让位项：top（标题或 y 轴名占顶部）、bottom（x 轴名所占条带）
   */
  function gridXY(o) {
    return Object.assign({ containLabel: true, left: 10, right: 16, top: 36, bottom: 20 }, o || {});
  }

  function download(id, filename) {
    const chart = instances[id];
    if (!chart) return;
    const url = chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
    const a = document.createElement('a');
    a.href = url; a.download = (filename || id) + '.png';
    document.body.appendChild(a); a.click(); a.remove();
  }

  // ---------------- 图号序列 ----------------
  let figSeq = 0;
  function resetFig() { figSeq = 0; }
  function nextFig() { return ++figSeq; }

  /**
   * 图表卡片：图号 + 标题 + 下载按钮 + 容器 + 右下角样本量/生成时间
   * opts: {fig, title, sub, note, tall}
   */
  function cardHtml(id, opts) {
    const o = opts || {};
    const fig = o.fig !== undefined ? o.fig : nextFig();
    return `<div class="chart-card">
      <div class="cc-head">
        <div>
          <div class="cc-title"><span class="fig">图 ${fig}</span> · ${esc(o.title || '')}</div>
          ${o.sub ? `<div class="cc-sub">${esc(o.sub)}</div>` : ''}
        </div>
        <button class="dl-btn" data-dl="${id}">PNG</button>
      </div>
      <div class="cc-body"><div id="${id}" class="chart${o.tall ? ' tall' : ''}"></div></div>
      ${o.note ? `<div class="cc-foot">${esc(o.note)}</div>` : ''}
    </div>`;
  }

  /** 图脚：样本量 + 生成时间 */
  function note(n) {
    return `n=${n} · ${Theme.stampText()}`;
  }

  /** 数据到位后回填真实样本量 */
  function setNote(id, n) {
    const el = document.getElementById(id);
    if (!el) return;
    const card = el.closest('.chart-card');
    const foot = card && card.querySelector('.cc-foot');
    if (!foot) return;
    foot.textContent = n === null || n === undefined || n === ''
      ? Theme.stampText() : `n=${n} · ${Theme.stampText()}`;
  }

  // ---------------- 具体图表 ----------------

  function histogram(data, id) {
    const edges = data.edges || [], counts = data.counts || [];
    const labels = edges.slice(0, -1).map((e, i) => `${e.toFixed(2)}~${edges[i + 1].toFixed(2)}`);
    const ref = data.reference_range;
    const marks = ref ? {
      yAxis: [
        { name: '参考下限', xAxis: ref.low, lineStyle: { color: C().amber, type: 'dashed' },
          label: { formatter: `下限 ${ref.low}`, fontSize: 10 } },
        { name: '参考上限', xAxis: ref.high, lineStyle: { color: C().bad, type: 'dashed' },
          label: { formatter: `上限 ${ref.high}`, fontSize: 10 } },
      ],
    } : {};
    render(id, {
      title: { text: `${data.field} 分布`, left: 'center', top: 4,
               textStyle: { fontSize: 12, fontWeight: 600, color: C().text } },
      grid: gridXY({ left: 52, top: 30, bottom: 18 }),
      xAxis: Theme.axisXY({ type: 'category', data: labels,
        axisLabel: { rotate: 45, fontSize: 10, color: C().dim, hideOverlap: true } }),
      yAxis: Theme.axisXY(Theme.axisName('频数', { type: 'value' })),
      series: [{ type: 'bar', data: counts, itemStyle: { color: C().primary }, barMaxWidth: 26, markLine: marks.yAxis }],
    });
  }

  function boxplot(data, id) {
    const groups = data.groups || [null];
    const cats = [], box = [], out = [];
    groups.forEach((g, gi) => {
      const src = g || data;
      if (src.n === undefined) return;
      cats.push(g ? g.level : (data.field || ''));
      box.push([src.lower_fence, src.q1, src.median, src.q3, src.upper_fence]);
      (src.outliers || []).forEach(v => out.push([gi, v]));
    });
    render(id, {
      // tooltip 以 HTML 渲染，p.name 为分组水平（来自 CSV 单元格，用户可控），必须转义
      tooltip: { formatter: (p) => {
        const name = esc(String(p.name));
        const nums = (Array.isArray(p.value) ? p.value : [p.value])
          .filter((v, i) => typeof v === 'number' && i > 0);
        return `${name}<br/>${esc(String(p.seriesName || ''))}：${nums.map(v => +v.toFixed(3)).join(' / ')}`;
      } },
      title: { text: `${data.field} 箱线图${data.group ? ' · 按 ' + data.group : ''}`, left: 'center', top: 4,
               textStyle: { fontSize: 12, fontWeight: 600, color: C().text } },
      grid: gridXY({ top: 30, bottom: 18 }),
      xAxis: Theme.axisXY({ type: 'category', data: cats,
        axisLabel: { color: C().dim, fontSize: 11, hideOverlap: true } }),
      yAxis: Theme.axisXY({ type: 'value', scale: true }),
      series: [
        { name: '箱线', type: 'boxplot', data: box,
          itemStyle: { color: C().primaryWeak, borderColor: C().primary } },
        { name: '异常点', type: 'scatter', data: out, itemStyle: { color: C().bad }, symbolSize: 5 },
      ],
    });
  }

  function qqplot(data, id) {
    const pts = (data.theoretical || []).map((x, i) => [x, data.sample[i]]);
    const min = Math.min(...data.theoretical), max = Math.max(...data.theoretical);
    render(id, {
      title: { text: `${data.field} Q-Q 图`, left: 'center', top: 4,
               textStyle: { fontSize: 12, fontWeight: 600, color: C().text } },
      grid: gridXY({ left: 52, top: 30, bottom: 48 }),
      xAxis: Theme.axisXY(Theme.axisName('理论分位数',
        { type: 'value', scale: true, axisLabel: { hideOverlap: true } })),
      yAxis: Theme.axisXY(Theme.axisName('样本分位数', { type: 'value', scale: true })),
      series: [
        { type: 'scatter', data: pts, symbolSize: 4, itemStyle: { color: C().primary } },
        { type: 'line', data: [[min, min], [max, max]], showSymbol: false,
          lineStyle: { color: C().bad, type: 'dashed', width: 1.5 } },
      ],
    });
  }

  function heatmap(data, id) {
    const cols = data.columns || [];
    const cells = [];
    for (let i = 0; i < cols.length; i++)
      for (let j = 0; j < cols.length; j++)
        cells.push([j, i, data.r[i][j] === null ? '-' : data.r[i][j]]);
    
    // 动态计算高度：每个变量约 40px，最小 300px
    const height = Math.max(300, cols.length * 40 + 80);
    
    render(id, {
      // tooltip 以 HTML 渲染，类目名为数据集列名（用户可控），必须转义
      tooltip: { formatter: (p) => {
        const v = p.value;
        const val = (Array.isArray(v) && typeof v[2] === 'number') ? v[2].toFixed(3) : '-';
        return `${esc(String(cols[v[1]]))} × ${esc(String(cols[v[0]]))}：${val}`;
      } },
      grid: gridXY({ top: 14, bottom: 44, left: 80 }),
      xAxis: Theme.axisXY({ type: 'category', data: cols, splitLine: { show: false },
        axisLabel: { rotate: 45, fontSize: 11, color: C().dim, hideOverlap: true } }),
      yAxis: Theme.axisXY({ type: 'category', data: cols, splitLine: { show: false },
        axisLabel: { fontSize: 11, color: C().dim, hideOverlap: true } }),
      visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: 4,
                   inRange: { color: [C().primary, '#FFFFFF', C().bad] } },
      series: [{ type: 'heatmap', data: cells, label: { 
        show: cols.length <= 10, // 只在变量少时显示数值
        fontSize: 9, 
        formatter: (p) => typeof p.data[2] === 'number' ? p.data[2].toFixed(2) : p.data[2]
      }}],
    }, height);
  }

  /** 森林图用 DOM 渲染（利于精确对齐），这里仅负责填充容器 */
  function forest(terms, id, opts) {
    const o = opts || {};
    const vals = terms.map(t => t[o.key || 'estimate']).filter(v => v !== null && v !== undefined);
    const cis = terms.flatMap(t => (t[o.ciKey || 'ci'] || []).filter(v => v !== null && v !== undefined));
    const lo = Math.min(...cis, 1), hi = Math.max(...cis, 1);
    const l = Math.log(Math.max(lo, 1e-6)), h = Math.log(Math.max(hi, 1e-6));
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = `<div class="forest">${terms.map(t => {
      const est = t[o.key || 'estimate'], ci = t[o.ciKey || 'ci'] || [null, null];
      if (est === null || est === undefined) return '';
      const pos = (v) => ((Math.log(Math.max(v, 1e-6)) - l) / (h - l || 1)) * 100;
      const left = pos(ci[0]), right = pos(ci[1]), pt = pos(est);
      const sig = t.significant;
      return `<div class="forest-row">
        <div class="forest-label" title="${esc(t.variable || t.term)}">
          ${esc(t.level ? `${t.variable} · ${t.level}` : (t.variable || t.term))}
        </div>
        <div class="forest-bar">
          <div class="ref"></div>
          <div class="ci" style="left:${left}%;width:${Math.max(1, right - left)}%"></div>
          <div class="pt${sig ? '' : ' ns'}" style="left:${pt}%"></div>
        </div>
        <div class="forest-val">${o.fmt ? o.fmt(t) : ''}</div>
      </div>`;
    }).join('')}</div>`;
  }

  function roc(data, id) {
    // 多分类：后端按 one-vs-rest 给出每个类别一条曲线（data.multi + data.classes），
    // 与二分类共用同一张图，只是序列变成多条
    if (data && data.multi && Array.isArray(data.classes)) {
      const seq = C().seq || [C().primary, C().amber, C().ok, C().dim];
      const series = data.classes.map((c, i) => ({
        type: 'line', data: c.fpr.map((x, j) => [x, c.tpr[j]]), showSymbol: false, smooth: true,
        lineStyle: { width: 2, color: seq[i % seq.length] },
        name: `${c.label}（AUC=${c.auc}）`,
      }));
      series.push({ type: 'line', data: [[0, 0], [1, 1]], showSymbol: false,
        lineStyle: { color: C().gray, type: 'dashed', width: 1 }, name: '随机猜测' });
      render(id, {
        grid: gridXY({ left: 52, top: 32, bottom: 48 }),
        xAxis: Theme.axisXY(Theme.axisName('1 - 特异度',
          { type: 'value', max: 1, min: 0, axisLabel: { hideOverlap: true } })),
        yAxis: Theme.axisXY(Theme.axisName('敏感度', { type: 'value', max: 1, min: 0 })),
        series,
        legend: { top: 2, itemHeight: 8, itemWidth: 12,
                  textStyle: { fontSize: 11, color: C().dim } },
      });
      return;
    }
    render(id, {
      grid: gridXY({ left: 52, top: 32, bottom: 48 }),
      xAxis: Theme.axisXY(Theme.axisName('1 - 特异度',
        { type: 'value', max: 1, min: 0, axisLabel: { hideOverlap: true } })),
      yAxis: Theme.axisXY(Theme.axisName('敏感度', { type: 'value', max: 1, min: 0 })),
      series: [
        { type: 'line', data: data.fpr.map((x, i) => [x, data.tpr[i]]), showSymbol: false, smooth: true,
          lineStyle: { color: C().primary, width: 2 },
          areaStyle: { color: 'rgba(14,94,111,.08)' },
          name: `AUC = ${data.auc}` },
        { type: 'line', data: [[0, 0], [1, 1]], showSymbol: false,
          lineStyle: { color: C().gray, type: 'dashed', width: 1 } },
      ],
      // 图例移到顶部：底部留白要留给 x 轴名，两者同占底边会互相挤压
      legend: { top: 2, itemHeight: 8, itemWidth: 12,
                textStyle: { fontSize: 11, color: C().dim } },
    });
  }

  function calibration(data, id) {
    // 多分类：每个类别一条「该类作为正类」的校准曲线
    if (data && data.multi && Array.isArray(data.classes)) {
      const seq = C().seq || [C().primary, C().amber, C().ok, C().dim];
      const series = data.classes.map((c, i) => ({
        type: 'line', data: c.prob_pred.map((x, j) => [x, c.prob_true[j]]),
        showSymbol: true, symbolSize: 5,
        lineStyle: { width: 2, color: seq[i % seq.length] },
        itemStyle: { color: seq[i % seq.length] }, name: String(c.label),
      }));
      series.push({ type: 'line', data: [[0, 0], [1, 1]], showSymbol: false,
        lineStyle: { color: C().gray, type: 'dashed', width: 1 }, name: '理想校准' });
      render(id, {
        grid: gridXY({ left: 52, top: 32, bottom: 48 }),
        xAxis: Theme.axisXY(Theme.axisName('预测概率',
          { type: 'value', max: 1, min: 0, axisLabel: { hideOverlap: true } })),
        yAxis: Theme.axisXY(Theme.axisName('实际发生率', { type: 'value', max: 1, min: 0 })),
        series,
        legend: { top: 2, itemHeight: 8, itemWidth: 12,
                  textStyle: { fontSize: 11, color: C().dim } },
      });
      return;
    }
    render(id, {
      grid: gridXY({ left: 52, top: 32, bottom: 48 }),
      xAxis: Theme.axisXY(Theme.axisName('预测概率',
        { type: 'value', max: 1, min: 0, axisLabel: { hideOverlap: true } })),
      yAxis: Theme.axisXY(Theme.axisName('实际发生率', { type: 'value', max: 1, min: 0 })),
      series: [
        { type: 'line', data: data.prob_pred.map((x, i) => [x, data.prob_true[i]]),
          showSymbol: true, symbolSize: 6, lineStyle: { color: C().primary, width: 2 },
          itemStyle: { color: C().primary }, name: '模型' },
        { type: 'line', data: [[0, 0], [1, 1]], showSymbol: false,
          lineStyle: { color: C().gray, type: 'dashed', width: 1 }, name: '理想校准' },
      ],
      legend: { top: 2, itemHeight: 8, itemWidth: 12,
                textStyle: { fontSize: 11, color: C().dim } },
    });
  }

  function dca(data, id) {
    render(id, {
      grid: gridXY({ left: 52, top: 32, bottom: 48 }),
      xAxis: Theme.axisXY(Theme.axisName('阈值概率',
        { type: 'value', min: 0, max: 1, axisLabel: { hideOverlap: true } })),
      yAxis: Theme.axisXY(Theme.axisName('净获益', { type: 'value', scale: true })),
      series: [
        { type: 'line', data: data.thresholds.map((x, i) => [x, data.net_benefit_model[i]]),
          showSymbol: false, lineStyle: { color: C().primary, width: 2 }, name: '模型' },
        { type: 'line', data: data.thresholds.map((x, i) => [x, data.net_benefit_all[i]]),
          showSymbol: false, lineStyle: { color: C().amber, width: 1.5 }, name: '全部干预' },
        { type: 'line', data: data.thresholds.map((x) => [x, 0]),
          showSymbol: false, lineStyle: { color: C().gray, width: 1.5, type: 'dashed' }, name: '全不干预' },
      ],
      legend: { top: 2, itemHeight: 8, itemWidth: 12,
                textStyle: { fontSize: 11, color: C().dim } },
    });
  }

  function importance(items, id) {
    const top = items.slice(0, 20).reverse();
    render(id, {
      // tooltip 的 p.name 为变量名（用户可控列名），必须转义
      tooltip: { formatter: (ps) => {
        const p = Array.isArray(ps) ? ps[0] : ps;
        return `${esc(String(p.name))}：${(+p.value).toFixed(3)}`;
      } },
      // 变量名列的长短完全取决于数据集（实测从 45px 到 233px），交给 containLabel 自动让位
      grid: gridXY({ top: 14, bottom: 48 }),
      xAxis: Theme.axisXY(Theme.axisName('归一化重要性',
        { type: 'value', axisLabel: { hideOverlap: true } })),
      yAxis: Object.assign(Theme.axisXY({ type: 'category', data: top.map(i => i.variable),
        splitLine: { show: false }, axisLabel: { fontSize: 11, color: C().dim } })),
      series: [{ type: 'bar', data: top.map(i => i.importance),
                 itemStyle: { color: C().primary }, barMaxWidth: 14 }],
    });
  }

  function beeswarm(shap, id) {
    const feats = shap.features || [];
    const data = [];
    feats.forEach((f, i) => { f.values.forEach(v => data.push([v, i])); });
    const maxAbs = Math.max(...feats.flatMap(f => f.values.map(Math.abs)), 1);
    render(id, {
      // tooltip 的特征名来自数据集聚合列名（用户可控），必须转义
      tooltip: { formatter: (p) => {
        const f = feats[p.value[1]];
        return `${esc(String(f ? f.variable : ''))}：SHAP ${(+p.value[0]).toFixed(4)}`;
      } },
      grid: gridXY({ top: 14, bottom: 56 }),
      xAxis: Theme.axisXY(Theme.axisName('SHAP 值',
        { type: 'value', min: -maxAbs * 1.1, max: maxAbs * 1.1,
          axisLabel: { hideOverlap: true } })),
      yAxis: Object.assign(Theme.axisXY({ type: 'category', data: feats.map(f => f.variable),
        splitLine: { show: false }, axisLabel: { fontSize: 11, color: C().dim } })),
      visualMap: { min: -maxAbs, max: maxAbs, dimension: 0, orient: 'horizontal',
                   left: 'center', bottom: 4, itemHeight: 60, itemWidth: 10,
                   inRange: { color: [C().primary, '#F7F7F7', C().bad] } },
      series: [{ type: 'scatter', data, symbolSize: 4, itemStyle: { opacity: .75 } }],
    }, 280 + Math.max(0, feats.length * 16));
  }

  /**
   * 销毁所有 ECharts 实例：在页面重新渲染前调用，避免内存泄漏和图表冲突。
   * 当 dashboard.js 用 innerHTML 重建 DOM 时，旧的 ECharts 实例会变成僵尸实例，
   * 必须显式 dispose 并清空 instances 对象。
   */
  function disposeAll() {
    Object.keys(instances).forEach(id => {
      if (instances[id] && !instances[id].isDisposed()) {
        unobserve(domOf(instances[id]));
        try { instances[id].dispose(); } catch (e) { /* 忽略已销毁的实例 */ }
      }
      delete instances[id];
    });
  }

  window.Charts = { render, download, cardHtml, resetFig, nextFig, note, setNote, setNoAnim, gridXY,
                    histogram, boxplot, qqplot, heatmap, forest,
                    roc, calibration, dca, importance, beeswarm, disposeAll, COLOR: C() };
})();
