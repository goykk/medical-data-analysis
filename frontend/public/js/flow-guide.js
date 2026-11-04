// medical-data-analysis/frontend/public/js/flow-guide.js
// 侧栏流程引导条：七格进度刻度 + 等宽计数 + 「下一步」+ 前往按钮。
// 同源约定：完成状态只有一份，存于 APP.steps（app.js 的 markDone 是唯一写入点），
// 导航 ✓ 与本引导条都从它渲染。本模块不持有状态、不发任何请求、不做动画。
// 刻度不可点击，整条只有一个动作：「前往下一步」。

(function () {
  'use strict';

  // 七个流程步骤（设置不是流程步骤，不参与计数与刻度）
  const STEPS = [
    { page: 'upload',     label: '数据上传' },
    { page: 'confirm',    label: '数据确认' },
    { page: 'cleaning',   label: '数据清洗' },
    { page: 'config',     label: '分析配置' },
    { page: 'dashboard',  label: '分析看板' },
    { page: 'prediction', label: '新样本预测' },
    { page: 'report',     label: '报告中心' },
  ];

  const $ = (id) => document.getElementById(id);

  function stepsState() {
    return (window.APP && APP.steps) || {};
  }

  function render() {
    const ticks = $('fgTicks'), count = $('fgCount'), go = $('fgGo');
    if (!ticks || !count || !go) return;

    const st = stepsState();
    const hasDataset = !!(window.APP && APP.dataset);
    const done = STEPS.filter(s => st[s.page]).length;
    const allDone = hasDataset && done === STEPS.length;

    // 刻度：完成 = 主色，未完成 = 灰；无数据集时全灰。纯展示，不可点。
    ticks.innerHTML = STEPS.map(s =>
      `<i${(hasDataset && st[s.page]) ? ' class="on"' : ''}></i>`).join('');
    count.textContent = `${done}/${STEPS.length}`;

    if (!hasDataset) {
      go.dataset.target = 'upload';
      go.textContent = '前往数据上传 →';
    } else if (allDone) {
      go.dataset.target = 'report';
      go.textContent = '前往报告中心 →';
    } else {
      const next = STEPS.find(s => !st[s.page]);
      go.dataset.target = next.page;
      go.textContent = `前往${next.label} →`;
    }
  }

  function init() {
    const go = $('fgGo');
    if (go) {
      go.addEventListener('click', () => {
        const t = go.dataset.target;
        if (t && window.APP && typeof APP.go === 'function') APP.go(t);
      });
    }
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

  window.FlowGuide = { refresh: render };
})();
