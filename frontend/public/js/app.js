// medical-data-analysis/frontend/public/js/app.js
// 应用框架：全局状态、导航切换、顶栏内联数据集信息、AI 状态、底部状态栏接线。

(function () {
  'use strict';

  const APP = {
    dataset: null,          // {id, name, n_rows, n_cols, variables, validation}
    cleaning: null,
    analysis: { outcome: null, predictors: [], covariates: [], task: 'auto' },
    results: { describe: null, screen: null, univariate: [], multivariable: null, ml: null },
    model: null,
    forceContinue: false,
    // 流程步骤完成状态 —— 唯一一份。导航 ✓ 与侧栏流程引导条都从这里渲染（同源），
    // 写入只走 markDone()，禁止任何页面自行维护第二套判定。
    steps: {},
  };
  window.APP = APP;

  const PAGES = ['upload', 'confirm', 'cleaning', 'config', 'automl', 'dashboard', 'prediction', 'report', 'settings'];
  // 前端版本唯一出处：设置页「关于」。侧栏不再重复展示版本 / 模式字样（状态栏与设置页已有）。
  const FE_VERSION = 'v1.3.2';

  function setDataset(payload) {
    APP.dataset = {
      id: payload.dataset_id, name: payload.name,
      n_rows: payload.n_rows, n_cols: payload.n_cols,
      variables: payload.variables, validation: payload.validation,
      parse_meta: payload.parse_meta,
    };
    APP.cleaning = null;
    APP.analysis = { outcome: null, predictors: [], covariates: [], task: 'auto' };
    APP.results = { describe: null, screen: null, univariate: [], multivariable: null, ml: null };
    APP.model = null;
    APP.forceContinue = false;
    refreshTopbar();
    if (window.Sidebar) Sidebar.refresh();
    // 切换数据集后旧模型失效，侧栏模型灯随之回到灰显
    if (window.LinkStatus) LinkStatus.onDatasetChanged(payload.dataset_id);
    // 流程进度按新数据集重置：旧状态不残留；载入任何数据集后第一步（上传）即算完成
    PAGES.forEach(p => markDone(p, false));
    markDone('upload', true);
  }
  APP.setDataset = setDataset;

  /**
   * 整体缺失率：取后端给出的各字段 missing_rate 的均值。
   * 仅用于顶栏显示；统计结论一律以后端返回值为准。
   */
  function overallMissingRate(d) {
    const vs = (d.variables || []).filter(v => !v.excluded);
    if (!vs.length) return '0.0';
    const sum = vs.reduce((s, v) => s + (typeof v.missing_rate === 'number' ? v.missing_rate : 0), 0);
    return ((sum / vs.length) * 100).toFixed(1);
  }

  const dot = '<span class="dot-sep">·</span>';
  const m = (v) => `<span class="mono">${v}</span>`;

  function refreshTopbar() {
    const bar = document.getElementById('dsBar');
    const d = APP.dataset;
    if (!d) {
      bar.innerHTML = '<span class="none">未加载数据集</span>';
      if (window.Theme) Theme.setDataset(null);
      return;
    }
    bar.innerHTML =
      `<span class="fname">${esc(d.name)}</span>${dot}${m(d.n_rows + ' 行')}${dot}` +
      `${m(d.n_cols + ' 列')}${dot}${m('缺失 ' + overallMissingRate(d) + '%')}`;
    if (window.Theme) Theme.setDataset(d);
  }
  APP.refreshTopbar = refreshTopbar;

  /**
   * 步骤完成状态的唯一写入点：写 APP.steps → 同步渲染导航 ✓ 与侧栏流程引导条。
   * 两处呈现消费同一份数据，任何操作后二者必然一致（同源）。
   */
  function markDone(page, done) {
    APP.steps[page] = !!done;
    const el = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (el) el.classList.toggle('done', !!done);
    if (window.FlowGuide) FlowGuide.refresh();
    if (window.Sidebar) {
      Sidebar.refresh();
      // 跑完一次分析后历史才更新，其余步骤不需要重复拉取
      if (page === 'config' && done) Sidebar.refreshRuns();
    }
  }
  APP.markDone = markDone;

  function go(page) {
    if (PAGES.indexOf(page) < 0) return;
    document.querySelectorAll('.nav-item').forEach(a => a.classList.toggle('active', a.dataset.page === page));
    document.querySelectorAll('.page').forEach(s => s.classList.remove('active'));
    const sec = document.getElementById('page-' + page);
    if (sec) sec.classList.add('active');
    document.getElementById('content').scrollTop = 0;
    const mod = window.Pages && window.Pages[page];
    if (mod && typeof mod.onShow === 'function') {
      try { mod.onShow(); } catch (e) { console.error(e); }
    }
  }
  APP.go = go;

  /** AI 状态统一由侧栏链路状态区展示，这里只触发一次刷新 */
  async function refreshAiStatus() {
    if (window.LinkStatus) LinkStatus.refreshAi();
  }
  APP.refreshAiStatus = refreshAiStatus;

  /**
   * 设置页「运行时 / 分析默认值 / 数据与存储 / 关于」：全部读取后端实际返回值，
   * 取不到就如实显示"读取失败"，不填任何看似合理的假值。
   */
  function fillRuntime(cfg, health) {
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('setApiBase', API.API_BASE);
    set('setMode', (cfg && cfg.mode === 'cloud') ? '云上模式（华为云）' : '本地模式');

    const d = (cfg && cfg.analysis_defaults) || null;
    set('setAlpha', d ? String(d.alpha) : '读取失败');
    set('setFdr', d ? String(d.fdr_method) : '读取失败');
    set('setSeed', d ? String(d.random_seed) : '读取失败');
    set('setUpload', cfg ? `${cfg.max_upload_mb} MB` : '读取失败');

    set('aboutFe', FE_VERSION);
    set('aboutBe', (health && health.version) ? 'v' + health.version : '读取失败');
    set('aboutLicense', 'MIT（见仓库根目录 LICENSE）');
    // 仓库地址由后端环境变量 REPO_URL 提供；未设置时如实显示未配置
    if (cfg) set('aboutRepo', cfg.repository || '未配置（后端未设置 REPO_URL）');
    else set('aboutRepo', '读取失败');

    set('sbVer', (health && health.version) ? 'v' + health.version : '离线');
  }

  function init() {
    if (window.Theme) Theme.init();

    document.getElementById('nav').addEventListener('click', (e) => {
      const item = e.target.closest('.nav-item');
      if (item) go(item.dataset.page);
    });
    document.getElementById('content').addEventListener('click', (e) => {
      const b = e.target.closest('[data-dl]');
      if (b) Charts.download(b.dataset.dl, b.dataset.dl);
    });

    // 运行时信息（模式与版本只在设置页 / 状态栏出现，侧栏不再重复）
    Promise.all([API.get('/api/config').catch(() => null), API.get('/api/health').catch(() => null)])
      .then(([c, h]) => {
        fillRuntime(c, h);
        const banner = document.getElementById('cloudBanner');
        if (c && c.mode === 'cloud' && banner) {
          banner.innerHTML = '<div class="alert warn">' +
            esc((c.storage && c.storage.notice) || '数据将上传至华为云，由使用者确认。') +
            '　历史存 RDS，缓存用 Redis，数据集与模型存 OBS。</div>';
        }
      });

    refreshAiStatus();
    Object.keys(window.Pages || {}).forEach(k => {
      const m = window.Pages[k];
      if (m && typeof m.init === 'function') { try { m.init(); } catch (e) { console.error(e); } }
    });
  }

  window.Pages = window.Pages || {};
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
