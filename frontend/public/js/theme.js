// medical-data-analysis/frontend/public/js/theme.js
// 设计层的单例：底部状态栏控制器、耗时计时、等宽字体工具类、ECharts 统一主题。
// 说明：本文件只负责「呈现」——数字排版、时间戳、耗时读数，不做任何统计计算。

(function () {
  'use strict';

  // ---------------- 字体与配色（与 style.css 的 token 保持同源） ----------------
  const MONO = '"JetBrains Mono","SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace';
  const SANS = '"PingFang SC","Microsoft YaHei","Segoe UI",system-ui,sans-serif';
  const COLORS = {
    primary: '#0E5E6F', primaryHover: '#0A4A57', primaryWeak: '#E6F2F4',
    amber: '#B45309', ok: '#2F855A', bad: '#C0392B',
    gray: '#9CA3AF', line: '#D1D5DB', lineSoft: '#E5E7EB',
    text: '#111827', dim: '#4B5563', ink: '#374151',
    // 分类序列：深青起手，之后降饱和，避免彩虹色
    seq: ['#0E5E6F', '#B45309', '#2F855A', '#6B7280', '#5E8FA0', '#C0392B', '#8A6A3B', '#4C7A68'],
  };

  /** ECharts 统一基础配置：坐标轴数字走等宽字体 */
  function echartsBase() {
    return {
      textStyle: { fontFamily: SANS, fontSize: 12, color: COLORS.ink },
      // containLabel 让 ECharts 按实际文字宽度自动留白：不同数据集的变量名长度可相差
      // 80px 以上，写死边距必然在某些数据上把轴标签切掉
      grid: { containLabel: true, left: 10, right: 16, top: 36, bottom: 20 },
      tooltip: {
        trigger: 'axis', confine: true,
        backgroundColor: 'rgba(255,255,255,.96)', borderColor: COLORS.line,
        textStyle: { color: COLORS.text, fontSize: 12 },
      },
      animationDuration: 300,
    };
  }

  /** 直角坐标轴：标签与轴名统一，标签等宽 */
  function axisXY(opt) {
    const o = opt || {};
    const common = {
      axisLine: { lineStyle: { color: COLORS.line } },
      axisTick: { lineStyle: { color: COLORS.line } },
      splitLine: { lineStyle: { color: COLORS.lineSoft, type: 'solid' } },
      nameTextStyle: { fontSize: 11, color: COLORS.dim },
      axisLabel: { color: COLORS.dim, fontSize: 11 },
    };
    return Object.assign({}, common, o);
  }

  /**
   * 轴名（axis.name）配置。
   *
   * 默认 nameLocation='end' 会把轴名甩到轴末端**之外**：x 轴名落在绘图区右边界以外
   * （如一图实测越出画布 48.7px），残留半个字符。这里统一改放轴中点：
   *   - x 轴名：居中于轴下方
   *   - y 轴名：竖排居中于轴左侧
   * 两种位置都落在各自边距里，不会越过画布。
   */
  const NAME_GAP = 30;
  function axisName(name, extra) {
    return Object.assign({ name: name, nameLocation: 'middle', nameGap: NAME_GAP,
                           nameTextStyle: { fontSize: 11, color: COLORS.dim } }, extra || {});
  }

  // ---------------- 等宽排版工具类 ----------------
  const pad = (n) => String(n).padStart(2, '0');

  /** 时间读数 HH:MM:SS */
  function clockText(d) {
    const t = d || new Date();
    return `${pad(t.getHours())}:${pad(t.getMinutes())}:${pad(t.getSeconds())}`;
  }
  /** 日期时间读数 YYYY-MM-DD HH:MM */
  function stampText(d) {
    const t = d || new Date();
    return `${t.getFullYear()}-${pad(t.getMonth() + 1)}-${pad(t.getDate())} ${pad(t.getHours())}:${pad(t.getMinutes())}`;
  }
  /** 耗时读数：ms → "3.2s" 或 "1.4s" */
  function durText(ms) {
    if (ms === null || ms === undefined) return '--';
    return (ms / 1000).toFixed(1) + 's';
  }
  /**
   * 数据集指纹：后端 dataset_id 由 uuid4().hex 生成，取其前 8 位作为短标识。
   * 仅用于界面标识与状态栏显示，不参与任何分析计算。
   */
  function fingerprint(id) {
    if (!id) return '--------';
    return String(id).replace(/[^0-9a-fA-F]/g, '').slice(0, 8).toLowerCase() || '--------';
  }
  /** 把字符串包一层等宽 span（调用方已保证转义） */
  function mono(v) { return `<span class="mono">${v}</span>`; }

  // ---------------- 底部状态栏 ----------------
  const state = { timer: null, startedAt: 0, lastMs: null, busy: false };

  function el(id) { return document.getElementById(id); }

  function setState(kind, text) {
    const box = el('sbState'), txt = el('sbText');
    if (!box || !txt) return;
    box.classList.remove('busy', 'err');
    if (kind === 'busy') box.classList.add('busy');
    if (kind === 'err') box.classList.add('err');
    txt.textContent = text;
  }

  function stopTimer() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.busy = false;
  }

  /** 进入忙态：状态栏左侧开始滚动耗时 */
  function beginBusy(label) {
    stopTimer();
    state.busy = true;
    state.startedAt = performance.now();
    const base = label || '分析中';
    setState('busy', `${base}… 0.0s`);
    state.timer = setInterval(() => {
      const ms = performance.now() - state.startedAt;
      setState('busy', `${base}… ${durText(ms)}`);
      syncPill(base, ms);
    }, 100);
    const pill = el('loading');
    if (pill) pill.hidden = false;
    syncPill(base, 0);
  }

  function syncPill(base, ms) {
    const tx = document.querySelector('#loading .lb-text');
    if (tx) tx.textContent = `${base}… ${durText(ms)}`;
  }

  /** 结束忙态：显示"完成，用时 x.xs"，并把读数保留为"上次分析耗时" */
  function endBusy(label) {
    if (!state.busy) return;
    const ms = performance.now() - state.startedAt;
    stopTimer();
    state.lastMs = ms;
    const pill = el('loading');
    if (pill) pill.hidden = true;
    setState('idle', `${label || '完成'}，用时 ${durText(ms)}`);
  }

  /** 失败收尾：保留读数，状态转 err */
  function failBusy(msg) {
    if (state.busy) {
      const ms = performance.now() - state.startedAt;
      stopTimer();
      state.lastMs = ms;
      const pill = el('loading');
      if (pill) pill.hidden = true;
    }
    setState('err', msg || '失败');
  }

  /** 回到就绪态，并把上次耗时并入文案 */
  function idle() {
    stopTimer();
    const pill = el('loading');
    if (pill) pill.hidden = true;
    setState('idle', state.lastMs !== null ? `就绪 · 上次分析耗时 ${durText(state.lastMs)}` : '就绪');
  }

  /** 数据集信息：指纹 + 行×列 */
  function setDataset(ds) {
    const fp = el('sbFp'), dims = el('sbDims');
    if (!ds) {
      if (fp) fp.textContent = '--------';
      if (dims) dims.textContent = '-- × --';
      return;
    }
    if (fp) fp.textContent = fingerprint(ds.id);
    if (dims) dims.textContent = `${ds.n_rows} × ${ds.n_cols}`;
  }

  function startClock() {
    const tick = () => { const c = el('sbClock'); if (c) c.textContent = clockText(); };
    tick();
    setInterval(tick, 1000);
  }

  function loadVersion() {
    API.get('/api/health').then((r) => {
      const v = el('sbVer');
      if (v) v.textContent = 'v' + (r.version || '--');
    }).catch(() => {
      const v = el('sbVer');
      if (v) v.textContent = '离线';
    });
  }

  function init() {
    startClock();
    loadVersion();
    idle();
  }

  window.Theme = {
    MONO, SANS, COLORS, NAME_GAP, echartsBase, axisXY, axisName,
    clockText, stampText, durText, fingerprint, mono,
    beginBusy, endBusy, failBusy, idle, setDataset, init,
    get lastMs() { return state.lastMs; },
  };
})();
