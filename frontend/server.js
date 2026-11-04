// medical-data-analysis/frontend/server.js
// Express 托管静态页面；端口与后端地址集中在配置区，可通过环境变量覆盖。

const path = require('path');
const express = require('express');

// ---------------- 配置区 ----------------
const PORT = Number(process.env.PORT || 3001);
const HOST = process.env.HOST || '127.0.0.1';
const BACKEND = process.env.BACKEND_ORIGIN || 'http://127.0.0.1:8002';
// ---------------------------------------

const app = express();
const PUBLIC_DIR = path.join(__dirname, 'public');

app.use(express.static(PUBLIC_DIR, {
  extensions: ['html'],
  // 开发期静态资源每次校验，避免浏览器缓存旧 JS 导致「改了但没生效」
  setHeaders: (res) => res.setHeader('Cache-Control', 'no-cache'),
}));

// 把后端地址注入页面，避免前端硬编码
app.get('/config.js', (req, res) => {
  res.type('application/javascript');
  res.send(`window.APP_CONFIG = ${JSON.stringify({ apiBase: BACKEND })};`);
});

// 单页应用：其余路径统一返回首页
app.get('*', (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

app.listen(PORT, HOST, () => {
  console.log('--------------------------------------------------------');
  console.log(' Medical Data Analysis Platform · 前端已启动');
  console.log(` 访问地址：http://${HOST}:${PORT}`);
  console.log(` 后端接口：${BACKEND}`);
  console.log('--------------------------------------------------------');
});
