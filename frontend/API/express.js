const express = require('express');
const fetch = require('node-fetch');
const cors = require('cors');
require('dotenv').config();

const app = express();
const port = 3000;

// 允许特定源访问（替换为你的前端地址）
app.use(cors({
  origin: 'http://127.0.0.1:5500',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// 代理路由：将请求转发到DeepSeek API
app.post('/deepseek-proxy', async (req, res) => {
  try {
    const response = await fetch('https://api.deepseek.com/v1', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.DEEPSEEK_API_KEY}`
      },
      body: JSON.stringify(req.body)
    });

    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    res.status(500).json({ error: '代理服务器错误' });
  }
});

app.listen(port, () => {
  console.log(`代理服务器运行在 http://localhost:${port}`);
});