# main_service.py
from fastapi import FastAPI, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import threading
import random
import time
from weibo_login import WeiboCookieFetcher
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import uvicorn

app = FastAPI(title="微博舆情分析服务", version="1.0")

# 静态文件目录
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 配置文件路径
CONFIG_FILE = "data/config.json"
COOKIE_FILE = "data/weibo_cookie.pkl"

# 全局配置
config = {
    "weibo_cookie": "",
    "feishu_webhook": "",
    "keywords": [],
    "enabled": True
}

# 读取配置
def load_config():
    global config
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config.update(json.load(f))

# 保存配置
def save_config():
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

# 加载配置
load_config()

# 微博爬虫类
class WeiboCrawler:
    def __init__(self, cookie: str = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://s.weibo.com/weibo",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin"
        })
        if cookie:
            self.session.headers["Cookie"] = cookie
            print("🍪 Cookie 已加载")

    def search_posts(self, keyword: str, max_pages: int = 2) -> List[Dict[str, str]]:
        base_url = "https://s.weibo.com/weibo"
        results = []

        for page in range(1, max_pages + 1):
            params = {"q": keyword, "page": page}
            try:
                print(f"🔍 正在请求第 {page} 页: {keyword}")
                response = self.session.get(
                    base_url,
                    params=params,
                    timeout=10,
                    allow_redirects=True
                )

                if response.status_code != 200:
                    print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                    continue

                text = response.text

                if "passport.weibo.com" in response.url or "login" in response.url:
                    raise Exception("登录失效，请更新 Cookie")
                if "验证" in text or "请开启 JavaScript" in text or "检查浏览器" in text:
                    raise Exception("触发反爬机制，请更换 IP 或使用代理")

                soup = BeautifulSoup(text, "lxml")

                cards = soup.find_all("div", class_="card-wrap")
                extracted = 0

                for card in cards:
                    script_tag = card.find("script")
                    if script_tag and "hotsearch" in script_tag.text:
                        continue

                    user_elem = card.find("a", class_="name")
                    username = user_elem.get_text(strip=True) if user_elem else "未知用户"

                    content_elem = card.find("p", class_=lambda x: x and "txt" in x)
                    if not content_elem:
                        continue
                    content = content_elem.get_text(strip=True).replace("收起全文", "").strip()
                    if len(content) < 5:
                        continue

                    date_elem = card.find("p", class_="from")
                    date = "未知时间"
                    url = ""
                    if date_elem and date_elem.find("a"):
                        date = date_elem.find("a").get_text(strip=True)
                        href = date_elem.find("a").get("href")
                        url = f"https://s.weibo.com{href}" if href.startswith("/") else href

                    results.append({
                        "username": username,
                        "text": content,
                        "date": date,
                        "source": "weibo",
                        "url": url
                    })
                    extracted += 1

                print(f"✅ 第 {page} 页提取到 {extracted} 条微博")
                time.sleep(random.uniform(2.5, 4.0))

            except Exception as e:
                print(f"🚨 爬取失败: {e}")
                break

        return results[:30]

# LLM 客户端
class SimpleLLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = "qwen-plus"

    def send_message(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }

        try:
            print("🧠 正在请求 LLM...")
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print("✅ LLM 返回成功")
                return content
            else:
                error_msg = response.text
                print(f"❌ LLM 请求失败: {response.status_code} {error_msg}")
                return f"[LLM 错误] {response.status_code}: {error_msg}"

        except Exception as e:
            print(f"🚨 LLM 请求异常: {str(e)}")
            return f"[请求异常] {str(e)}"

# 分析关键词
def analyze_keyword(keyword: str, api_key: str):
    if not config["weibo_cookie"]:
        print("❌ 未配置微博 Cookie，跳过分析")
        return

    crawler = WeiboCrawler(cookie=config["weibo_cookie"])
    posts = crawler.search_posts(keyword, max_pages=2)

    if not posts:
        print(f"❌ 未找到关于 '{keyword}' 的微博信息")
        return

    texts = "\n".join([f"{p['username']}: {p['text']}" for p in posts])
    prompt = f"""
你是一个舆情分析专家，请根据以下微博内容，总结公众对【{keyword}】的看法。
要求：
1. 分点列出主要观点（至少3点）
2. 每个观点附上代表性原句（引用用户名+内容）
3. 总结整体情绪倾向（乐观/悲观/中立）
4. 使用中文输出

微博内容：
{texts[:8000]}

请按上述格式回答。
"""
    
    llm_client = SimpleLLMClient(api_key)
    summary = llm_client.send_message(prompt)

    # 推送飞书
    if config["feishu_webhook"]:
        msg = {
            "msg_type": "text",
            "content": {"text": f"【舆情分析报告】{keyword}\n\n{summary}\n\n---\n🤖 自动推送于 {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        try:
            requests.post(config["feishu_webhook"], json=msg, timeout=5)
            print("✅ 已推送到飞书")
        except Exception as e:
            print(f"❌ 推送失败: {e}")

# 定时任务
def scheduled_analysis():
    if not config["enabled"] or not config["keywords"]:
        print("❌ 任务被禁用或无关键词，跳过")
        return

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("❌ 未配置 DASHSCOPE_API_KEY，跳过分析")
        return

    print(f"🔄 开始定时分析 {len(config['keywords'])} 个关键词...")
    for keyword in config["keywords"]:
        print(f"🔍 分析关键词: {keyword}")
        analyze_keyword(keyword, api_key)
        time.sleep(5)  # 避免请求过频

# 启动定时任务
scheduler = BackgroundScheduler()
scheduler.add_job(
    scheduled_analysis,
    CronTrigger(hour=9, minute=0),  # 每天上午9点执行
    id='daily_analysis'
)
scheduler.start()

# API 接口
@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>微博舆情分析配置</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .status { margin-top: 20px; padding: 10px; border-radius: 4px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>微博舆情分析配置</h1>
            
            <form id="configForm">
                <div class="form-group">
                    <label>飞书 Webhook URL:</label>
                    <input type="text" id="feishu_webhook" name="feishu_webhook" placeholder="https://open.feishu.cn/...">
                </div>
                
                <div class="form-group">
                    <label>监控关键词 (每行一个):</label>
                    <textarea id="keywords" name="keywords" rows="5" placeholder="黄金价格
比特币
新能源汽车"></textarea>
                </div>
                
                <div class="form-group">
                    <label>阿里云 API Key:</label>
                    <input type="password" id="api_key" name="api_key" placeholder="sk-...">
                </div>
                
                <button type="submit">保存配置</button>
            </form>
            
            <div style="margin-top: 30px;">
                <h3>微博账号绑定</h3>
                <button onclick="startWeiboLogin()">扫码绑定微博账号</button>
                <div id="loginStatus" style="margin-top: 10px;"></div>
            </div>
            
            <div style="margin-top: 30px;">
                <h3>服务状态</h3>
                <div id="status"></div>
            </div>
        </div>

        <script>
            // 加载现有配置
            fetch('/api/config')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('feishu_webhook').value = data.feishu_webhook || '';
                    document.getElementById('keywords').value = data.keywords ? data.keywords.join('\\n') : '';
                });

            // 保存配置
            document.getElementById('configForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const config = {
                    feishu_webhook: formData.get('feishu_webhook'),
                    keywords: formData.get('keywords').split('\\n').filter(k => k.trim()),
                    api_key: formData.get('api_key')
                };
                
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                
                if (response.ok) {
                    alert('配置保存成功！');
                } else {
                    alert('保存失败！');
                }
            });

            // 获取服务状态
            function updateStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').innerHTML = `
                            <p>定时任务: ${data.scheduler_running ? '运行中' : '已停止'}</p>
                            <p>上次执行: ${data.last_run || '从未执行'}</p>
                            <p>关键词数量: ${data.keyword_count}</p>
                            <p>Cookie状态: ${data.has_cookie ? '已绑定' : '未绑定'}</p>
                        `;
                    });
            }
            setInterval(updateStatus, 5000);
            updateStatus();

            // 扫码登录
            async function startWeiboLogin() {
                const statusDiv = document.getElementById('loginStatus');
                statusDiv.innerHTML = '<p style="color: blue;">正在启动浏览器...</p>';
                
                try {
                    const response = await fetch('/api/weibo-login', { method: 'POST' });
                    const result = await response.json();
                    
                    if (result.success) {
                        statusDiv.innerHTML = '<p style="color: green;">扫码登录成功！</p>';
                    } else {
                        statusDiv.innerHTML = `<p style="color: red;">登录失败: ${result.error}</p>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<p style="color: red;">请求失败: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/config")
async def get_config():
    return {
        "feishu_webhook": config["feishu_webhook"],
        "keywords": config["keywords"],
        "has_cookie": bool(config["weibo_cookie"])
    }

@app.post("/api/config")
async def save_config_api(data: dict):
    config["feishu_webhook"] = data.get("feishu_webhook", "")
    config["keywords"] = [k.strip() for k in data.get("keywords", []) if k.strip()]
    save_config()
    
    # 保存 API Key 到环境变量
    if data.get("api_key"):
        os.environ["DASHSCOPE_API_KEY"] = data["api_key"]
    
    return {"success": True}

@app.post("/api/weibo-login")
async def weibo_login():
    try:
        fetcher = WeiboCookieFetcher(cookie_file=COOKIE_FILE)
        cookie = fetcher.get_cookie()
        
        if cookie:
            config["weibo_cookie"] = cookie
            save_config()
            return {"success": True}
        else:
            return {"success": False, "error": "扫码登录失败或超时"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/status")
async def get_status():
    return {
        "scheduler_running": scheduler.running,
        "last_run": None,  # 可以添加实际的执行记录
        "keyword_count": len(config["keywords"]),
        "has_cookie": bool(config["weibo_cookie"])
    }

# 启动服务
if __name__ == "__main__":
    print("🚀 启动微博舆情分析服务...")
    print("🌐 访问 http://localhost:8000 配置服务")
    uvicorn.run(app, host="0.0.0.0", port=8000)