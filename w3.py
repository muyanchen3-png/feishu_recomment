# workflow_system.py

from abc import ABC, abstractmethod
import json
import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import threading
import random
import time
import pickle
from dotenv import load_dotenv  # 新增：环境变量管理

# ========== 加载环境变量 ==========
load_dotenv()

# ================== 抽象接口定义 ==================
class LLM_INTERFACE(ABC):
    @abstractmethod
    def send_message(self, prompt: str, json_flag: bool = False) -> str:
        pass


# ================== 节点配置类 ==================
@dataclass
class NodeConfig:
    node_id: int
    node_type: str
    node_name: str
    input_map: Dict[str, str]
    choice_map: Dict[str, str]
    attrs: Dict[str, Any]


# ================== Selenium 登录助手 ==================
class WeiboLoginHelper:
    """
    使用 Selenium 自动登录微博，获取并保存 Cookie
    """

    def __init__(self, username: str, password: str, cookie_path: str = "weibo_cookie.pkl"):
        self.username = username
        self.password = password
        self.cookie_path = cookie_path
        self.driver = None

    def create_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        options.add_argument("--start-maximized")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--remote-allow-origins=*")
        # 可选：无头模式（部署时启用）
        # options.add_argument('--headless')

        # 防止被识别为自动化工具
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => false});"
        })
        return driver

    def login(self) -> str:
        try:
            self.driver = self.create_driver()
            self.driver.get("https://weibo.com/login.php")
            print("🌐 打开微博登录页...")

            time.sleep(3)

            # 输入账号密码
            print("📝 正在输入账号密码...")
            self.driver.find_element("name", "username").send_keys(self.username)
            self.driver.find_element("name", "password").send_keys(self.password)
            login_btn = self.driver.find_element("xpath", '//div[@class="info_list login_btn"]/a')
            login_btn.click()

            time.sleep(5)

            # 等待登录完成（可人工处理滑块验证码）
            while True:
                current_url = self.driver.current_url
                if "myprofile" in current_url or "weibo.com/u" in current_url or "home" in current_url:
                    print("✅ 登录成功！")
                    break
                else:
                    print("⏳ 请手动完成验证码或扫码登录...")
                    time.sleep(3)

            # 获取 Cookies
            cookies = self.driver.get_cookies()
            with open(self.cookie_path, "wb") as f:
                pickle.dump(cookies, f)
            print(f"💾 Cookie 已保存至 {self.cookie_path}")

            # 转成字符串格式供 requests 使用
            cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
            return cookie_str

        except Exception as e:
            print(f"🚨 登录失败: {e}")
            return None
        finally:
            try:
                self.driver.quit()
            except:
                pass

    def load_cookie(self) -> str:
        """从本地加载 Cookie"""
        if os.path.exists(self.cookie_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(self.cookie_path))
            # 超过 12 小时认为过期
            if (datetime.now() - file_time).total_seconds() > 12 * 3600:
                print("🕒 Cookie 已过期，将重新登录")
                return None

            with open(self.cookie_path, "rb") as f:
                cookies = pickle.load(f)
            print("🍪 已加载本地 Cookie")
            return "; ".join([f"{c['name']}={c['value']}" for c in cookies])
        return None


# ================== 微博爬虫实现（增强版）==================
class WeiboCrawler:
    def __init__(self, cookie: str = None, username: str = None, password: str = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://s.weibo.com/weibo?q=%E9%BB%84%E9%87%91%E4%BB%B7%E6%A0%BC",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin"
        })

        # 自动获取 Cookie
        self.cookie = cookie or self._auto_login(username, password)
        if not self.cookie:
            raise ValueError("⚠️ 无法获取有效 Cookie，请检查账号密码或网络环境")
        self.session.headers["Cookie"] = self.cookie
        print("🍪 Cookie 加载成功")

    def _auto_login(self, username: str, password: str) -> str:
        if not username or not password:
            raise ValueError("❌ 缺少微博用户名或密码，无法自动登录")

        helper = WeiboLoginHelper(username, password, "weibo_cookie.pkl")
        cookie = helper.load_cookie()
        if not cookie:
            print("🔄 开始自动登录微博...")
            cookie = helper.login()
        return cookie

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

                # 检查是否被重定向或需要验证
                if "passport.weibo.com" in response.url or "login" in response.url:
                    raise Exception("登录失效，请更新 Cookie")
                if "验证" in text or "请开启 JavaScript" in text or "检查浏览器" in text:
                    raise Exception("触发反爬机制，请更换 IP 或使用代理")

                soup = BeautifulSoup(text, "lxml")

                cards = soup.find_all("div", class_="card-wrap")
                extracted = 0

                for card in cards:
                    # 过滤非用户微博（如热搜榜、广告）
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
                time.sleep(random.uniform(2.5, 4.0))  # 控制频率

            except Exception as e:
                print(f"🚨 爬取失败: {e}")
                break  # 出错即停止

        return results[:30]  # 返回最多30条


# ================== 具体节点实现 ==================
class ReceiveInputNode(WorkflowNode):
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"📥 接收到输入关键词: {input_data['keyword']}")
        return {"keyword": input_data["keyword"], "timestamp": datetime.now().isoformat()}


class WeiboCrawlNode(WorkflowNode):
    def __init__(self, config: NodeConfig, workflow_context):
        super().__init__(config, workflow_context)
        # 优先使用节点属性中的 cookie
        self.cookie = self.attrs.get("cookie")
        self.username = self.attrs.get("username") or os.getenv("WEIBO_USERNAME")
        self.password = self.attrs.get("password") or os.getenv("WEIBO_PASSWORD")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        keyword = input_data.get("keyword")
        if not keyword:
            return {"success": False, "posts": [], "error": "缺少关键词"}

        print(f"🕷️ 开始爬取微博数据: {keyword}")
        crawler = WeiboCrawler(
            cookie=self.cookie,
            username=self.username,
            password=self.password
        )
        posts = crawler.search_posts(keyword, max_pages=2)

        if not posts:
            return {"success": False, "posts": [], "error": f"未找到关于 '{keyword}' 的微博信息"}

        print(f"📌 成功获取 {len(posts)} 条微博")
        return {"success": True, "posts": posts}


class LLMSummarizeNode(WorkflowNode):
    def __init__(self, config: NodeConfig, workflow_context):
        super().__init__(config, workflow_context)
        self.llm_client = workflow_context["llm_client"]

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        posts = input_data.get("posts", [])
        if not posts:
            return {"summary": "无可用内容进行总结"}

        texts = "\n".join([f"{p['username']}: {p['text']}" for p in posts])
        prompt_file = "./prompt/summarize_weibo.md"
        os.makedirs("./prompt", exist_ok=True)

        if not os.path.exists(prompt_file):
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write("""你是一个舆情分析专家，请根据以下微博内容，总结公众对【{{keyword}}】的看法。
要求：
1. 分点列出主要观点（至少3点）
2. 每个观点附上代表性原句（引用用户名+内容）
3. 总结整体情绪倾向（乐观/悲观/中立）
4. 使用中文输出

微博内容：
{{content}}

请按上述格式回答。
""")
            print(f"✅ 提示词文件创建: {prompt_file}")

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        final_prompt = prompt_template \
            .replace("{{keyword}}", input_data.get('keyword', '未知话题')) \
            .replace("{{content}}", texts[:8000])  # 截断防止超限

        print("🧠 正在调用 LLM 生成摘要...")
        summary = self.llm_client.send_message(final_prompt)
        return {"summary": summary}


class FeishuNotifyNode(WorkflowNode):
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        summary = input_data.get("summary", "无摘要内容")
        webhook_url = self.attrs.get("webhook") or os.getenv("FEISHU_WEBHOOK")
        if not webhook_url:
            print("❌ 未配置飞书 Webhook，跳过通知")
            return {"notified": False}

        msg = {
            "msg_type": "text",
            "content": {"text": f"【今日舆情简报】\n\n{summary}\n\n---\n🤖 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        try:
            res = requests.post(webhook_url, json=msg, timeout=5)
            if res.status_code == 200:
                print("✅ 已推送到飞书")
                return {"notified": True}
            else:
                print(f"❌ 推送失败: {res.text}")
                return {"notified": False}
        except Exception as e:
            print(f"❌ 推送异常: {e}")
            return {"notified": False}


# ================== 工作流管理器 ==================
class WorkflowManager:
    def __init__(self, api_key: str, prompt_folder: str, memory_path: str):
        self.api_key = api_key
        self.prompt_folder = prompt_folder
        self.memory_path = memory_path
        self.nodes = {}
        self.workflow_context = {
            "llm_client": SimpleLLMClient(api_key),
            "memory": self.load_memory()
        }

    def register_node(self, config: NodeConfig):
        node_type = config.node_type
        if node_type == "receive_input":
            node = ReceiveInputNode(config, self.workflow_context)
        elif node_type == "weibo_crawl":
            node = WeiboCrawlNode(config, self.workflow_context)
        elif node_type == "llm_summarize":
            node = LLMSummarizeNode(config, self.workflow_context)
        elif node_type == "feishu_notify":
            node = FeishuNotifyNode(config, self.workflow_context)
        else:
            raise ValueError(f"不支持的节点类型: {node_type}")
        self.nodes[config.node_name] = node

    def run_workflow(self, inputs: Dict[str, Any], flow_config: List[Dict]):
        data_pool = {"input": inputs}
        last_output = None

        for step in flow_config:
            node_name = step["node_name"]
            node = self.nodes[node_name]

            # 构建输入
            input_data = {}
            for key, src in step["input_map"].items():
                source_node, output_key = src.split(".")
                input_data[key] = data_pool[source_node][output_key]

            # 执行节点
            print(f"\n🚀 执行节点: {node_name}")
            try:
                output = node.execute(input_data)
                data_pool[node_name] = output
                last_output = output
            except Exception as e:
                print(f"❌ 执行失败: 节点 {node_name} 执行失败: {e}")
                return None

        return last_output

    def load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}


# ================== 简易 LLM 客户端（通义千问）==================
class SimpleLLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = "deepseek-r1"

    def send_message(self, prompt: str, json_flag: bool = False) -> str:
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

        if json_flag:
            payload["response_format"] = {"type": "json_object"}

        try:
            print("🧠 正在请求阿里云 Qwen...")
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


# ================== FastAPI 接口 ==================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🌐 启动中：初始化 WorkflowManager...")
    init_workflow()
    yield
    print("👋 关闭中：释放资源...")

app = FastAPI(title="舆情分析工作流 API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 manager
manager = None

class AnalysisRequest(BaseModel):
    keyword: str

def init_workflow():
    global manager

    # 从环境变量读取
    FEISHU_WEBHOOK = 'https://open.feishu.cn/open-apis/bot/v2/hook/77bf45ee-a658-4476-ac7e-cf3e9f538fae'
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    WEIBO_USERNAME = os.getenv("WEIBO_USERNAME")
    WEIBO_PASSWORD = os.getenv("WEIBO_PASSWORD")

    if not DASHSCOPE_API_KEY:
        raise ValueError("❌ 缺少 DASHSCOPE_API_KEY 环境变量")

    os.makedirs("./prompt", exist_ok=True)
    os.makedirs("./workflow_memory", exist_ok=True)

    manager = WorkflowManager(
        api_key=DASHSCOPE_API_KEY,
        prompt_folder="./prompt",
        memory_path="./workflow_memory/memory.json"
    )

    nodes_config = [
        {
            "node_id": 1,
            "node_type": "receive_input",
            "node_name": "receive_01",
            "input_map": {},
            "choice_map": {"default": "weibo_crawl_01"},
            "attrs": {}
        },
        {
            "node_id": 2,
            "node_type": "weibo_crawl",
            "node_name": "weibo_crawl_01",
            "input_map": {"keyword": "receive_01.keyword"},
            "choice_map": {"default": "llm_summarize"},
            "attrs": {
                "username": WEIBO_USERNAME,
                "password": WEIBO_PASSWORD
            }
        },
        {
            "node_id": 3,
            "node_type": "llm_summarize",
            "node_name": "llm_summarize",
            "input_map": {"posts": "weibo_crawl_01.posts", "keyword": "receive_01.keyword"},
            "choice_map": {"default": "feishu_notify"},
            "attrs": {}
        },
        {
            "node_id": 4,
            "node_type": "feishu_notify",
            "node_name": "feishu_notify",
            "input_map": {"summary": "llm_summarize.summary"},
            "choice_map": {},
            "attrs": {"webhook": FEISHU_WEBHOOK}
        }
    ]

    for cfg in nodes_config:
        config = NodeConfig(**cfg)
        manager.register_node(config)

    print("✅ 工作流系统初始化完成")


@app.post("/analyze", summary="启动舆情分析")
async def start_analysis(request: AnalysisRequest):
    if not manager:
        raise HTTPException(status_code=500, detail="工作流未初始化")

    print(f"🌐 收到来自前端的请求：关键词 = {request.keyword}")

    flow = [
        {"node_name": "receive_01", "input_map": {"keyword": "input.keyword"}},
        {"node_name": "weibo_crawl_01", "input_map": {"keyword": "receive_01.keyword"}},
        {"node_name": "llm_summarize", "input_map": {"posts": "weibo_crawl_01.posts", "keyword": "receive_01.keyword"}},
        {"node_name": "feishu_notify", "input_map": {"summary": "llm_summarize.summary"}}
    ]

    try:
        result = manager.run_workflow({"keyword": request.keyword}, flow)
        if result and "summary" in result:
            return {
                "success": True,
                "keyword": request.keyword,
                "summary": result["summary"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"success": False, "error": "分析失败，无结果"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "ok", "time": datetime.now().isoformat()}