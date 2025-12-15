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
import time  # 必须显式导入
# 在顶部添加
from weibo_login import WeiboCookieFetcher

class LLM_INTERFACE(ABC):
    @abstractmethod
    def send_message(self, prompt: str, json_flag: bool = False) -> str:
        pass

@dataclass
class NodeConfig:
    node_id: int
    node_type: str
    node_name: str
    input_map: Dict[str, str]
    choice_map: Dict[str, str]
    attrs: Dict[str, Any]


class WorkflowNode(ABC):
    def __init__(self, config: NodeConfig, workflow_context):
        self.config = config
        self.workflow_context = workflow_context
        self.attrs = config.attrs or {}

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class WeiboCrawler:
    def __init__(self, cookie: str = None):
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



class ReceiveInputNode(WorkflowNode):
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"📥 接收到输入关键词: {input_data['keyword']}")
        return {"keyword": input_data["keyword"], "timestamp": datetime.now().isoformat()}


class WeiboCrawlNode(WorkflowNode):
    def __init__(self, config: NodeConfig, workflow_context):
        super().__init__(config, workflow_context)
        # 优先使用节点属性中的 cookie，否则用环境变量或全局硬编码
        self.cookie = self.attrs.get("cookie") or os.getenv("WEIBO_COOKIE") or WEIBO_COOKIE
        if not self.cookie:
            raise ValueError("⚠️ 错误：未提供 WEIBO_COOKIE，无法爬取微博数据")
        self.crawler = WeiboCrawler(cookie=self.cookie)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        keyword = input_data.get("keyword")
        if not keyword:
            return {"success": False, "posts": [], "error": "缺少关键词"}

        # 检查是否是测试推送，如果是则爬取更多页数，收集更多数据
        push_time = input_data.get("push_time", "normal")
        if push_time == "test_10s":
            max_pages = 5  # 测试时爬取5页，收集更多数据
            print(f"🔥 测试模式: 开始深度爬取微博数据 (5页): {keyword}")
        else:
            max_pages = 2  # 正常模式2页
            print(f"🕷️ 开始爬取微博数据: {keyword}")

        posts = self.crawler.search_posts(keyword, max_pages=max_pages)

        if not posts:
            return {"success": False, "posts": [], "error": f"未找到关于 '{keyword}' 的微博信息"}

        print(f"📌 成功获取 {len(posts)} 条微博")

        # 将微博数据保存为JSON文件
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"./weibo_data_{keyword}_{timestamp}.json"

        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
            print(f"💾 微博数据已保存到: {json_filename}")
        except Exception as e:
            print(f"🚨 保存微博数据失败: {e}")

        return {"success": True, "posts": posts, "data_file": json_filename}


class LLMSummarizeNode(WorkflowNode):
    def __init__(self, config: NodeConfig, workflow_context):
        super().__init__(config, workflow_context)
        self.llm_client = workflow_context["llm_client"]
        self.topic_memory = workflow_context.get("topic_memory")  # 获取话题记忆管理器

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        posts = input_data.get("posts", [])
        keyword = input_data.get("keyword", "未知话题")

        if not posts:
            return {"summary": "无可用内容进行总结"}

        texts = "\n".join([f"{p['username']}: {p['text']}" for p in posts])

        # 获取历史记忆
        history_summaries = []
        if self.topic_memory:
            history_summaries = self.topic_memory.get_topic_history(keyword, limit=3)
            print(f"🧠 找到 {len(history_summaries)} 条历史记录作为分析参考")

        # 构建历史记忆上下文
        history_context = ""
        if history_summaries:
            history_list = "\n".join([f"• {summary}" for summary in history_summaries])
            history_context = f"\n\n📚 历史分析记录（最近3次）：\n{history_list}"

        prompt_file = "./prompt/summarize_weibo.md"
        os.makedirs("./prompt", exist_ok=True)

        if not os.path.exists(prompt_file):
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write("""你是一位专业的社交媒体分析专家，请根据以下从微博获取的真实用户发言，对"{{ keyword }}"话题进行深入分析。

## 分析要求：
1. **内容定位**：基于发言内容自动判断这是（金融/投资话题）还是（新闻事件/社会热点），选择合适的分析框架

2. **智能分析**：
   - **金融领域**（如股票、黄金、外汇、基金等）：重点分析舆论对价格走势的影响、市场情绪预测、投资建议倾向
   - **新闻事件**（如火灾、事故、政策、社会事件等）：重点梳理事件脉络、目前发展状况、公众关注焦点

3. **输出结构**：
   - 📊 **事件概述**：发生了什么，目前状态如何
   - 💭 **舆论态度**：大家的态度是乐观/悲观，中性占主导吗
   - 🔥 **关键热点**：最受关注的几个讨论点
   - 📈 **趋势展望**（金融话题）：市场或价格可能会如何发展

4. **呈现方式**：清新易懂，简洁不啰嗦，不超过400字

**微博发言数据**：
{% for post in weibo_posts %}
- [{{ post.username }} @ {{ post.date }}]：{{ post.text }}
{% endfor %}
""")
            print(f"✅ 提示词文件创建: {prompt_file}")

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        final_prompt = prompt_template \
            .replace("{{ keyword }}", keyword) \
            .replace("{% for post in weibo_posts %}", "") \
            .replace("{% endfor %}", "") \
            .replace("{{content}}", texts[:7500] + history_context)  # 组合微博内容和历史记忆

        print("🧠 正在调用 LLM 生成摘要...")
        summary = self.llm_client.send_message(final_prompt)

        # 将新的总结存储到记忆中
        if self.topic_memory and summary and len(summary.strip()) > 20:
            self.topic_memory.add_topic_summary(keyword, summary)
            print(f"💾 新总结已存储到话题记忆: {keyword}")

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
            "content": {"text": f"【今日关注点简报】\n\n{summary}\n\n---\n🤖 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
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


# ================== 每日数据累积器 ==================
class DailyDataCollector:
    def __init__(self, data_dir="./daily_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def get_daily_file(self, date_str: str) -> str:
        """获取当天的累积数据文件路径"""
        return os.path.join(self.data_dir, f"daily_{date_str}.json")

    def load_daily_data(self, date_str: str) -> Dict[str, Any]:
        """加载当天的累积数据"""
        file_path = self.get_daily_file(date_str)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_daily_data(self, date_str: str, data: Dict[str, Any]):
        """保存当天的累积数据"""
        file_path = self.get_daily_file(date_str)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存每日数据失败: {e}")

    def append_posts(self, date_str: str, keyword: str, posts: List[Dict]):
        """向当天数据累积中添加新爬取的帖子"""
        daily_data = self.load_daily_data(date_str)

        if keyword not in daily_data:
            daily_data[keyword] = {"posts": [], "total_count": 0, "timestamps": []}

        # 检查是否重复（相同URL）
        existing_urls = {post['url'] for post in daily_data[keyword]["posts"]}

        new_posts = []
        for post in posts:
            if post['url'] not in existing_urls:
                new_posts.append({
                    **post,
                    "collected_at": datetime.now().isoformat(),
                    "batch_hour": datetime.now().hour
                })

        daily_data[keyword]["posts"].extend(new_posts)
        daily_data[keyword]["total_count"] = len(daily_data[keyword]["posts"])
        daily_data[keyword]["timestamps"].append(datetime.now().isoformat())

        # 只保留最近5批次的时间戳
        if len(daily_data[keyword]["timestamps"]) > 5:
            daily_data[keyword]["timestamps"] = daily_data[keyword]["timestamps"][-5:]

        self.save_daily_data(date_str, daily_data)

        print(f"📚 {keyword} 当日累积数据: 新增 {len(new_posts)} 条，累计 {daily_data[keyword]['total_count']} 条")
        return daily_data

    def get_daily_posts(self, date_str: str, keyword: str) -> List[Dict]:
        """获取关键词当天的累积帖子"""
        daily_data = self.load_daily_data(date_str)
        return daily_data.get(keyword, {}).get("posts", [])

    def cleanup_old_data(self, days_to_keep: int = 7):
        """清理超过一定天数的数据文件"""
        import glob
        current_date = datetime.now()
        files = glob.glob(os.path.join(self.data_dir, "daily_*.json"))

        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                # 提取日期部分 (daily_20251130.json -> 20251130)
                date_str = filename.replace("daily_", "").replace(".json", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")

                if (current_date - file_date).days > days_to_keep:
                    os.remove(file_path)
                    print(f"🗑️ 删除过期每日数据文件: {filename}")
            except:
                continue

# ================== 话题记忆管理器 ==================
class TopicMemoryManager:
    def __init__(self, memory_file="./topic_memories.json"):
        self.memory_file = memory_file
        self.memories = self.load_memories()

    def load_memories(self):
        """加载话题记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_memories(self):
        """保存话题记忆"""
        try:
            os.makedirs(os.path.dirname(self.memory_file) if os.path.dirname(self.memory_file) else '.', exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存记忆失败: {e}")

    def get_topic_history(self, keyword: str, limit: int = 5) -> List[str]:
        """获取话题的历史总结，按时间倒序"""
        if keyword not in self.memories:
            return []
        history = self.memories[keyword]
        # 按时间排序最新的在前面
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return [item['summary'] for item in history[:limit]]

    def add_topic_summary(self, keyword: str, summary: str):
        """添加话题的新总结"""
        if keyword not in self.memories:
            self.memories[keyword] = []

        timestamp = datetime.now().isoformat()
        self.memories[keyword].append({
            'timestamp': timestamp,
            'summary': summary,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # 只保留最近20条记录
        if len(self.memories[keyword]) > 20:
            self.memories[keyword] = self.memories[keyword][-20:]

        self.save_memories()

# ================== 工作流管理器 ==================
class WorkflowManager:
    def __init__(self, api_key: str, prompt_folder: str, memory_path: str):
        self.api_key = api_key
        self.prompt_folder = prompt_folder
        self.memory_path = memory_path
        self.nodes = {}
        self.workflow_context = {
            "llm_client": SimpleLLMClient(api_key),
            "memory": self.load_memory(),
            "topic_memory": TopicMemoryManager(),  # 添加话题记忆管理器
            "daily_data": DailyDataCollector()  # 添加每日数据累积器
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
        # ✅ 使用阿里云 DashScope 的 OpenAI 兼容接口
        self.endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = "deepseek-r1"  # 也支持 qwen-plus, qwen-turbo 等

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
            # 启用 JSON 输出模式（需模型支持）
            payload["response_format"] = {"type": "json_object"}

        try:
            print("🧠 正在请求阿里云 Qwen...")  # 调试信息
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

# ================== 主程序入口 ==================
if __name__ == "__main__":
    # ========== ⚠️ 请替换为你的飞书 Webhook（可选）==========
    FEISHU_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/77bf45ee-a658-4476-ac7e-cf3e9f538fae"  # 可留空跳过推送

    # ========== ✅ 你的 Cookie（已正确加载）==========
    WEIBO_COOKIE = "__itrace_wid=5183d80c-8409-4bd6-aadf-b6cb3913f006; XSRF-TOKEN=n_M1OEoDnlMf_3KDZkBDs_u-; cross_origin_proto=SSL; _s_tentry=security.weibo.com; Apache=8514952670743.361.1764319483545; SINAGLOBAL=8514952670743.361.1764319483545; ULV=1764319483548:1:1:1:8514952670743.361.1764319483545:; ALF=02_1766912926; SCF=AtJxDRAJhb0kFq4S0x0diFZ5wYp67yN-uAqb1OC2du4Drb06vHPi-Sdd-pueG56Th6IHkUYoNz8tS28tXJSQXX8.; SUB=_2A25ELRbODeRhGeFN6lAW9y7EyTuIHXVnQxYGrDV8PUJbkNANLUfRkW1NQHnFhJxR7tmM7tC1woWzPSYQ_FVzKrqz; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5zhvkUBHraz8XPXb2gWpk.5NHD95QNe02ES0M71hzNWs4Dqcjwi--4iK.4iKnRi--ci-zEiK.7i--Ri-2RiKn7qJLf; WBPSESS=Dt2hbAUaXfkVprjyrAZT_EqNgjBcBvTShE0WQ5uUendglafG2tHDNXo7heXuRfiNnRmnpPgFQSfMCuBEch1qHc8lsGhlcV78IdZmZjWzp8l1gzq1JjECNPPzf4gQJvY9R_WYaeOSHWeYeIBzuIF4uNHoSW1Ujeer16JoAcFcylOYG6GIMr7aVlpKxhOSQAYguKyMbbbOX08ig6HmWpkjmw=="

    # 设置环境变量（推荐方式）
    os.environ["WEIBO_COOKIE"] = WEIBO_COOKIE
    if FEISHU_WEBHOOK:
        os.environ["FEISHU_WEBHOOK"] = FEISHU_WEBHOOK

    # 创建目录
    os.makedirs("./prompt", exist_ok=True)
    os.makedirs("./workflow_memory", exist_ok=True)

    # 初始化管理器
    manager = WorkflowManager(
        api_key="sk-addb15e06fef4c19a46122a39aac8caa",  # 替换为你自己的 API Key
        prompt_folder="./prompt",
        memory_path="./workflow_memory/memory.json"
    )

    # 注册节点（模拟流程图）
    nodes_config = [
    {
        "node_id": 1,  # ← 改这里！
        "node_type": "receive_input",
        "node_name": "receive_01",
        "input_map": {},
        "choice_map": {"default": "weibo_crawl_01"},
        "attrs": {}
    },
    {
        "node_id": 2,  # ← 改这里！
        "node_type": "weibo_crawl",
        "node_name": "weibo_crawl_01",
        "input_map": {"keyword": "receive_01.keyword"},
        "choice_map": {"default": "llm_summarize"},
        "attrs": {
            "cookie": WEIBO_COOKIE
        }
    },
    {
        "node_id": 3,  # ← 改这里！
        "node_type": "llm_summarize",
        "node_name": "llm_summarize",
        "input_map": {"posts": "weibo_crawl_01.posts", "keyword": "receive_01.keyword"},
        "choice_map": {"default": "feishu_notify"},
        "attrs": {}
    },
    {
        "node_id": 4,  # ← 改这里！
        "node_type": "feishu_notify",
        "node_name": "feishu_notify",
        "input_map": {"summary": "llm_summarize.summary"},
        "choice_map": {},
        "attrs": {
            "webhook": FEISHU_WEBHOOK
        }
    }
]

    for cfg in nodes_config:
        config = NodeConfig(**cfg)
        manager.register_node(config)

    # 定义工作流顺序
    flow = [
        {"node_name": "receive_01", "input_map": {"keyword": "input.keyword"}},
        {"node_name": "weibo_crawl_01", "input_map": {"keyword": "receive_01.keyword"}},
        {"node_name": "llm_summarize", "input_map": {"posts": "weibo_crawl_01.posts", "keyword": "receive_01.keyword"}},
        {"node_name": "feishu_notify", "input_map": {"summary": "llm_summarize.summary"}}
    ]

    # 执行测试
    test_inputs = {"keyword": "黄金价格今天"}
    print(f"🔄 开始执行工作流，关键词: {test_inputs['keyword']}")
    result = manager.run_workflow(test_inputs, flow)

    if result:
        print("\n🎉 工作流执行成功！")
        if "summary" in result:
            print("\n📋 分析结果:\n" + result["summary"])
    else:
        print("\n💥 工作流执行失败，请检查日志")
