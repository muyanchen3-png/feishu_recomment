from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pickle
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import threading
import sys

app = Flask(__name__)
CORS(app)

# 导入w2的工作流，但不运行主程序
from w2 import WorkflowManager, NodeConfig, WeiboCrawler, WeiboCookieFetcher as W2WeiboCookieFetcher
from weibo_login import WeiboCookieFetcher

# 全局存储登录实例
login_instances = {}

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'database': 'weibo_push_service',
    'user': 'root',
    'password': ''  # 空密码，生产环境请设置强密码
}

# SQLite数据库
DB_FILE = './weibo_push_service.db'

# 初始化数据库
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT UNIQUE,
                cookie TEXT,
                feishu_webhook TEXT,
                keywords TEXT,  -- JSON字符串
                push_time TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS push_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                keyword TEXT,
                summary TEXT,
                push_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ SQLite数据库初始化完成")
    except Exception as e:
        print(f"❌ SQLite数据库初始化失败: {e}")

# 简化的手机号登录类（模拟）
class PhoneWeiboLogin:
    def __init__(self, phone: str):
        self.phone = phone

    def login_and_get_cookie(self):
        # 这里应该实现实际的手机号登录逻辑
        # 现在先简化，返回一个虚拟cookie用于开发测试
        print(f"📱 开始手机号登录: {self.phone}")

        # 简化逻辑：直接返回预设cookie（开发测试用）
        mock_cookie = "__itrace_wid=5183d80c-8409-4bd6-aadf-b6cb3913f006; XSRF-TOKEN=test12345678; cross_origin_proto=SSL; _s_tentry=security.weibo.com; Apache=1234567890123456.361.1764319483545; SINAGLOBAL=1234567890123456.361.1764319483545; ULV=173199483548:1:1:1:123456780123548:; ALF=02_1766912926; SCF=AtJxDRAJhb0kFq4S0x0diFZ5wYp67yN-uAqb1OC2du4Drb06vHPi-Sdd-pueG56Th6IHkUYoNz8tS28tXJSQXX8.; SUB=_2A25ELRbODeRhGeFN6lAW9y7EyTuIHXVnQxYGrDV8PUJbkNANLUfRkW1NQHnFhJxR7tmM7tC1woWzPSYQ; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5zhvkUBHraz8XPXb2gWpk.5NHD95QNe02ES0M71hzNWs4Dqcjwi--4iK.4iKnRi--ci-zEiK.7i--Ri-2RiKn7qJLf; WBPSESS=Dt2hbAUaXfkVprjyrAZT_EqNgjBcBvTShE0WQ5uUendglafG2tHDNXo7heXuRfiNnRmnpPgFQSfMCuBEch1qHc8lsGhlcV78IdZmZjWzp8l1gzq1JjECNPPzf4gQJvY9R_WYaeOSHWeYeIBzuIF4uNHoSW1Ujeer16JoAcFcylOYG6GIMr7aVlpKxhOSQAYguKyMbbbOX08ig6HmWpkjmw=="

        print(f"✅ Cookie获取成功: {mock_cookie}")
        return mock_cookie

# API路由
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    phone = data.get('phone')

    if not phone:
        return jsonify({'error': '缺少手机号'}), 400

    try:
        login = PhoneWeiboLogin(phone)
        cookie = login.login_and_get_cookie()

        if cookie:
            return jsonify({'cookie': cookie})
        else:
            return jsonify({'error': '登录失败，请重试'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 为单个用户执行推送任务
def daily_push_user(phone, cookie, webhook, keywords):
    print(f"🔥 开始执行用户测试推送: {phone}")
    try:
        for keyword in keywords:
            # 调用w2的工作流
            manager = WorkflowManager(
                api_key="sk-addb15e06fef4c19a46122a39aac8caa",  # 从环境变量获取
                prompt_folder="./prompt",
                memory_path=f"./workflow_memory/{phone}_memory.json"
            )

            # 注册节点，修改为用户特定的cookie
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
                    "attrs": {"cookie": cookie}
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
                    "attrs": {"webhook": webhook}
                }
            ]

            for cfg in nodes_config:
                config = NodeConfig(**cfg)
                manager.register_node(config)

            flow = [
                {"node_name": "receive_01", "input_map": {"keyword": "input.keyword"}},
                {"node_name": "weibo_crawl_01", "input_map": {"keyword": "receive_01.keyword"}},
                {"node_name": "llm_summarize", "input_map": {"posts": "weibo_crawl_01.posts", "keyword": "receive_01.keyword"}},
                {"node_name": "feishu_notify", "input_map": {"summary": "llm_summarize.summary"}}
            ]

            # 执行工作流
            result = manager.run_workflow({"keyword": keyword}, flow)
            if result and "summary" in result:
                print(f"✅ 用户 {phone} 关键字 {keyword} 推送成功")

        print(f"✅ 用户 {phone} 测试推送完成")

    except Exception as e:
        print(f"🚨 用户 {phone} 测试推送失败: {e}")

@app.route('/api/save_config', methods=['POST'])
def save_config():
    data = request.json
    phone = data.get('phone')
    cookie = data.get('cookie')
    feishu_webhook = data.get('feishu_webhook')
    keywords = data.get('keywords', [])
    push_time = data.get('push_time')

    # 详细检查参数
    missing_params = []
    if not phone:
        missing_params.append('手机号(phone)')
    if not cookie:
        missing_params.append('Cookie(cookie)')
    if not feishu_webhook:
        missing_params.append('飞书Webhook(feishu_webhook)')
    if not keywords or len(keywords) == 0:
        missing_params.append('关键词(keywords)')
    if not push_time:
        missing_params.append('推送时间(push_time)')

    if missing_params:
        error_msg = f'参数不完整，缺少: {", ".join(missing_params)}'
        print(f"❌ 配置保存失败: {error_msg}")
        return jsonify({'error': error_msg}), 400

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users (phone, cookie, feishu_webhook, keywords, push_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (phone, cookie, feishu_webhook, json.dumps(keywords), push_time))
        conn.commit()
        cursor.close()
        conn.close()

        # 如果用户选择的是测试推送，立即执行一次推送
        if push_time == 'test_10s':
            # 添加10秒延迟后执行
            import threading
            import time
            def delayed_push():
                time.sleep(10)
                daily_push_user(phone, cookie, feishu_webhook, keywords)
            # 在新线程中执行延迟测试推送
            threading.Thread(target=delayed_push, daemon=True).start()

        return jsonify({'message': '配置保存成功' if push_time != 'test_10s' else '测试推送已触发，请等待10秒后查看飞书消息'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_cookie', methods=['POST'])
def update_cookie():
    data = request.json
    phone = data.get('phone')
    cookie = data.get('cookie')

    if not phone or not cookie:
        return jsonify({'error': '缺少手机号或Cookie'}), 400

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 检查用户是否存在
        cursor.execute("SELECT id FROM users WHERE phone = ?", (phone,))
        user = cursor.fetchone()

        if not user:
            return jsonify({'error': '用户不存在，请先设置配置'}), 404

        # 更新cookie
        cursor.execute("UPDATE users SET cookie = ? WHERE phone = ?", (cookie, phone))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'message': 'Cookie更新成功'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_file('qianduan.html')


# 定时推送任务 - 支持4小时累积数据模式
def daily_push():
    current_hour = datetime.now().hour
    today_date = datetime.now().strftime("%Y%m%d")

    print(f"⏰ 开始执行定时任务，当前时间: {current_hour}:00")

    # 初始化数据管理器
    import importlib
    spec = importlib.util.spec_from_file_location("w2_module", "./w2.py")
    w2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w2_module)

    # 创建管理器实例
    manager = w2_module.WorkflowManager(
        api_key="sk-addb15e06fef4c19a46122a39aac8caa",
        prompt_folder="./prompt",
        memory_path="./workflow_memory/daily_memory.json"
    )

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT phone, cookie, feishu_webhook, keywords FROM users")
        users = cursor.fetchall()

        for user in users:
            phone, cookie, webhook, keywords_json = user
            keywords = json.loads(keywords_json) if keywords_json else []

            # 数据累积阶段 (8:00, 12:00, 16:00，只收集数据）
            if current_hour in [8, 12, 16]:
                print(f"📚 数据累积阶段: 为用户{phone}收集粉丝关键词数据")
                for keyword in keywords:
                    try:
                        # 创建工作流，只执行爬取和累积
                        crawl_node = {
                            "node_id": 1,
                            "node_type": "receive_input",
                            "node_name": "receive_input",
                            "input_map": {},
                            "choice_map": {},
                            "attrs": {}
                        }

                        weibo_crawl = {
                            "node_id": 2,
                            "node_type": "weibo_crawl",
                            "node_name": "weibo_crawl",
                            "input_map": {"keyword": "receive_input.keyword"},
                            "choice_map": {},
                            "attrs": {"cookie": cookie}
                        }

                        manager.register_node(w2_module.NodeConfig(**crawl_node))
                        manager.register_node(w2_module.NodeConfig(**weibo_crawl))

                        # 获取当日累积数据
                        daily_data = manager.workflow_context["daily_data"]
                        if daily_data:
                            inputs = {"keyword": keyword}
                            result = manager.run_workflow({"keyword": keyword}, [
                                {"node_name": "receive_input", "input_map": {"keyword": "input.keyword"}},
                                {"node_name": "weibo_crawl", "input_map": {"keyword": "receive_input.keyword"}}
                            ])

                            if result and result.get("posts"):
                                daily_data.append_posts(today_date, keyword, result["posts"])
                                print(f"✅ {keyword}数据已累积到当日文件中")
                        else:
                            print("❌ 每日数据管理器未初始化")

                    except Exception as e:
                        print(f"❌ 数据累积失败 {keyword}: {e}")

            # 最终推送阶段 (20:00，读取累积数据并推送）
            elif current_hour == 20:
                print(f"🚀 推送阶段: 为用户{phone}执行每日分析推送")

                for keyword in keywords:
                    try:
                        # 获取当日累积数据
                        daily_data = manager.workflow_context.get("daily_data")
                        if daily_data:
                            accumulated_posts = daily_data.get_daily_posts(today_date, keyword)

                            if accumulated_posts:
                                print(f"📊 {keyword}累计数据: {len(accumulated_posts)}条")

                                # 执行完整的分析流程
                                result = manager.run_workflow({
                                    "keyword": keyword,
                                    "posts": accumulated_posts,  # 使用累积数据
                                    "analyze_mode": "daily_summary"  # 标志这是每日总结推送
                                }, [
                                    {"node_name": "receive_input", "input_map": {"keyword": "input.keyword"}},
                                    {"node_name": "llm_summarize", "input_map": {"posts": "input.posts", "keyword": "input.keyword"}},  # 直接使用累积数据
                                    {"node_name": "feishu_notify", "input_map": {"summary": "llm_summarize.summary"}}
                                ])

                                print(f"✅ 每日分析推送完成: {keyword}")
                            else:
                                print(f"⚠️ {keyword}今日暂无累积数据，跳过推送")
                        else:
                            print("❌ 每日数据管理器未初始化")

                    except Exception as e:
                        print(f"❌ 每日推送失败 {keyword}: {e}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"🚨 定时任务异常: {e}")

# 定时推送任务
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 获取所有用户
        cursor.execute("SELECT id, phone, cookie, feishu_webhook, keywords FROM users")
        users = cursor.fetchall()

        for user in users:
            user_id, phone, cookie, webhook, keywords_json = user
            keywords = json.loads(keywords_json) if keywords_json else []

            for keyword in keywords:
                # 调用w2的工作流
                manager = WorkflowManager(
                    api_key="sk-addb15e06fef4c19a46122a39aac8caa",  # 从环境变量获取
                    prompt_folder="./prompt",
                    memory_path=f"./workflow_memory/{phone}_memory.json"
                )

                # 注册节点，修改为用户特定的cookie
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
                        "attrs": {"cookie": cookie}
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
                        "attrs": {"webhook": webhook}
                    }
                ]

                for cfg in nodes_config:
                    config = NodeConfig(**cfg)
                    manager.register_node(config)

                flow = [
                    {"node_name": "receive_01", "input_map": {"keyword": "input.keyword"}},
                    {"node_name": "weibo_crawl_01", "input_map": {"keyword": "receive_01.keyword"}},
                    {"node_name": "llm_summarize", "input_map": {"posts": "weibo_crawl_01.posts", "keyword": "receive_01.keyword"}},
                    {"node_name": "feishu_notify", "input_map": {"summary": "llm_summarize.summary"}}
                ]

                # 执行工作流
                result = manager.run_workflow({"keyword": keyword}, flow)
                if result and "summary" in result:
                    # 记录推送日志
                    cursor.execute("INSERT INTO push_logs (user_id, keyword, summary) VALUES (?, ?, ?)",
                                   (user_id, keyword, result["summary"]))
                    conn.commit()

        cursor.close()
        conn.close()
        print("✅ 每日推送任务完成")

    except Exception as e:
        print(f"🚨 推送任务失败: {e}")

# 修改调度器，支持4小时数据累积模式
def start_scheduler(test_mode=False):
    global scheduler
    scheduler = BackgroundScheduler()
    if test_mode:
        # 测试模式: 10秒后执行
        scheduler.add_job(daily_push, 'interval', seconds=10, id='daily_push')
    else:
        # 正常模式: 每4小时执行一次数据累积，20:00执行每日推送
        # 8:00, 12:00, 16:00, 20:00 触发
        scheduler.add_job(daily_push, 'cron', hour='8,12,16,20', id='daily_push')
    scheduler.start()

atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    # 解析命令行参数
    port = 5000
    test_mode = False

    if '--port' in sys.argv:
        port_idx = sys.argv.index('--port')
        if port_idx + 1 < len(sys.argv):
            try:
                port = int(sys.argv[port_idx + 1])
            except ValueError:
                print("端口号必须是数字")
                sys.exit(1)

    if '--test' in sys.argv:
        test_mode = True

    init_db()
    start_scheduler(test_mode=test_mode)
    app.run(debug=True, port=port, host='0.0.0.0')
