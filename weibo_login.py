# weibo_login.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pickle
import os

class WeiboCookieFetcher:
    def __init__(self, username: str, password: str, cookie_file: str = "weibo_cookie.pkl"):
        self.username = username
        self.password = password
        self.cookie_file = cookie_file
        self.driver = None

    def create_driver(self):
        """创建防检测的 Chrome 浏览器实例"""
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
        # 隐藏 webdriver 特征
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => false});"
        })
        return driver

    def wait_for_login(self):
        """等待用户完成登录（可手动处理滑块/扫码）"""
        print("⏳ 请在浏览器中完成登录（可手动处理滑块或扫码）...")
        while True:
            try:
                current_url = self.driver.current_url
                # 成功登录后的典型 URL 特征
                if "myprofile" in current_url or "weibo.com/u" in current_url or "home" in current_url:
                    print("✅ 检测到已登录！")
                    break
                else:
                    print("🌐 当前页面:", current_url)
                    time.sleep(3)
            except:
                time.sleep(1)
        time.sleep(2)

    def login_and_save_cookie(self):
        """启动浏览器，登录并保存 Cookie"""
        self.driver = self.create_driver()
        try:
            print("🌐 打开微博登录页...")
            self.driver.get("https://weibo.com/login.php")
            time.sleep(5)

            # 输入账号密码
            print("📝 正在输入账号密码...")
            self.driver.find_element(By.NAME, "username").send_keys(self.username)
            self.driver.find_element(By.NAME, "password").send_keys(self.password)

            # 点击登录按钮（注意：可能有多个登录按钮，选第一个可见的）
            login_btns = self.driver.find_elements(By.XPATH, '//div[@class="info_list login_btn"]/a')
            if login_btns:
                login_btns[0].click()
            else:
                raise Exception("未找到登录按钮")

            time.sleep(3)

            # 等待用户完成验证（滑块、短信、扫码等）
            self.wait_for_login()

            # 获取 Cookie
            cookies = self.driver.get_cookies()
            with open(self.cookie_file, "wb") as f:
                pickle.dump(cookies, f)
            print(f"💾 Cookie 已保存至 {self.cookie_file}")

            # 返回 Cookie 字符串格式
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
        """从本地加载 Cookie（若未过期）"""
        if os.path.exists(self.cookie_file):
            file_time = os.path.getmtime(self.cookie_file)
            # 超过 12 小时认为过期
            if (time.time() - file_time) > 12 * 3600:
                print("🕒 Cookie 已过期，将重新登录")
                return None

            with open(self.cookie_file, "rb") as f:
                cookies = pickle.load(f)
            print("🍪 已加载本地 Cookie")
            return "; ".join([f"{c['name']}={c['value']}" for c in cookies])
        return None

    def get_cookie(self) -> str:
        """主入口：优先加载本地 Cookie，否则自动登录获取"""
        cookie = self.load_cookie()
        if not cookie:
            print("🔄 开始自动登录获取新 Cookie...")
            cookie = self.login_and_save_cookie()
        return cookie