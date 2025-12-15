#!/bin/bash

# 设置MySQL数据库
echo "设置MySQL数据库..."
mysql -u root -p -e "
CREATE DATABASE IF NOT EXISTS weibo_push_service;
CREATE USER IF NOT EXISTS 'weibo_user'@'localhost' IDENTIFIED BY 'weibo_pass';
GRANT ALL PRIVILEGES ON weibo_push_service.* TO 'weibo_user'@'localhost';
FLUSH PRIVILEGES;
"

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 检查并安装ChromeDriver（用于Selenium）
echo "检查ChromeDriver..."
if ! command -v chromedriver &> /dev/null; then
    echo "安装ChromeDriver..."
    brew install chromium --no-quarantine
    brew install --cask chromedriver
    # 或者手动下载到PATH
    # wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_mac64.zip
    # unzip chromedriver_mac64.zip
    # sudo mv chromedriver /usr/local/bin/
    # chmod +x /usr/local/bin/chromedriver
fi

# 创建必要目录
mkdir -p workflow_memory
mkdir -p prompt
mkdir -p static

echo "设置完成!"
echo "请启动服务: python service.py"
echo "然后在浏览器打开: http://localhost:5000"
