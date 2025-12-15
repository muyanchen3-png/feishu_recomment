#!/bin/bash

# 重启后台服务
echo "启动微博推送服务..."

# 杀死之前的进程
pkill -f "python service.py" || true

# 后台运行服务
nohup python service.py > service.log 2>&1 &
echo "服务已启动，请查看 service.log 文件监控日志"

# 等待几秒让服务启动
sleep 3

# 检查服务是否运行
if curl -s http://localhost:5000 > /dev/null; then
    echo "✅ 服务启动成功！"
    echo "访问地址: http://localhost:5000"
else
    echo "❌ 服务启动失败，请检查 service.log"
fi
