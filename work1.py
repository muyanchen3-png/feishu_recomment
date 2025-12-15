import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader
import requests
from datetime import datetime


class NodeType(Enum):
    """节点类型枚举"""
    RECEIVE = "receive"
    LLM_STEP = "llm_step"
    GET_OUTPUT = "get_output"
    SAVE_RESULT = "save_result"
    FEISHU_BOT = "feishu_bot"  # 新的飞书机器人节点类型
    CUSTOM = "custom"


@dataclass
class NodeConfig:
    """节点配置数据类"""
    id: int
    node_type: str
    node_name: str
    input_map: Dict[str, str]
    choice_map: Dict[str, str]
    attrs: Dict[str, Any]


class FeishuBotClient:
    """飞书机器人客户端"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_message(self, title: str, content: str) -> Dict[str, Any]:
        """发送消息到飞书机器人"""
        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": title
                    },
                    "template": "blue"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": content
                        }
                    },
                    {
                        "tag": "hr"
                    },
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            }
                        ]
                    }
                ]
            }
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") == 0:
                return {"status": "success", "message_id": result.get("data", {}).get("message_id")}
            else:
                raise Exception(f"飞书机器人消息发送失败: {result.get('msg')}")
        except Exception as e:
            raise Exception(f"飞书机器人消息发送失败: {str(e)}")


class PromptManager:
    """提示词管理器"""
    
    def __init__(self, prompt_folder: str = './prompt/'):
        self.prompt_folder = prompt_folder
        if not os.path.exists(prompt_folder):
            os.makedirs(prompt_folder)
        
        self.env = Environment(loader=FileSystemLoader(prompt_folder))
    
    def get_template(self, template_name: str):
        """获取模板"""
        template_path = os.path.join(self.prompt_folder, template_name)
        if not os.path.exists(template_path):
            raise Exception(f"提示词文件不存在: {template_path}")
        
        return self.env.get_template(template_name)
    
    def render_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """渲染模板"""
        template = self.get_template(template_name)
        return template.render(**data)


class DeepSeekLLM:
    """DeepSeek R1 大模型接口"""
    
    def __init__(self, api_key: str, model: str = "deepseek-r1"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    def send_message(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """发送消息到大模型"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            print("正在调用DeepSeek API...")
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print("DeepSeek API调用成功")
            return content
        except Exception as e:
            error_msg = f"大模型调用失败: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)


class WorkflowNode(ABC):
    """工作流节点基类"""
    
    def __init__(self, config: NodeConfig, workflow_context):
        self.config = config
        self.workflow_context = workflow_context
        self.name = config.node_name
        self.input_map = config.input_map
        self.choice_map = config.choice_map
        self.attrs = config.attrs

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行节点逻辑
        返回: {"status": "success/error", "output": {...}, "next_node": "..."}
        """
        pass

    def _map_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """根据input_map映射输入数据"""
        mapped_input = {}
        for key, value_path in self.input_map.items():
            if "." in value_path:
                parts = value_path.split(".")
                if parts[0] == "start":
                    mapped_input[key] = input_data.get(parts[1])
                else:
                    node_name = parts[0]
                    output_key = parts[1] if len(parts) > 1 else key
                    if node_name in self.workflow_context.node_outputs:
                        node_output = self.workflow_context.node_outputs[node_name]
                        if isinstance(node_output, dict):
                            mapped_input[key] = node_output.get(output_key)
                        else:
                            mapped_input[key] = node_output
                    else:
                        mapped_input[key] = input_data.get(output_key)
            else:
                mapped_input[key] = input_data.get(value_path, value_path)
        
        return mapped_input


class ReceiveNode(WorkflowNode):
    """接收节点 - 工作流入口"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            mapped_input = self._map_input(input_data)
            next_node = self.choice_map.get("default", "finish")
            
            return {
                "status": "success",
                "output": mapped_input,
                "next_node": next_node
            }
        except Exception as e:
            return {
                "status": "error",
                "output": {"error": str(e)},
                "next_node": "finish"
            }


class LLMStepNode(WorkflowNode):
    """LLM步骤节点"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            mapped_input = self._map_input(input_data)
            llm = self.workflow_context.llm
            prompt_file = self.attrs.get("prompt", "user.md")
            model_name = self.attrs.get("model", "deepseek-r1")
            
            if hasattr(llm, 'model'):
                llm.model = model_name
            
            prompt_text = self.workflow_context.prompt_manager.render_template(prompt_file, mapped_input)
            response = llm.send_message(prompt_text)
            next_node = self.choice_map.get("default", "finish")
            
            return {
                "status": "success",
                "output": {
                    "content": response,
                    "input_params": mapped_input,
                    "prompt_used": prompt_text
                },
                "next_node": next_node
            }
        except Exception as e:
            return {
                "status": "error",
                "output": {"error": str(e)},
                "next_node": "finish"
            }


class FeishuBotNode(WorkflowNode):
    """飞书机器人节点 - 将结果发送到飞书机器人"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            mapped_input = self._map_input(input_data)
            content = mapped_input.get("content", "")
            title = mapped_input.get("title", "AI生成内容")
            query = mapped_input.get("query", "未知查询")
            
            # 获取飞书机器人客户端
            feishu_bot = self.workflow_context.feishu_bot
            if not feishu_bot:
                raise Exception("飞书机器人客户端未初始化")
            
            # 构建消息内容
            message_content = f"**查询**: {query}\n\n**生成内容**:\n{content}"
            
            # 发送消息
            result = feishu_bot.send_message(title, message_content)
            
            next_node = self.choice_map.get("default", "finish")
            
            return {
                "status": "success",
                "output": {
                    "feishu_bot_result": result,
                    "title": title,
                    "query": query,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content
                },
                "next_node": next_node
            }
        except Exception as e:
            return {
                "status": "error",
                "output": {"error": str(e)},
                "next_node": "finish"
            }


class WorkflowContext:
    """工作流执行上下文"""
    
    def __init__(self, workflow_config: Dict[str, Any], llm, model, prompt_manager: PromptManager, feishu_bot: FeishuBotClient = None):
        self.config = workflow_config
        self.start_node = workflow_config["start_node"]
        self.input_parameters = workflow_config["input_parameters"]
        self.nodes = workflow_config["nodes"]
        self.llm = llm
        self.model = model
        self.prompt_manager = prompt_manager
        self.feishu_bot = feishu_bot
        
        self.node_outputs = {}
        self.node_map = self._build_node_map()
    
    def _build_node_map(self) -> Dict[str, WorkflowNode]:
        """构建节点映射"""
        node_map = {}
        for node_data in self.nodes:
            config = NodeConfig(**node_data)
            node_type = NodeType(config.node_type)
            
            if node_type == NodeType.RECEIVE:
                node_instance = ReceiveNode(config, self)
            elif node_type == NodeType.LLM_STEP:
                node_instance = LLMStepNode(config, self)
            elif node_type == NodeType.FEISHU_BOT:
                node_instance = FeishuBotNode(config, self)
            else:
                # 简化处理其他节点类型
                class GenericNode(WorkflowNode):
                    def execute(self, input_data):
                        return {
                            "status": "success", 
                            "output": input_data, 
                            "next_node": self.choice_map.get("default", "finish")
                        }
                node_instance = GenericNode(config, self)
            
            node_map[config.node_name] = node_instance
        
        return node_map


class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self, workflow_config: Dict[str, Any], llm, model, prompt_manager: PromptManager, feishu_bot: FeishuBotClient = None):
        self.context = WorkflowContext(workflow_config, llm, model, prompt_manager, feishu_bot)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流"""
        execution_input = {**self.context.input_parameters, **input_data}
        
        current_node_name = self.context.start_node
        step_count = 0
        max_steps = len(self.context.nodes) * 2
        
        while current_node_name != "finish" and step_count < max_steps:
            if current_node_name not in self.context.node_map:
                raise ValueError(f"找不到节点: {current_node_name}")
            
            node = self.context.node_map[current_node_name]
            result = node.execute(execution_input)
            self.context.node_outputs[current_node_name] = result["output"]
            
            if result["status"] == "error":
                raise Exception(f"节点 {current_node_name} 执行失败: {result['output']['error']}")
            
            current_node_name = result["next_node"]
            step_count += 1
        
        return self.context.node_outputs


class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, llm, model, prompt_folder: str = './prompt/', feishu_bot: FeishuBotClient = None):
        self.llm = llm
        self.model = model
        self.prompt_manager = PromptManager(prompt_folder)
        self.feishu_bot = feishu_bot
        self.workflows = {}
    
    def register_workflow(self, name: str, config: Dict[str, Any]):
        """注册工作流"""
        self.workflows[name] = config
    
    def execute_workflow(self, name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定名称的工作流"""
        if name not in self.workflows:
            raise ValueError(f"工作流 {name} 未注册")
        
        workflow_config = self.workflows[name]
        executor = WorkflowExecutor(workflow_config, self.llm, self.model, self.prompt_manager, self.feishu_bot)
        return executor.execute(input_data)


def create_workflow_manager_with_feishu_bot(api_key: str, feishu_webhook_url: str, 
                                          model: str = "deepseek-r1", prompt_folder: str = './prompt/'):
    """创建使用DeepSeek和飞书机器人的WorkflowManager"""
    llm = DeepSeekLLM(api_key, model)
    feishu_bot = FeishuBotClient(feishu_webhook_url)
    
    return WorkflowManager(llm, model, prompt_folder, feishu_bot)


# 使用示例 - 支持飞书机器人的工作流
if __name__ == "__main__":
    # 飞书机器人工作流配置
    feishu_bot_workflow = {
        "start_node": "receive_01",
        "input_parameters": {
            "question": "通用问题"
        },
        "nodes": [
            {
                "id": 0,
                "node_type": "receive",
                "node_name": "receive_01",
                "input_map": {
                    "question": "start.question"
                },
                "choice_map": {
                    "default": "llm_01"
                },
                "attrs": {}
            },
            {
                "id": 1,
                "node_type": "llm_step",
                "node_name": "llm_01",
                "input_map": {
                    "question": "receive_01.question"
                },
                "choice_map": {
                    "default": "feishu_bot_01"
                },
                "attrs": {
                    "prompt": "general_user.md"
                }
            },
            {
                "id": 2,
                "node_type": "feishu_bot",
                "node_name": "feishu_bot_01",
                "input_map": {
                    "content": "llm_01.content",
                    "title": "receive_01.question",
                    "query": "receive_01.question"
                },
                "choice_map": {
                    "default": "finish"
                },
                "attrs": {}
            }
        ]
    }
    
    # 确保提示词目录存在
    os.makedirs('./prompt/', exist_ok=True)
    
    # 创建通用问答提示词文件
    with open('./prompt/general_sys.md', 'w', encoding='utf-8') as f:
        f.write('''你是一个专业的AI助手。请根据用户的问题，提供准确、有用的信息。''')
    
    with open('./prompt/general_user.md', 'w', encoding='utf-8') as f:
        f.write('''用户问题: {{ question }}
请回答这个问题。''')
    
    print("通用问答工作流提示词文件已创建")
    
    # 使用示例（请替换为实际的API密钥和飞书Webhook）
    feishu_webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/77bf45ee-a658-4476-ac7e-cf3e9f538fae"
    
    try:
        manager = create_workflow_manager_with_feishu_bot(
            api_key="sk-addb15e06fef4c19a46122a39aac8caa",
            feishu_webhook_url=feishu_webhook_url
        )
        
        # 示例：执行飞书机器人工作流
        manager.register_workflow("feishu_bot_qa", feishu_bot_workflow)
        
        # 测试问题
        test_questions = [
            {"question": "人工智能的发展现状和未来趋势"},
        ]
        
        for i, q in enumerate(test_questions):
            try:
                print(f"\n执行问题 {i+1}: {q['question']}")
                result = manager.execute_workflow("feishu_bot_qa", q)
                bot_result = result.get('feishu_bot_01', {}).get('feishu_bot_result', {})
                if bot_result.get('status') == 'success':
                    print(f"✅ 问题 {i+1} 的结果已发送到飞书机器人")
                else:
                    print(f"❌ 问题 {i+1} 发送失败: {bot_result}")
                    
            except Exception as e:
                print(f"❌ 执行失败: {e}")
                
    except Exception as e:
        print(f"❌ 初始化失败: {e}")