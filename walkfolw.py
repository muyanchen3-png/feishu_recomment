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


class PromptManager:
    """提示词管理器"""
    
    def __init__(self, prompt_folder: str = './prompt/'):
        self.prompt_folder = prompt_folder
        if not os.path.exists(prompt_folder):
            raise Exception(f"提示词文件夹不存在: {prompt_folder}")
        
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
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"大模型调用失败: {str(e)}")

    def send_embedding(self, text_list):
        """发送embedding请求 - 当前不需要embedding功能"""
        # 由于当前不需要embedding功能，返回空结果
        return []

    def get_lang_embedding(self):
        """获取langchain embedding工具 - 当前不需要embedding功能"""
        # 由于当前不需要embedding功能，返回None
        return None


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
            # 解析路径，例如 "start.pkg" 或 "llm_01.content"
            if "." in value_path:
                parts = value_path.split(".")
                if parts[0] == "start":
                    # 从工作流开始时的输入参数获取
                    mapped_input[key] = input_data.get(parts[1])
                else:
                    # 从之前节点的输出获取
                    node_name = parts[0]
                    output_key = parts[1] if len(parts) > 1 else key
                    if node_name in self.workflow_context.node_outputs:
                        node_output = self.workflow_context.node_outputs[node_name]
                        if isinstance(node_output, dict):
                            mapped_input[key] = node_output.get(output_key)
                        else:
                            # 如果输出不是字典，直接使用整个输出
                            mapped_input[key] = node_output
                    else:
                        # 如果节点输出不存在，尝试从输入数据中获取
                        mapped_input[key] = input_data.get(output_key)
            else:
                # 直接从输入数据获取
                mapped_input[key] = input_data.get(value_path, value_path)
        
        return mapped_input


class ReceiveNode(WorkflowNode):
    """接收节点 - 工作流入口"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 映射输入
            mapped_input = self._map_input(input_data)
            
            # 确定下一个节点
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
            # 映射输入
            mapped_input = self._map_input(input_data)
            
            # 获取LLM和prompt
            llm = self.workflow_context.llm
            prompt_file = self.attrs.get("prompt", "user.md")
            model_name = self.attrs.get("model", "deepseek-r1")
            
            # 更新LLM模型（如果配置了）
            if hasattr(llm, 'model'):
                llm.model = model_name
            
            # 渲染提示词模板
            prompt_text = self.workflow_context.prompt_manager.render_template(prompt_file, mapped_input)
            
            # 执行LLM调用
            response = llm.send_message(prompt_text)
            
            # 确定下一个节点
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


class GetOutputNode(WorkflowNode):
    """输出节点"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 映射输入
            mapped_input = self._map_input(input_data)
            
            # 确定下一个节点
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


class SaveResultNode(WorkflowNode):
    """保存结果节点"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 映射输入
            mapped_input = self._map_input(input_data)
            
            # 获取要保存的内容
            content = mapped_input.get("content", "")
            query = mapped_input.get("query", "unknown")
            
            # 确保results目录存在
            results_dir = "./results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # 构建保存的数据
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "content": content,
                "workflow_node": self.name
            }
            
            # 保存到JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            # 确定下一个节点
            next_node = self.choice_map.get("default", "finish")
            
            return {
                "status": "success",
                "output": {
                    "file_path": filepath,
                    "saved_data": result_data
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
    
    def __init__(self, workflow_config: Dict[str, Any], llm, model, prompt_manager: PromptManager):
        self.config = workflow_config
        self.start_node = workflow_config["start_node"]
        self.input_parameters = workflow_config["input_parameters"]
        self.nodes = workflow_config["nodes"]
        self.llm = llm
        self.model = model
        self.prompt_manager = prompt_manager
        
        # 存储节点输出
        self.node_outputs = {}
        
        # 构建节点映射
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
            elif node_type == NodeType.GET_OUTPUT:
                node_instance = GetOutputNode(config, self)
            elif node_type == NodeType.SAVE_RESULT:
                node_instance = SaveResultNode(config, self)
            else:
                raise ValueError(f"不支持的节点类型: {config.node_type}")
            
            node_map[config.node_name] = node_instance
        
        return node_map


class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self, workflow_config: Dict[str, Any], llm, model, prompt_manager: PromptManager):
        self.context = WorkflowContext(workflow_config, llm, model, prompt_manager)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流"""
        # 合并初始输入参数
        execution_input = {**self.context.input_parameters, **input_data}
        
        current_node_name = self.context.start_node
        step_count = 0
        max_steps = len(self.context.nodes) * 2  # 防止无限循环
        
        while current_node_name != "finish" and step_count < max_steps:
            if current_node_name not in self.context.node_map:
                raise ValueError(f"找不到节点: {current_node_name}")
            
            node = self.context.node_map[current_node_name]
            
            # 执行节点
            result = node.execute(execution_input)
            
            # 存储节点输出
            self.context.node_outputs[current_node_name] = result["output"]
            
            # 检查执行状态
            if result["status"] == "error":
                raise Exception(f"节点 {current_node_name} 执行失败: {result['output']['error']}")
            
            # 更新当前节点
            current_node_name = result["next_node"]
            step_count += 1
        
        # 返回最终结果
        return self.context.node_outputs


class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, llm, model, prompt_folder: str = './prompt/'):
        self.llm = llm
        self.model = model
        self.prompt_manager = PromptManager(prompt_folder)
        self.workflows = {}
    
    def register_workflow(self, name: str, config: Dict[str, Any]):
        """注册工作流"""
        self.workflows[name] = config
    
    def execute_workflow(self, name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定名称的工作流"""
        if name not in self.workflows:
            raise ValueError(f"工作流 {name} 未注册")
        
        workflow_config = self.workflows[name]
        executor = WorkflowExecutor(workflow_config, self.llm, self.model, self.prompt_manager)
        return executor.execute(input_data)
    
    def get_workflow_names(self) -> List[str]:
        """获取所有注册的工作流名称"""
        return list(self.workflows.keys())
    
    def create_prompt_file(self, filename: str, content: str):
        """创建提示词文件"""
        filepath = os.path.join(self.prompt_manager.prompt_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"提示词文件已创建: {filepath}")


def create_workflow_manager_with_deepseek_casevo(api_key: str, model: str = "deepseek-r1", 
                                                prompt_folder: str = './prompt/',
                                                graph=None, memory_path=None):
    """创建使用DeepSeek和CaseVO框架的WorkflowManager"""
    llm = DeepSeekLLM(api_key, model)
    
    # 导入CaseVO模块
    try:
        from casevo.model_base import ModelBase
        from casevo.llm_interface import LLM_INTERFACE
        
        # 创建CaseVO模型
        if graph is None:
            import networkx as nx
            graph = nx.Graph()
            graph.add_node(0)  # 添加一个默认节点
        
        # 由于不需要embedding功能，我们创建一个简化的模型实例
        # 或者使用一个模拟的模型
        model = ModelBase(graph, llm, prompt_path=prompt_folder, memory_path=memory_path)
        
        return WorkflowManager(llm, model, prompt_folder)
    
    except ImportError:
        print("警告: CaseVO模块未找到，使用模拟模型")
        class MockModel:
            def __init__(self):
                from casevo.prompt import PromptFactory
                
                class MockPromptFactory:
                    def get_template(self, template_name):
                        class MockTemplate:
                            def send_prompt(self, input_data, agent=None, model=None):
                                return f"Processed: {input_data}"
                        return MockTemplate()
                
                self.prompt_factory = MockPromptFactory()
        
        model = MockModel()
        return WorkflowManager(llm, model, prompt_folder)


# 使用示例 - 通用问答工作流（带结果保存）
if __name__ == "__main__":
    # 通用问答工作流配置（带结果保存）
    general_qa_workflow = {
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
                "attrs": {
                    "model": "deepseek-r1",
                    "prompt": "general_sys.md"
                }
            },
            {
                "id": 1,
                "node_type": "llm_step",
                "node_name": "llm_01",
                "input_map": {
                    "question": "receive_01.question",
                    "context": "start.context"
                },
                "choice_map": {
                    "default": "save_result_01"
                },
                "attrs": {
                    "model": "deepseek-r1",
                    "prompt": "general_user.md"
                }
            },
            {
                "id": 2,
                "node_type": "save_result",
                "node_name": "save_result_01",
                "input_map": {
                    "content": "llm_01.content",
                    "question": "llm_01.input_params.question"
                },
                "choice_map": {
                    "default": "output_01"
                },
                "attrs": {}
            },
            {
                "id": 3,
                "node_type": "get_output",
                "node_name": "output_01",
                "input_map": {
                    "file_path": "save_result_01.file_path"
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
        f.write('''系统提示词:
你是一个专业的AI助手。请根据用户的问题，提供准确、有用的信息。
你可以回答各种类型的问题，包括但不限于：
- 天气查询
- 新闻咨询
- 技术问题
- 生活建议
- 学术问题
- 娱乐话题
请根据用户的具体问题进行回答。
''')
    
    with open('./prompt/general_user.md', 'w', encoding='utf-8') as f:
        f.write('''用户问题:
{{ question }}
上下文: {{ context | default("无") }}
请根据以上信息回答用户的问题。
''')
    
    print("通用问答工作流提示词文件已创建")
    
    # 使用示例（请替换为实际的API密钥）
    manager = create_workflow_manager_with_deepseek_casevo("sk-addb15e06fef4c19a46122a39aac8caa", "deepseek-r1")
    
    print("\n使用说明:")
    print("1. 使用DeepSeek集成CaseVO: manager = create_workflow_manager_with_deepseek_casevo('your_api_key')")
    print("2. 注册通用问答工作流: manager.register_workflow('general_qa', general_qa_workflow)")
    print("3. 执行问答: result = manager.execute_workflow('general_qa', {'question': '今天天气怎么样', 'context': '北京'})")
    print("4. 结果将自动保存到 ./results/ 目录下的JSON文件中")
    
    # 示例：执行通用问答
    manager.register_workflow("general_qa", general_qa_workflow)
    
    # 测试多个不同类型的问题
    questions = [
        {"question": "今天天气怎么样", "context": "北京"},
        {"question": "Python编程语言有什么特点", "context": "技术问题"},
        {"question": "如何做一道简单的意大利面", "context": "生活问题"}
    ]
    
    for i, q in enumerate(questions):
        result = manager.execute_workflow("general_qa", q)
        print(f"问题 {i+1} 的结果已保存到: {result}")



