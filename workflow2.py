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
    FEISHU_SAVE = "feishu_save"
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


class FeishuMCPClient:
    """飞书MCP服务客户端"""
    
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = "https://open.feishu.cn/open-apis"
        self.tenant_access_token = None
        self._get_tenant_access_token()
    
    def _get_tenant_access_token(self):
        """获取租户访问令牌"""
        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        headers = {
            "Content-Type": "application/json; charset=utf-8"
        }
        data = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            if result.get("code") == 0:
                self.tenant_access_token = result["tenant_access_token"]
            else:
                raise Exception(f"获取tenant_access_token失败: {result.get('msg')}")
        except Exception as e:
            raise Exception(f"飞书认证失败: {str(e)}")
    
    def create_doc(self, folder_token: str, title: str, content: str) -> Dict[str, Any]:
        """在飞书云文档中创建文档"""
        if not self.tenant_access_token:
            self._get_tenant_access_token()
            
        url = f"https://open.feishu.cn/open-apis/drive/v1/files/{folder_token}"
        headers = {
            "Authorization": f"Bearer {self.tenant_access_token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        data = {
            "folder_token": folder_token,
            "title": title,
            "type": "doc"
        }
        
        try:
            # 创建文档
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") == 0:
                doc_token = result["data"]["file"]["token"]
                # 更新文档内容
                update_result = self.update_doc_content(doc_token, content)
                return {
                    "doc_token": doc_token,
                    "url": result["data"]["file"]["url"],
                    "title": title,
                    "update_status": update_result
                }
            else:
                raise Exception(f"创建文档失败: {result.get('msg')}")
        except Exception as e:
            raise Exception(f"飞书文档创建失败: {str(e)}")
    
    def update_doc_content(self, doc_token: str, content: str) -> Dict[str, Any]:
        """更新文档内容"""
        if not self.tenant_access_token:
            self._get_tenant_access_token()
            
        url = f"{self.base_url}/docx/v1/documents/{doc_token}/blocks/{doc_token}/children"
        headers = {
            "Authorization": f"Bearer {self.tenant_access_token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        # 构建文档内容结构
        doc_content = {
            "children": [
                {
                    "block_type": "paragraph",
                    "paragraph": {
                        "elements": [
                            {
                                "text_run": {
                                    "content": content
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        try:
            response = requests.patch(url, headers=headers, json=doc_content)
            response.raise_for_status()
            result = response.json()
            return {"status": "success", "result": result}
        except Exception as e:
            raise Exception(f"更新文档内容失败: {str(e)}")
    
    def upload_to_bitable(self, app_token: str, table_id: str, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """上传数据到飞书多维表格"""
        if not self.tenant_access_token:
            self._get_tenant_access_token()
            
        url = f"{self.base_url}/bitable/v1/apps/{app_token}/tables/{table_id}/records"
        headers = {
            "Authorization": f"Bearer {self.tenant_access_token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        data = {
            "fields": record_data
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            raise Exception(f"上传到多维表格失败: {str(e)}")


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
        return []

    def get_lang_embedding(self):
        """获取langchain embedding工具 - 当前不需要embedding功能"""
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


class GetOutputNode(WorkflowNode):
    """输出节点"""
    
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


class SaveResultNode(WorkflowNode):
    """保存结果节点"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            mapped_input = self._map_input(input_data)
            content = mapped_input.get("content", "")
            query = mapped_input.get("query", "unknown")
            
            results_dir = "./results"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "content": content,
                "workflow_node": self.name
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
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


class FeishuSaveNode(WorkflowNode):
    """飞书保存节点 - 将结果保存到飞书文档"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            mapped_input = self._map_input(input_data)
            content = mapped_input.get("content", "")
            title = mapped_input.get("title", f"AI生成内容_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            query = mapped_input.get("query", "未知查询")
            
            # 获取飞书客户端
            feishu_client = self.workflow_context.feishu_client
            if not feishu_client:
                raise Exception("飞书客户端未初始化")
            
            # 获取飞书配置
            folder_token = self.attrs.get("folder_token", "")
            if not folder_token:
                raise Exception("未配置飞书文件夹token")
            
            # 构建文档内容
            doc_content = f"""# {title}

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**原始查询**: {query}

## 生成内容

{content}

---
*本内容由AI工作流自动生成*
"""
            
            # 创建飞书文档
            result = feishu_client.create_doc(folder_token, title, doc_content)
            
            next_node = self.choice_map.get("default", "finish")
            
            return {
                "status": "success",
                "output": {
                    "feishu_doc": result,
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
    
    def __init__(self, workflow_config: Dict[str, Any], llm, model, prompt_manager: PromptManager, feishu_client: FeishuMCPClient = None):
        self.config = workflow_config
        self.start_node = workflow_config["start_node"]
        self.input_parameters = workflow_config["input_parameters"]
        self.nodes = workflow_config["nodes"]
        self.llm = llm
        self.model = model
        self.prompt_manager = prompt_manager
        self.feishu_client = feishu_client
        
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
            elif node_type == NodeType.GET_OUTPUT:
                node_instance = GetOutputNode(config, self)
            elif node_type == NodeType.SAVE_RESULT:
                node_instance = SaveResultNode(config, self)
            elif node_type == NodeType.FEISHU_SAVE:
                node_instance = FeishuSaveNode(config, self)
            else:
                raise ValueError(f"不支持的节点类型: {config.node_type}")
            
            node_map[config.node_name] = node_instance
        
        return node_map


class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self, workflow_config: Dict[str, Any], llm, model, prompt_manager: PromptManager, feishu_client: FeishuMCPClient = None):
        self.context = WorkflowContext(workflow_config, llm, model, prompt_manager, feishu_client)
    
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
    
    def __init__(self, llm, model, prompt_folder: str = './prompt/', feishu_client: FeishuMCPClient = None):
        self.llm = llm
        self.model = model
        self.prompt_manager = PromptManager(prompt_folder)
        self.feishu_client = feishu_client
        self.workflows = {}
    
    def register_workflow(self, name: str, config: Dict[str, Any]):
        """注册工作流"""
        self.workflows[name] = config
    
    def execute_workflow(self, name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定名称的工作流"""
        if name not in self.workflows:
            raise ValueError(f"工作流 {name} 未注册")
        
        workflow_config = self.workflows[name]
        executor = WorkflowExecutor(workflow_config, self.llm, self.model, self.prompt_manager, self.feishu_client)
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


def create_workflow_manager_with_feishu(api_key: str, feishu_app_id: str, feishu_app_secret: str, 
                                      model: str = "deepseek-r1", prompt_folder: str = './prompt/'):
    """创建使用DeepSeek和飞书MCP的WorkflowManager"""
    llm = DeepSeekLLM(api_key, model)
    feishu_client = FeishuMCPClient(feishu_app_id, feishu_app_secret)
    
    return WorkflowManager(llm, model, prompt_folder, feishu_client)


# 使用示例 - 支持飞书保存的工作流
if __name__ == "__main__":
    # 支持飞书保存的工作流配置
    feishu_workflow = {
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
                    "default": "feishu_save_01"
                },
                "attrs": {
                    "model": "deepseek-r1",
                    "prompt": "general_user.md"
                }
            },
            {
                "id": 2,
                "node_type": "feishu_save",
                "node_name": "feishu_save_01",
                "input_map": {
                    "content": "llm_01.content",
                    "title": "llm_01.input_params.question",
                    "query": "llm_01.input_params.question"
                },
                "choice_map": {
                    "default": "output_01"
                },
                "attrs": {
                    "folder_token": "你的飞书文件夹token"
                }
            },
            {
                "id": 3,
                "node_type": "get_output",
                "node_name": "output_01",
                "input_map": {
                    "feishu_doc": "feishu_save_01.feishu_doc"
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
   

    # 使用示例（请替换为实际的API密钥和飞书配置）
    feishu_app_id = "cli_a8771266f2fb5013"
    feishu_app_secret = "bVh7WT9QoYIFQrowSR2UCepcJTgrTaMs"
    feishu_folder_token = 'u-fhyvEL5I55IF3pcjdOeNUQl1i6GBggqVVy2aVBI005Ed'  # 可以从飞书云文档URL中获取
    
    # 更新工作流配置中的飞书文件夹token
    feishu_workflow["nodes"][2]["attrs"]["folder_token"] = feishu_folder_token
    
    manager = create_workflow_manager_with_feishu(
        api_key="sk-addb15e06fef4c19a46122a39aac8caa",
        feishu_app_id=feishu_app_id,
        feishu_app_secret=feishu_app_secret
    )
    
    # 示例：执行飞书保存工作流
    manager.register_workflow("feishu_qa", feishu_workflow)
    
    # 测试问题
    test_questions = [
        {"question": "人工智能的发展现状和未来趋势", "context": "技术分析"},
    ]
    
    for i, q in enumerate(test_questions):
        try:
            result = manager.execute_workflow("feishu_qa", q)
            print(f"问题 {i+1} 的结果已保存到飞书文档: {result.get('feishu_save_01', {}).get('feishu_doc', {}).get('url', '未知URL')}")
        except Exception as e:
            print(f"执行失败: {e}")


#os.environ["APP_ID"] = "cli_a8771266f2fb5013"
 #   os.environ["APP_SECRET"] = "bVh7WT9QoYIFQrowSR2UCepcJTgrTaMs"
  #  os.environ["LARK_DOMAIN"] = "https://open.feishu.cn"