"""
LangChain适配器 - 将自定义w2.py工作流转换为LangChain架构
"""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain.chains.base import Chain
from langchain_community.llms import QwenLLM
from langchain_core.tools import BaseTool
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any, List
import json
import os

class DailyDataAccumulator(BaseTool):
    """LangChain工具：每日数据累积器"""

    name = "data_accumulator"
    description = "积累每日微博数据，用于定时分析"

    def _run(self, keyword: str) -> str:
        # 集成现有的DailyDataCollector功能
        from w2 import DailyDataCollector
        collector = DailyDataCollector()
        date_str = datetime.now().strftime("%Y%m%d")

        # 这里可以调用现有的数据累积逻辑
        return f"数据累积完成：{keyword}"

class WeiboAnalyzerChain(Chain):
    """微博分析链 - 基于LangChain构建"""

    llm: BaseLanguageModel
    memory: BaseMemory
    prompt_template: PromptTemplate

    @property
    def input_keys(self) -> List[str]:
        return ["keyword", "weibo_data", "historical_context"]
    @property
    def output_keys(self) -> List[str]:
        return ["analysis_result", "push_message"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # 准备prompt
        context = inputs.get("historical_context", "")
        keyword = inputs["keyword"]
        weibo_data = inputs["weibo_data"]

        prompt = self.prompt_template.format(
            keyword=keyword,
            weibo_posts=weibo_data,
            context=context
        )

        # 调用LLM
        response = self.llm.invoke(prompt)

        # 格式化输出
        result = {
            "analysis_result": response.content,
            "push_message": f"【今日关注点简报】\n\n{response.content}\n\n---\n🤖 AI分析推送"
        }

        # 更新记忆
        self.memory.save_context(
            {"input": f"分析关键词：{keyword}"},
            {"output": response.content}
        )

        return result

def create_langchain_weibo_analyzer():
    """创建基于LangChain的微博分析器"""

    # LLM配置 - 替换现有阿里云集成
    llm = QwenLLM(
        model_name="qwen-turbo",
        api_key=os.getenv("QWQEN_API_KEY") or "sk-addb15e06fef4c19a46122a39aac8caa",
        endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )

    # 记忆系统
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 提示词模板
    prompt_template = PromptTemplate(
        template="""你是一位专业的社交媒体分析专家，请根据以下从微博获取的真实用户发言，对"{{ keyword }}"话题进行深入分析。

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

4. **呈现方式**：清新易懂，简洁不啰嗪，不超过400字

**历史分析记录**：
{{ context }}

**微博发言数据**：
{% for post in weibo_posts %}
- [{{ post.username }} @ {{ post.date }}]：{{ post.text }}
{% endfor %}

答案：
""",
        input_variables=["keyword", "weibo_posts", "context"]
    )

    # 创建链
    chain = WeiboAnalyzerChain(
        llm=llm,
        memory=memory,
        prompt_template=prompt_template
    )

    return chain

# 迁移指南：如何从w2.py切换到LangChain
"""
1. 替换LLM调用：
   - 移除自定义SimpleLLMClient
   - 使用LangChain的LLMs (如ChatOpenAI, QwenLLM等)

2. 工作流重构：
   - w2.py的WorkflowManager -> LangChain Chains/Agents
   - NodeConfig -> Chain 或 Tool

3. 记忆系统：
   - TopicMemoryManager -> langchain.memory模块

4. 工具集成：
   - 爬取功能可以包装为BaseTool
   - 数据累积器作为Tool

5. 分析链：
   - 使用Chain类构建完整的分析流程
   - 支持复杂的prompt模板和输出解析
"""

if __name__ == "__main__":
    # 示例使用
    analyzer = create_langchain_weibo_analyzer()
    result = analyzer.invoke({
        "keyword": "金融市场",
        "weibo_data": [{"username": "用户A", "text": "市场走势很好", "date": "2025-12-02"}],
        "historical_context": ""
    })
    print(result)
