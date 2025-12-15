from election_agent import ElectionAgent

from casevo import ModelBase

from casevo import TotLog
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#测试样例模型
class ElectionModel(ModelBase):

    #根据config生成model
    def __init__(self, tar_graph, person_list, llm):
        """
        初始化对话系统中的每个人物和他们的对话流程。

        :param tar_graph: 目标图，表示对话系统的结构。
        :param person_list: 人物列表，包含所有参与对话的人物信息。
        :param llm: 语言模型，用于生成对话内容。
        """
        
        super().__init__(tar_graph, llm)
        
        #设置Agent        
        for cur_id in range(len(person_list)):
            cur_person = person_list[cur_id]
            cur_agent = ElectionAgent(cur_id, self, cur_person, None)
            self.add_agent(cur_agent, cur_id)
        self.vote_data = []
            
    def public_debate(self):
        """
        所有Agent听取一阶段的公开辩论。

        此函数模拟了一次公开辩论的过程。它首先打印出辩论开始的时间，然后读取当前辩论的主题总结，
        接着让每个参与者（agent）听取辩论总结。最后，它将此次辩论的内容记录到日志中，并打印出辩论结束的时间。
        
        该方法不接受任何参数，也没有返回值。
        """
        
        print('public_debate: %d start' % self.schedule.time)
        log_item = {
        "time": self.schedule.time,
        "event_type": "public_debate",
        "status": "start",
        "details": f"Public debate started at time step {self.schedule.time}"
        }
        print(f"调试:log_item 已定义 - {log_item}")
        TotLog.add_model_log(self.schedule.time, 'public_debate', log_item)
        #获取辩论文本
        cur_debate_num = self.schedule.time + 1
        with open('content/%d.txt' % cur_debate_num) as f:
            cur_debate_summary = f.read()
        
        #循环每个agent听取辩论文本
        for cur_agent in self.agents:
            cur_agent.listen(cur_debate_summary, 'public')
        self.collect_vote_data()
        #事件加入日志
        log_item = {
            'debate_content': cur_debate_summary
        }

        print('public_debate: %d end' % self.schedule.time)   
    def collect_vote_data(self):
        """收集当前时间步的投票数据"""
        time_step = self.schedule.time
        vote_counts = {'特朗普': 0, '拜登': 0, '未决定': 0}
        
        for agent in self.agents:
            vote = agent.current_vote if agent.current_vote else '未决定'
            vote_counts[vote] += 1
            
        self.vote_data.append({
            'time_step': time_step,
            'trump': vote_counts['特朗普'],
            'biden': vote_counts['拜登'],
            'undecided': vote_counts['未决定'],
            'total_agents': len(self.agents)
        })
    def reflect(self):
        """
        所有Agent进行一阶段的反思。
        """
        for cur_agent in self.agents:
            cur_agent.reflect()
    

    #整体模型的step函数
    def step(self):
        #听取辩论文本
        self.public_debate()
        #节点自由讨论
        self.schedule.step()
        
        #节点反思
        self.reflect()
        return 0
  
    def visualize_vote_results(self):
        """绘制智能体投票结果的可视化图"""
        if not self.vote_data:
            print("没有投票数据可可视化")
            return
            
        df = pd.DataFrame(self.vote_data)
        
        # 创建可视化图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 子图1：投票数量随时间变化
        ax1.plot(df['time_step'], df['trump'], 'r-', label='特朗普', linewidth=2, marker='o')
        ax1.plot(df['time_step'], df['biden'], 'b-', label='拜登', linewidth=2, marker='s')
        ax1.plot(df['time_step'], df['undecided'], 'gray', label='未决定', linewidth=2, marker='^')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('智能体数量')
        ax1.set_title('智能体投票意向随时间变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：投票比例饼图（最新时间步）
        latest = df.iloc[-1]
        labels = ['特朗普', '拜登', '未决定']
        sizes = [latest['trump'], latest['biden'], latest['undecided']]
        colors = ['red', 'blue', 'lightgray']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'时间步 {latest["time_step"]} 的投票分布')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细数据
        print("\n投票数据详情:")
        print(df.to_string(index=False))
        
        return fig
    
    def get_vote_statistics(self):
        """获取投票统计数据"""
        if not self.vote_data:
            return None
            
        df = pd.DataFrame(self.vote_data)
        latest = df.iloc[-1]
        
        stats = {
            'final_trump_percentage': (latest['trump'] / latest['total_agents']) * 100,
            'final_biden_percentage': (latest['biden'] / latest['total_agents']) * 100,
            'final_undecided_percentage': (latest['undecided'] / latest['total_agents']) * 100,
            'total_time_steps': len(df),
            'max_trump_support': df['trump'].max(),
            'max_biden_support': df['biden'].max()
        }
        
        return stats       