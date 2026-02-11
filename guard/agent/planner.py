from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from guard.agent.executor import monitors, get_monitor_report, get_camera_report, PlannerContext, root_analyze_info
from langgraph.checkpoint.memory import InMemorySaver

from env_utils.llm_args import *
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from guard.common.prompt import planner_sys_prompt

class Planner:
    """智能体规划器，是主要的智能体实现"""
    def __init__(self,
                 type_name: str,
                 tools: list | None = [get_monitor_report, get_camera_report],
                 system_prompt: str = planner_sys_prompt.format(monitor_info=monitors)):
        """
        智能体初始化
        :param type_name: 类型名称，用于查询监控信息和根因分析信息
        """
        self.type_name: str = type_name
        self.planner: CompiledStateGraph = create_agent(
            model=ChatOpenAI(model=model, base_url=base_url, api_key=api_key),
            tools=tools,
            system_prompt=system_prompt,
            context_schema=PlannerContext,
            checkpointer=InMemorySaver()  # 智能体记忆
        )

    def run(self, task_uuid: str, user_prompt: str, type_id: int) -> str:
        """
        执行智能体规划流程
        :param task_uuid: 任务 uuid
        :param user_prompt: 用户 prompt
        :param type_id: type_name 类型下的 type_id，用于读取数据集
        :return: 最终报告
        """
        response = self.planner.invoke(
            {"messages": [HumanMessage(content=f"市民举报信息如下：{user_prompt}")]},
            {"configurable": {"thread_id": task_uuid}},
            context=PlannerContext(type_name=self.type_name, id=type_id)
        )

        content = response["messages"][-1].content_blocks
        return content[-1]['text']

class DefaultPlanner(Planner):
    """
    默认规划器，用来做默认的测试
    """
    def __init__(self, type_name: str = "garbage", id: int = 0):
        """
        初始化，默认运行 garbage 类型的第一个样例
        """
        super().__init__(type_name=type_name)
        self.data = root_analyze_info[type_name][id]

    def run_default(self) -> str:
        return self.run(task_uuid="uuid-1", user_prompt=self.data.user_prompt, type_id=self.data.id)

    def run_default_stream(self) -> None:
        """流式打印到控制台"""
        for chunk in self.planner.stream(
            {"messages": [HumanMessage(content=f"市民举报信息如下：{self.data.user_prompt}")]},
            {"configurable": {"thread_id": "uuid-1"}},
            context=PlannerContext(type_name=self.data.type_name, id=self.data.id),
            stream_mode="updates"
        ):
            for step, data in chunk.items():
                print(f"step: {step}")
                response = data['messages'][-1]
                if isinstance(response, AIMessage):
                    print(f"AIMessage: {response.content}")
                    if response.tool_calls is not None and len(response.tool_calls) > 0:
                        print(f"ToolCalls: {response.tool_calls}")
                elif isinstance(response, ToolMessage):
                    print(f"ToolMessage: {response.content}")
                elif isinstance(response, HumanMessage):
                    print(f"HumanMessage: {response.content}")
                else:
                    print(f"Message: {response}")

if __name__ == "__main__":
    default_planner = DefaultPlanner(type_name="burn")
    default_planner.run_default()