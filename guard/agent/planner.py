from langchain_core.messages import HumanMessage

from guard.agent.executor import monitors, get_monitor_report, get_camera_report, PlannerContext, root_analyze_info
from langgraph.checkpoint.memory import InMemorySaver

from env_utils.llm_args import *
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from guard.common.prompt import planner_sys_prompt

class Planner:
    """规划器"""
    def __init__(self, type_name: str):
        self.type_name = type_name
        self.planner = create_agent(
            model=ChatOpenAI(model=model, base_url=base_url, api_key=api_key),
            tools=[get_monitor_report, get_camera_report],
            system_prompt=planner_sys_prompt.format(
                monitor_info=monitors
            ),
            context_schema=PlannerContext,
            checkpointer=InMemorySaver()  # 智能体记忆
        )

    def run(self, task_uuid: str, user_prompt: str, type_id: int) -> str:
        response = self.planner.invoke(
            {"messages": [HumanMessage(content=f"市民举报信息如下：{user_prompt}")]},
            {"configurable": {"thread_id": task_uuid}},
            context=PlannerContext(type_name=self.type_name, id=type_id)
        )

        content = response["messages"][-1].content_blocks
        return content

class DefaultPlanner(Planner):
    """
    默认规划器，用来做默认的测试
    """
    def __init__(self):
        super().__init__(type_name="garbage")
        self.data = root_analyze_info["garbage"][0]

    def run_default(self) -> str:
        return self.run(task_uuid="uuid-1", user_prompt=self.data.user_prompt, type_id=self.data.id)

    def run_default_stream(self) -> str:
        pass

if __name__ == "__main__":
    default_planner = DefaultPlanner()
    result = default_planner.run_default()
    print(result)