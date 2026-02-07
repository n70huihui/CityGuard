from langchain_core.messages import HumanMessage

from guard.agent.executor import monitors, get_monitor_report, get_camera_report
from langgraph.checkpoint.memory import InMemorySaver

from env_utils.llm_args import *
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from guard.common.prompt import planner_sys_prompt


class Planner:
    """规划器"""
    def __init__(self, task_description: str):
        self.task_description = task_description
        self.planner = create_agent(
            model=ChatOpenAI(model=model, base_url=base_url, api_key=api_key),
            system_prompt=planner_sys_prompt.format(
                monitor_info=monitors,
                task_description=task_description
            ),
            tools=[get_monitor_report, get_camera_report],
            checkpointer=InMemorySaver()    # 智能体记忆
        )

    def run(self, task_uuid: str) -> None:
        response = self.planner.invoke(
            {"messages": [HumanMessage(content=self.task_description)]},
            {"configurable": {"thread_id": task_uuid}}
        )

        content = response["messages"][-1].content_blocks
        print(content)

if __name__ == "__main__":
    user_prompt = "road_1_1 区域有噪音"
    planner = Planner(task_description=user_prompt)
    planner.run("uuid-1")