from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel

from demo.constants.memory_constants import VEHICLE_SIMPLE_REPORT_KEY
from demo.coordinators.AgentCoordinator import AgentCoordinator
from demo.globals.executor import vehicle_executor
from demo.globals.memory import long_term_memory
from demo.llm_io.output_models import SummaryVo
from demo.llm_io.system_prompts import multi_view_understanding_summary_template


class MultiViewByCloudCoordinator(AgentCoordinator):
    """
    流程编排，由云端大模型来做多视角理解
    """
    def __vehicle_execute_task(self,
                               best_vehicle_id_set: set[str],
                               task_location: str,
                               task_description: str,
                               task_uuid: str,
                               is_log: bool) -> None:
        """
        对应的车辆执行任务，这里直接模拟
        :param best_vehicle_id_set: 车辆 id 列表
        :param task_location: 任务地点
        :param task_description: 任务描述
        :param task_uuid: 任务 uuid
        :param is_log: 是否打印日志
        :return: None
        """
        # 线程池并行模拟
        vehicle_executor.execute_tasks(
            best_vehicle_id_set=best_vehicle_id_set,
            method_name='execute_task',
            args=(task_location, task_description, task_uuid, is_log)
        )

    def __multi_view_understanding_and_summarize(self,
                                                 llm: BaseChatModel,
                                                 task_uuid: str,
                                                 task_description: str,
                                                 is_log: bool) -> SummaryVo:
        """
        云端多视角理解
        :param llm: 云端大模型
        :param task_uuid: 任务 id
        :param task_description: 任务描述
        :param is_log: 是否打印日志
        :return: 总结报告
        """
        # 获取车辆的简单报告列表
        simple_report_list = long_term_memory.get_list(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid))

        # 多视角理解
        prompt = multi_view_understanding_summary_template.format(
            simple_report_list=simple_report_list,
            task_id=task_uuid,
            task_description=task_description
        )

        agent = create_agent(model=llm, tools=[], response_format=ToolStrategy(SummaryVo))

        response = agent.invoke({"messages": [prompt]})

        if is_log:
            print(f"summary ===> ")
            print(f"prompt: {prompt}")
            print(f"summary_list: {simple_report_list}")
            print(f"summary <===")
            print()

        return response["structured_response"]


    def multi_agent_execute(self,
                            best_vehicle_id_set: set[str],
                            task_uuid: str,
                            task_description: str,
                            task_location: str,
                            llm: BaseChatModel,
                            is_log: bool) -> SummaryVo:

        # 1. 每辆车执行任务
        self.__vehicle_execute_task(
            best_vehicle_id_set=best_vehicle_id_set,
            task_location=task_location,
            task_description=task_description,
            task_uuid=task_uuid,
            is_log=is_log
        )

        # 2. 云端大模型做多视角理解，返回总结报告
        report = self.__multi_view_understanding_and_summarize(
            llm=llm,
            task_uuid=task_uuid,
            task_description=task_description,
            is_log=is_log
        )

        return report