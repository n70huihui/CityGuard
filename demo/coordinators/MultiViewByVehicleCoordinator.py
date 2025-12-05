from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel

from demo.constants.memory_constants import VEHICLE_SIMPLE_REPORT_KEY, VEHICLE_FINAL_REPORT_KEY
from demo.coordinators.AgentCoordinator import AgentCoordinator
from demo.globals.executor import vehicle_executor
from demo.globals.memory import long_term_memory
from demo.llm_io.output_models import SummaryVo
from demo.llm_io.system_prompts import summary_template


class MultiViewByVehicleCoordinator(AgentCoordinator):
    """
    流程编排，由车辆自身来做多视角理解
    """
    def __vehicle_execute_task(self,
                               best_vehicle_id_set: set[str],
                               task_description: str,
                               task_uuid: str,
                               is_log: bool) -> None:
        """
        对应的车辆执行任务，这里直接模拟
        :param best_vehicle_id_set: 车辆 id 列表
        :param task_description: 任务描述
        :param task_uuid: 任务 uuid
        :param is_log: 是否打印日志
        :return: None
        """
        # 线程池并行模拟
        vehicle_executor.execute_tasks(
            best_vehicle_id_set=best_vehicle_id_set,
            method_name='execute_task',
            args=(task_description, task_uuid, is_log)
        )

    def __multi_view_understanding(self,
                                   best_vehicle_id_set: set[str],
                                   task_uuid: str,
                                   task_description: str,
                                   is_log: bool) -> None:
        """
        多视角理解，让每一辆车修正自己的结果
        :param best_vehicle_id_set: 车辆 id 列表
        :param task_uuid: 任务 uuid
        :param task_description: 任务描述
        :param is_log: 是否打印日志
        :return: None
        """
        simple_report_list = long_term_memory.get_list(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid))

        # 线程池并行模拟
        vehicle_executor.execute_tasks(
            best_vehicle_id_set=best_vehicle_id_set,
            method_name='multi_view_understanding',
            args=(simple_report_list, task_description, task_uuid, is_log)
        )

    def __summary(self,
                  llm: BaseChatModel,
                  task_uuid: str,
                  task_description: str,
                  is_log: bool) -> SummaryVo:
        """
        总结
        :param task_uuid: 任务 uuid
        :param task_description: 任务描述
        :param is_log: 是否打印日志
        :return: 最终总结报告
        """

        # 拿到所有车辆的总结报告
        summary_list = long_term_memory.get_list(VEHICLE_FINAL_REPORT_KEY.format(task_uuid=task_uuid))

        prompt = summary_template.format(
            report_list=summary_list,
            task_description=task_description,
            task_id=task_uuid
        )

        # LLM 总结
        agent = create_agent(
            model=llm,
            response_format=ToolStrategy(SummaryVo)
        )
        response = agent.invoke({"messages": [prompt]})

        if is_log:
            print(f"summary ===> ")
            print(f"prompt: {prompt}")
            print(f"summary_list: {summary_list}")
            print(f"summary <===")
            print()

        return response["structured_response"]

    def multi_agent_execute(self,
                            best_vehicle_id_set: set[str],
                            task_uuid: str,
                            task_description: str,
                            llm: BaseChatModel,
                            is_log: bool) -> SummaryVo:
        # 1. 每辆车执行任务
        self.__vehicle_execute_task(best_vehicle_id_set, task_description, task_uuid, is_log)

        # 2. 多视角理解
        self.__multi_view_understanding(best_vehicle_id_set, task_uuid, task_description, is_log)

        # 3. 总结
        final_report = self.__summary(llm, task_uuid, task_description, is_log)

        return final_report