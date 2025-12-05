from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from demo.llm_io.output_models import SummaryVo

class AgentCoordinator(ABC):
    """
    Agent 流程编排器，用于切换不同的 Agent 流程，基类
    """
    @abstractmethod
    def multi_agent_execute(self,
                            best_vehicle_id_set: set[str],
                            task_uuid: str,
                            task_description: str,
                            llm: BaseChatModel,
                            is_log: bool) -> SummaryVo:
        """
        多 Agent 执行任务，返回总结报告
        :param best_vehicle_id_set: 最佳车辆 id 集合
        :param task_uuid: 任务 uuid
        :param task_description: 任务描述
        :param llm: 云端大模型
        :param is_log: 是否打印日志
        :return: 总结报告
        """
        pass