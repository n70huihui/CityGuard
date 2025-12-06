from abc import ABC, abstractmethod

from langgraph.graph.state import CompiledStateGraph

from demo.llm_io.output_models import HandleObservationVo

class ObservationHandler(ABC):
    """
    观测处理器抽象类，用来处理观测数据
    """
    @abstractmethod
    def get_observation(self) -> tuple[str | list[str] | dict[str, object], int]:
        """
        获取观测数据，返回观测数据和观测时间
        :return: 观测数据和观测时间
        """
        pass

    @abstractmethod
    def handle_observation(self,
                           text_agent: CompiledStateGraph,
                           visual_agent: CompiledStateGraph,
                           observation: str | list[str] | dict[str, object],
                           timestamp: int,
                           task_description: str,
                           task_uuid: str,
                           car_id: str
                           ) -> HandleObservationVo:
        """
        处理观测数据，返回处理结果
        :param text_agent: 文本 agent
        :param visual_agent: 视觉 agent
        :param observation: 观测数据
        :param timestamp: 观测时间
        :param task_description: 任务描述
        :param task_uuid: 任务 id
        :param car_id: 车辆 id
        :return: 处理结果报告
        """
        pass