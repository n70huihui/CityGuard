from langgraph.graph.state import CompiledStateGraph

from demo.llm_io.output_models import HandleObservationVo
from demo.observations.ObservationHandler import ObservationHandler

class ImageObservationHandler(ObservationHandler):
    """
    图片观测处理器，使用图片模拟观测数据
    """
    def get_observation(self) -> tuple[str | list[str] | dict[str, object], int]:

        pass

    def handle_observation(self,
                           text_agent: CompiledStateGraph,
                           visual_agent: CompiledStateGraph,
                           observation: str | list[str] | dict[str, object],
                           timestamp: int,
                           task_description: str,
                           task_uuid: str, car_id: str
                           ) -> HandleObservationVo:
        pass