import random
import time

from langgraph.graph.state import CompiledStateGraph

from demo.constants.memory_constants import VEHICLE_OBSERVATION_KEY
from demo.globals.memory import long_term_memory
from demo.llm_io.output_models import HandleObservationVo
from demo.llm_io.system_prompts import handle_text_observation_template
from demo.observations.ObservationHandler import ObservationHandler

class TextObservationHandler(ObservationHandler):
    """
    文本观测处理器，使用文本模拟观测数据
    """
    def get_observation(self) -> tuple[str | list[str] | dict[str, object], int]:
        observations: list[str] = [
            "在本车辆观测中，阜埠河路东侧有两辆单车违停",
            "在本车辆观测中，阜埠河路西侧有三辆单车违停，另外还有两辆单车停得东倒西歪",
            "在本车辆观测中，没有单车违停",
            "在本车辆观测中，路面被其他行人、车辆挡住，看不到什么有效信息"
        ]
        return random.choice(observations), int(time.time())

    def handle_observation(self,
                           text_agent: CompiledStateGraph,
                           visual_agent: CompiledStateGraph,
                           observation: str | list[str] | dict[str, object],
                           timestamp: int,
                           task_description: str,
                           task_uuid: str,
                           car_id: str
                           ) -> HandleObservationVo:
        long_term_memory.rpush(VEHICLE_OBSERVATION_KEY.format(task_uuid=task_uuid, car_id=car_id), observation)

        # 提供车辆报告
        prompt = handle_text_observation_template.format(
            observation=observation,
            task_description=task_description,
            task_id=task_uuid,
            car_id=car_id,
            observation_timestamp=timestamp
        )
        response = text_agent.invoke({"messages": [prompt]}, {"configurable": {"thread_id": task_uuid}})
        return response["structured_response"]