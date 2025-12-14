import base64
import os
import random
import time

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from demo.constants.memory_constants import VEHICLE_OBSERVATION_KEY
from demo.globals.memory import long_term_memory
from demo.llm_io.output_models import HandleObservationVo
from demo.llm_io.system_prompts import handle_text_observation_template, handle_image_observation_template
from demo.observations.ObservationHandler import ObservationHandler


class ImageObservationHandler(ObservationHandler):
    """
    图片观测处理器，使用图片模拟观测数据
    """

    def __image_to_base64(self, image_filename: str) -> str:
        """
        将图片转换为 base64 编码
        :param image_filename: 图片文件名
        :return: base64 编码
        """
        # 获取当前脚本所在目录的绝对路径
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # 向上两级获取项目根路径（CityGuard 层级）
        project_root = os.path.dirname(os.path.dirname(current_script_dir))

        # 拼接自定义文件夹路径
        custom_folder = os.path.join(project_root, "datasets", "garbage", image_filename)

        if not os.path.exists(custom_folder):
            raise FileNotFoundError(f"图片文件不存在: {custom_folder}")

        with open(custom_folder, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_str

    def get_observation(self) -> tuple[str | list[str] | dict[str, object], int]:
        # TODO 这里暂时只在一个子文件夹里随机读取图片，后续需要改为从大文件夹或者对象存储桶里读取图片
        observations: list[str] = [
            'garbage_1.jpg',
            'garbage_2.jpg',
            'garbage_3.jpg',
            'garbage_4.jpg',
            'garbage_5.jpg'
        ]
        choice = random.choice(observations)
        print(choice)
        base64_str = self.__image_to_base64(choice)
        return base64_str, int(time.time())

    def handle_observation(self,
                           agent: CompiledStateGraph,
                           observation: str | list[str] | dict[str, object],
                           timestamp: int,
                           task_location: str,
                           task_description: str,
                           task_uuid: str,
                           car_id: str,
                           car_direction: float
                           ) -> HandleObservationVo:
        # 提供车辆报告
        prompt = handle_text_observation_template.format(
            task_location=task_location,
            task_description=task_description,
            task_id=task_uuid,
            car_id=car_id,
            car_direction=car_direction,
            observation_timestamp=timestamp
        )

        inputs = {"messages": [HumanMessage(content=[
            {"type": "text", "text": prompt.content},
            {
                "type": "image",
                "source_type": "base64",
                "data": observation,
                "mime_type": "image/jpeg"
            }
        ])]}

        response = agent.invoke(inputs, {"configurable": {"thread_id": task_uuid}})
        return response["structured_response"]