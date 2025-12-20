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

    def __image_to_base64(self, type_name: str, file_name: str) -> list[str]:
        """
        将图片转换为 base64 编码
        :param type_name: 图片类型名称
        :param file_name: 图片文件名
        :return: base64 编码
        """
        # 获取当前脚本所在目录的绝对路径
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # 向上两级获取项目根路径（CityGuard 层级）
        project_root = os.path.dirname(os.path.dirname(current_script_dir))

        # 拼接目标目录路径
        target_dir = os.path.join(project_root, "datasets", type_name, file_name)

        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"目录不存在: {target_dir}")

        # 支持的图片扩展名
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        base64_list = []

        # 遍历目录下的所有文件
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)

            # 检查是否为文件且是图片
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in image_extensions:
                try:
                    with open(file_path, "rb") as image_file:
                        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
                        base64_list.append(base64_str)
                except Exception as e:
                    print(f"处理图片 {filename} 失败: {str(e)}")

        if not base64_list:
            raise ValueError(f"目录 {target_dir} 下未找到有效的图片文件")

        return base64_list

    def get_observation(self) -> tuple[str | list[str] | dict[str, object], int]:
        # TODO 这里暂时只在一个子文件夹里随机读取图片，后续需要改为从大文件夹或者对象存储桶里读取图片
        cars: list[str] = [
            'car1',
            'car2',
            'car3'
        ]
        choice = random.choice(cars)
        print(choice)
        base64_lst = self.__image_to_base64("garbage" ,choice)
        return base64_lst, int(time.time())

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

        # 初始化消息正文
        message_content = [
            {"type": "text", "text": prompt.content}
        ]

        # 插入图片消息
        for observation_item in observation:
            message_content.append({
                "type": "image",
                "source_type": "base64",
                "data": observation_item,
                "mime_type": "image/jpeg"
            })

        inputs = {"messages": [HumanMessage(content=message_content)]}

        response = agent.invoke(inputs, {"configurable": {"thread_id": task_uuid}})
        return response["structured_response"]