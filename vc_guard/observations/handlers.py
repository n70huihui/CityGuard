import base64
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage

from vc_guard.common.models import HandleObservationVo
from langgraph.graph.state import CompiledStateGraph

from vc_guard.common.prompts import handle_text_observation_template

def get_project_root() -> Any:
    """
    获取当前脚本所在目录的绝对路径
    :return: 项目路径
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上两级获取项目根路径（CityGuard 层级）
    project_root = os.path.dirname(os.path.dirname(current_script_dir))
    return project_root

def file_image_to_base64(dataset_path, type_name: str, file_id: str, file_name: str) -> list[str]:

    target_dir = os.path.join(dataset_path, type_name, file_id, file_name)

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

def single_file_image_to_base64(file_name: str) -> list[str]:
    """
    将单个文件夹中的图片转换为 base64 编码
    :param file_name: 图片文件名
    :return: base64 编码
    """
    project_root = get_project_root()

    # 拼接目标目录路径
    dataset_dir = os.path.join(project_root, "datasets")

    return file_image_to_base64(dataset_dir, "garbage", "1", file_name)

class BaseObservationHandler(ABC):
    """
    观测处理器抽象类，用来处理观测数据
    """
    @abstractmethod
    def get_observation(self, *args) -> tuple[str | list[str] | dict[str, object], int]:
        """
        获取观测数据，返回观测数据和观测时间
        :param: *args: 可变参数，用于传递额外的参数
        :return: 观测数据和观测时间
        """
        pass

    @abstractmethod
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
        """
        处理观测数据，返回处理结果
        :param agent: agent
        :param observation: 观测数据
        :param timestamp: 观测时间
        :param task_location: 任务地点
        :param task_description: 任务描述
        :param task_uuid: 任务 id
        :param car_id: 车辆 id
        :param car_direction: 车辆方向
        :return: 处理结果报告
        """
        pass

class ImageObservationHandler(BaseObservationHandler):
    """
    图像观测处理器，处理多个栏目的内容
    """
    def get_observation(self, *args) -> tuple[str | list[str] | dict[str, object], int]:
        cars: list[str] = ['car1', 'car2', 'car3', 'car4']

        choice = random.choice(cars)
        print(choice)

        base64_lst = file_image_to_base64(args[0], args[1], args[2], choice)

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


class SimpleImageObservationHandler(BaseObservationHandler):
    """
    建议图像观测处理器，只处理一个栏目的内容
    """
    def get_observation(self, *args) -> tuple[str | list[str] | dict[str, object], int]:
        cars: list[str] = ['car1', 'car2', 'car3', 'car4']

        choice = random.choice(cars)
        print(choice)

        base64_lst = single_file_image_to_base64(choice)

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