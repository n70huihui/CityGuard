import base64
import json
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage

from env_utils.llm_args import *
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from vc_guard.common.constants import VEHICLE_SIMPLE_REPORT_KEY
from vc_guard.common.models import HandleObservationVo, Task, DataUnit
from vc_guard.common.prompts import handle_text_observation_template
from vc_guard.globals.memory import long_term_memory_store
from vc_guard.observations.handlers import BaseObservationHandler, MapImageObservationHandler


@dataclass
class AgentCard:
    """
    AgentCard 类，用于模拟 A2A 协议，表示车辆的能力
    """
    car_id: str
    location: tuple
    is_working: bool
    ability: list[str]
    speed: float
    direction: float


def get_observation(data_unit: DataUnit) -> list:

    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # 向上两级获取项目根路径（CityGuard 层级）
    dataset_path = os.path.dirname(os.path.dirname(current_script_dir))

    target_dir = os.path.join(dataset_path, data_unit.type_name, data_unit.file_id, data_unit.file_name)

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


class Vehicle:
    """
    车辆类，用于模拟车辆，每一辆车内置 Agent
    """
    def __init__(self, height: int = 30, width: int = 30, observation_handler: BaseObservationHandler = MapImageObservationHandler()):
        # 车辆属性赋值
        self.car_id: str = uuid.uuid4().__str__()
        self.map_location: tuple[int, int] = (random.randint(0, height - 1), random.randint(0, width - 1))
        self.is_working: bool = False
        self.speed: float = random.uniform(30, 60)
        self.direction: float = random.uniform(0, 360)
        self.observation_handler = observation_handler  # 车辆观测处理器

        # 车辆内置 Agent
        self.agent = create_agent(
            model=ChatOpenAI(api_key=api_key, base_url=base_url, model=visual_model),
            tools=[],
            response_format=ToolStrategy(HandleObservationVo)
        )

    def get_agent_card(self) -> str:
        """
        获取车辆的 AgentCard 字符串，模拟 A2A 协议，表示车辆的能力
        :return: AgentCard (json)
        """
        def to_dict() -> dict[str, object]:
            return {
                "car_id": self.car_id,
                "map_location": self.map_location,
                "is_working": self.is_working,
                "speed": self.speed,
                "direction": self.direction
            }

        return json.dumps(to_dict(), indent=2)

    def _handle_observation(self, task: Task, observation: list) -> HandleObservationVo:

        # 提供车辆报告
        prompt = handle_text_observation_template.format(
            task_location=task.task_location,
            task_description=task.task_description,
            task_id=task.task_uuid,
            car_id=self.car_id,
            car_direction=self.direction,
            observation_timestamp=time.time()
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

        response = self.agent.invoke(inputs)

        return response["structured_response"]

    def execute_task(self,
                     task: Task,
                     data_unit: DataUnit,
                     is_log: bool
                     ) -> HandleObservationVo | None:
        """
        执行任务
        :param task: 任务对象
        :param data_unit: 数据单元对象
        :param is_log: 是否打印日志
        :return: None
        """
        # 车辆开始工作，工作中的车辆会被排除
        self.is_working = True

        # 获取观测
        observation = get_observation(data_unit)

        # 处理观测
        simple_report = self._handle_observation(task, observation)

        # 车辆生成完本次的报告后，保存报告到云上
        long_term_memory_store.rpush(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task.task_uuid), simple_report)

        if is_log:
            print(f"execute_task ===> ")
            print(f"car_id: {self.car_id}")
            print(f"simple_report: {simple_report}")
            print(f"execute_task <===")
            print()

        self.is_working = False

        return simple_report

class VehicleExecutor:

    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _get_results(self, futures: list[Any], return_results: bool) -> list[Any]:
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                if return_results:
                    results.append(result)
            except Exception as e:
                print(f"任务执行失败: {str(e)}")
                if return_results:
                    results.append(None)
        return results

    """
    使用线程池模拟车辆并行执行任务
    """
    def execute_tasks_different_args(self,
                                     vehicle_id_list: list[str],
                                     vehicles: list[Vehicle] | dict[str, Vehicle],
                                     method_name: str,
                                     args: list[tuple],
                                     return_results: bool = False
                                     ) -> list[object] | None:
        """
        执行车辆对象的指定方法，所有方法都有相同的参数
        :param vehicles: 车辆对象信息
        :param vehicle_id_list: 车辆 id 集合
        :param method_name: 要执行的方法名称
        :param args: 车辆方法的参数元组，默认为 None 表示无参数
        :param return_results: 是否返回结果，默认为 False
        :return: 如果 return_results=True，返回结果列表；否则返回 None
        """
        if len(args) != len(vehicles):
            raise ValueError("参数个数与车辆对象个数不匹配")

        futures = []
        if isinstance(vehicles, list):
            for idx, car_id in enumerate(vehicle_id_list):
                vehicle = None
                for v in vehicles:
                    if v.car_id == car_id:
                        vehicle = v
                        break
                if vehicle is None:
                    continue
                method = getattr(vehicle, method_name)
                future = self.executor.submit(method, *args[idx])
                futures.append(future)
        elif isinstance(vehicles, dict):
            for idx, car_id in enumerate(vehicle_id_list):
                vehicle = vehicles.get(car_id)
                if vehicle is None:
                    continue
                method = getattr(vehicle, method_name)
                future = self.executor.submit(method, *args[idx])
                futures.append(future)

        # 等待所有任务完成
        results = self._get_results(futures, return_results)

        return results if return_results else None


    def execute_tasks(self,
                      vehicle_id_set: set[str],
                      vehicles: list[Vehicle] | dict[str, Vehicle],
                      method_name: str,
                      args: tuple = None,
                      return_results: bool = False
                      ) -> list[object] | None:
        """
        执行车辆对象的指定方法，所有方法都有相同的参数
        :param vehicles: 车辆对象信息
        :param vehicle_id_set: 车辆 id 集合
        :param method_name: 要执行的方法名称
        :param args: 车辆方法的参数元组，默认为 None 表示无参数
        :param return_results: 是否返回结果，默认为 False
        :return: 如果 return_results=True，返回结果列表；否则返回 None
        """
        if args is None:
            args = ()

        futures = []
        if isinstance(vehicles, list):
            for vehicle in vehicles:
                if vehicle.car_id not in vehicle_id_set:
                    continue
                method = getattr(vehicle, method_name)
                future = self.executor.submit(method, *args)
                futures.append(future)
        elif isinstance(vehicles, dict):
            for car_id in vehicle_id_set:
                vehicle = vehicles.get(car_id)
                if vehicle is None:
                    continue
                method = getattr(vehicle, method_name)
                future = self.executor.submit(method, *args)
                futures.append(future)

        # 等待所有任务完成
        results = self._get_results(futures, return_results)

        return results if return_results else None