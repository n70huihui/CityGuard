import json
import random
import uuid

from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver

from env_utils.llm_args import *
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from vc_guard.common.constants import VEHICLE_OBSERVATION_KEY, VEHICLE_SIMPLE_REPORT_KEY
from vc_guard.common.models import HandleObservationVo
from vc_guard.globals.memory import long_term_memory_store
from vc_guard.observations.handlers import BaseObservationHandler, SimpleImageObservationHandler

@dataclass
class AgentCard:
    """
    AgentCard 类，用于模拟 A2A 协议，表示车辆的能力
    """
    car_id: str
    location: tuple[float, float]
    is_working: bool
    ability: list[str]
    speed: float
    direction: float


def get_vehicle_original_observation(task_id: str, car_id: str) -> list:
    """
    根据任务 id 和车辆 id 获取车辆的原始观测信息
    :param task_id: 任务 id
    :param car_id: 车辆 id
    :return: 原始观测信息
    """
    return long_term_memory_store.get_list(VEHICLE_OBSERVATION_KEY.format(task_uuid=task_id, car_id=car_id))


class Vehicle:
    """
    车辆类，用于模拟车辆，每一辆车内置 Agent
    """
    def __init__(self, observation_handler: BaseObservationHandler = SimpleImageObservationHandler()):
        # 车辆属性赋值
        self.car_id: str = uuid.uuid4().__str__()
        self.location: tuple[float, float] = (random.uniform(-90, 90), random.uniform(-180, 180))
        self.is_working: bool = False
        self.ability: list[str] = []
        self.speed: float = random.uniform(30, 60)
        self.direction: float = random.uniform(0, 360)
        self.observation_handler = observation_handler  # 车辆观测处理器

        # 车辆内置 Agent，分有文本和视觉两个
        self.agent = create_agent(
            model=ChatOpenAI(api_key=api_key, base_url=base_url, model=visual_model),
            tools=[],
            response_format=ToolStrategy(HandleObservationVo),
            checkpointer=InMemorySaver()  # short-term-memory
        )

    def get_agent_card(self) -> str:
        """
        获取车辆的 AgentCard 字符串，模拟 A2A 协议，表示车辆的能力
        :return: AgentCard (json)
        """
        def to_dict() -> dict[str, object]:
            return {
                "car_id": self.car_id,
                "location": self.location,
                "is_working": self.is_working,
                "ability": self.ability,
                "speed": self.speed,
                "direction": self.direction
            }

        return json.dumps(to_dict(), indent=2)

    def execute_task(self,
                     task_location: str,
                     task_description: str,
                     task_uuid: str,
                     is_log: bool
                     ) -> None:
        """
        执行任务
        :param task_description: 任务描述
        :param task_location: 任务地点
        :param task_uuid: 任务 UUID
        :param is_log: 是否打印日志
        :return: None
        """

        # 车辆开始工作，工作中的车辆会被排除
        self.is_working = True

        # 调用观测处理器处理观测
        observation, timestamp = self.observation_handler.get_observation()
        simple_report = self.observation_handler.handle_observation(
            agent=self.agent,
            observation=observation,
            timestamp=timestamp,
            task_location=task_location,
            task_description=task_description,
            task_uuid=task_uuid,
            car_id=self.car_id,
            car_direction=self.direction
        )

        # 车辆生成完本次的报告后，保存报告到云上
        long_term_memory_store.rpush(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid), simple_report)

        if is_log:
            print(f"execute_task ===> ")
            print(f"car_id: {self.car_id}")
            # print(f"observation: {observation}")
            print(f"simple_report: {simple_report}")
            # print(f"long_term_list_memory: {long_term_memory.list_memory}")
            print(f"execute_task <===")
            print()

        self.is_working = False