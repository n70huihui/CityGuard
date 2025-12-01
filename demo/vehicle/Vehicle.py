import json
import random
import uuid

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from env_utils.llm_args import *

class Vehicle:
    """
    车辆类，用于模拟车辆，每一辆车内置 Agent
    """
    def __init__(self):
        # 车辆属性赋值
        self.car_id: str = uuid.uuid4().__str__()
        self.location: tuple[float, float] = (random.uniform(-90, 90), random.uniform(-180, 180))
        self.is_working: bool = False
        self.ability: list[str] = []
        self.speed: float = random.uniform(30, 60)

        # 车辆内置 LLM Agent
        # self.agent = create_agent(
        #     model=ChatOpenAI(api_key=api_key, base_url=base_url, model=model),
        #     tools=[]
        # )

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
                "speed": self.speed
            }

        return json.dumps(to_dict(), indent=2)
