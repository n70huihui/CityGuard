import json
import random
import time
import uuid

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from demo.llm_io.output_models import HandleObservationVo
from demo.llm_io.system_prompts import handle_observation_template, multi_view_understanding_template
from env_utils.llm_args import *
from demo.globals.memory import long_term_memory
from demo.constants.memory_constants import *

class Vehicle:
    """
    车辆类，用于模拟车辆，每一辆车内置 Agent
    """
    def __init__(self):
        # 车辆属性赋值
        self.car_id: str = uuid.uuid4().__str__()
        self.location: tuple[float, float] = (random.uniform(-90, 90), random.uniform(-180, 180))
        self.is_working: bool = False   # TODO 这里的状态是不是要搞多一点，比如说正在执行任务，正在多视角理解等，粒度更细一点？
        self.ability: list[str] = []
        self.speed: float = random.uniform(30, 60)
        # TODO 车辆朝向

        # 车辆内置 LLM Agent
        self.agent = create_agent(
            model=ChatOpenAI(api_key=api_key, base_url=base_url, model=model),
            tools=[self.__get_vehicle_original_observation],
            response_format=ToolStrategy(HandleObservationVo),
            checkpointer=InMemorySaver()    # short-term-memory
        )

    def __get_vehicle_original_observation(self,
                                           task_id: str,
                                           car_id: str) -> list:
        """
        根据任务 id 和车辆 id 获取车辆的原始观测信息
        :param task_id: 任务 id
        :param car_id: 车辆 id
        :return: 原始观测信息
        """
        return long_term_memory.get_list(VEHICLE_OBSERVATION_KEY.format(task_uuid=task_id, car_id=car_id))

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

    def __get_text_observation(self) -> tuple[str, int]:
        """
        获取车辆的观测信息，使用文本模拟
        :return: 观测信息，观测时间
        """

        observations: list[str] = [
            "在本车辆观测中，阜埠河路东侧有两辆单车违停",
            "在本车辆观测中，阜埠河路西侧有三辆单车违停，另外还有两辆单车停得东倒西歪",
            "在本车辆观测中，没有单车违停",
            "在本车辆观测中，路面被其他行人、车辆挡住，看不到什么有效信息"
        ]

        return random.choice(observations), int(time.time())

    def __handle_observation(self, observation: str,
                             timestamp: int,
                             task_description: str,
                             task_uuid: str) -> HandleObservationVo:
        """
        处理车辆观测信息
        :param observation: 车辆观测信息
        :param timestamp: 观测时间
        :param task_description: 任务描述
        :param task_uuid: 任务 UUID
        :return:
        """

        # TODO 这里如果是视频，应该先抽取关键帧，然后存关键帧上去
        long_term_memory.rpush(VEHICLE_OBSERVATION_KEY.format(task_uuid=task_uuid, car_id=self.car_id), observation)

        # 提供车辆报告
        prompt = handle_observation_template.format(
            observation=observation,
            task_description=task_description,
            task_id=task_uuid,
            car_id=self.car_id,
            observation_timestamp=timestamp
        )
        response = self.agent.invoke({"messages": [prompt]}, {"configurable": {"thread_id": task_uuid}})
        return response["structured_response"]

    def execute_task(self, task_description: str, task_uuid: str, is_log: bool) -> None:
        """
        执行任务
        :param task_description: 任务描述
        :param task_uuid: 任务 UUID
        :param is_log: 是否打印日志
        :return: None
        """

        # 车辆开始工作，工作中的车辆会被排除
        self.is_working = True

        observation, timestamp = self.__get_text_observation()
        simple_report = self.__handle_observation(observation, timestamp, task_description, task_uuid)

        # 车辆生成完本次的报告后，保存报告到云上
        # TODO 这里理论上最好还是在本地保存一份自己的报告，不然的话要依靠 agent 的记忆功能，目前的话暂时依靠 agent 的 short-term-memory 来存
        long_term_memory.rpush(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid), simple_report)

        if is_log:
            print(f"execute_task ===> ")
            print(f"car_id: {self.car_id}")
            print(f"observation: {observation}")
            print(f"simple_report: {simple_report}")
            # print(f"long_term_list_memory: {long_term_memory.list_memory}")
            print(f"execute_task <===")
            print()

        self.is_working = False

    def multi_view_understanding(self,
                                 simple_report_list: list[HandleObservationVo],
                                 task_description: str,
                                 task_uuid: str,
                                 is_log: bool) -> None:
        """
        多视角理解
        :param simple_report_list: 简单报告列表
        :param task_description: 任务描述
        :param task_uuid: 任务 id
        :param is_log: 是否打印日志
        :return: None
        """
        self.is_working = True

        # 整理出自己的报告以及别人的报告列表
        my_report = None
        other_report_list = []
        for simple_report in simple_report_list:
            if simple_report.car_id == self.car_id:
                my_report = simple_report
            else:
                other_report_list.append(simple_report)

        # 车辆修正自己的报告
        prompt = multi_view_understanding_template.format(
            simple_report_list=other_report_list,
            self_report=my_report,
            task_description=task_description
        )
        response = self.agent.invoke({"messages": [prompt]}, {"configurable": {"thread_id": task_uuid}})
        final_report = response["structured_response"]

        # 车辆生成完最终的报告后，保存报告到云上
        long_term_memory.rpush(VEHICLE_FINAL_REPORT_KEY.format(task_uuid=task_uuid), final_report)

        if is_log:
            print(f"multi_view_understanding ===> ")
            print(f"car_id: {self.car_id}")
            print(f"simple_report: {my_report}")
            print(f"final_report: {final_report}")
            # print(f"long_term_list_memory: {long_term_memory.list_memory}")
            print(f"multi_view_understanding <===")
            print()

        self.is_working = False