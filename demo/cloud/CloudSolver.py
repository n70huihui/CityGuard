import random, json, heapq
import uuid

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

from demo.coordinators.AgentCoordinator import AgentCoordinator
from demo.llm_io.system_prompts import parse_user_prompt_template
from demo.llm_io.output_models import ParseUserPromptVo, SummaryVo
from demo.globals.vehicles import vehicle_list
from demo.vehicle.AgentCard import AgentCard
from env_utils.llm_args import *

class CloudSolver:
    """
    云端计算
    """
    def __init__(self, coordinator: AgentCoordinator):
        self.llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model)
        self.coordinator = coordinator  # 多智能体协作执行器，可以切换不同的执行流程

    def __parse_user_prompt(self, user_prompt: str, is_log: bool) -> ParseUserPromptVo:
        """
        解析用户输入，从用户输入中提取出需要执行的任务以及对应的位置坐标信息
        :param user_prompt: 用户输入
        :param is_log: 是否打印日志
        :return: 任务 + 位置对象
        """

        def get_location_xy(location: str) -> tuple[float, float]:
            """
            解析位置坐标信息，返回经纬度坐标
            :param location: 位置信息
            :return: 经纬度坐标元组
            """
            # TODO : 这里需要根据实际情况调用第三方 API 或者模拟器 API 解析位置信息，这里简单模拟
            return random.uniform(-90, 90), random.uniform(-180, 180)

        agent = create_agent(
            model=self.llm,
            tools=[get_location_xy],
            response_format=ToolStrategy(ParseUserPromptVo)
        )
        prompt = parse_user_prompt_template.format(user_input=user_prompt)
        response = agent.invoke({"messages": [prompt]})

        if is_log:
            print(f"parse_user_prompt ===> ")
            print(f"prompt: {prompt}")
            print(f"response: {response["structured_response"]}")
            print(f"parse_user_prompt <===")
            print()

        return response["structured_response"]

    def __get_agent_cards(self, location: tuple[float, float], is_log: bool) -> list[str]:
        """
        获取任务执行地点附近所有车辆的 agent_card
        :param location: 任务执行地点
        :param is_log: 是否打印日志
        :return: agent_card 列表
        """
        agent_cards: list[str] = []
        # TODO 这里的 vehicle_list 实际上需要从模拟器中利用 location 获取附近的车辆，这里直接模拟
        for vehicle in vehicle_list:
            agent_cards.append(vehicle.get_agent_card())

        if is_log:
            print(f"get_agent_cards ===> ")
            # print(f"location: {location}")
            print(f"agent_cards: {agent_cards}")
            print(f"get_agent_cards <===")
            print()

        return agent_cards

    def __get_best_best_vehicle_id_list(self,
                                        location: tuple[float, float],
                                        agent_cards: list[str],
                                        num_of_vehicles: int,
                                        is_log: bool) -> list[str]:
        """
        挑选最优的车辆执行任务，使用欧式距离 TOP-K
        :param location: 任务地点坐标
        :param agent_cards: agent_card 列表
        :param num_of_vehicles: 执行任务的车辆数量
        :return: 车辆 id 列表
        """
        agent_card_models = [AgentCard(**json.loads(agent_card)) for agent_card in agent_cards]

        # 堆，TOP-K
        score_heap = []
        for agent_card in agent_card_models:
            # 在工作中的车辆不参与评选
            if agent_card.is_working:
                continue

            # TODO 这里使用欧式距离 + 速度权重来评分，评选又近又慢的车辆，考虑后续改进该评判方式
            distance = (agent_card.location[0] - location[0]) ** 2 + (agent_card.location[1] - location[1]) ** 2
            score = -distance * 0.7 - agent_card.speed * 0.3    # Python 默认小根堆，这里取反实现大根堆
            heapq.heappush(score_heap, (score, agent_card.car_id))
            if len(score_heap) > num_of_vehicles:
                heapq.heappop(score_heap)

        # 拿到最优车辆的 id 列表
        best_vehicle_id_list = [car_id for _, car_id in score_heap]

        if is_log:
            print(f"get_best_vehicle_id_list ===> ")
            print(f"location: {location}")
            print(f"best_vehicle_id_list: {best_vehicle_id_list}")
            print(f"get_best_vehicle_id_list <===")
            print()

        return best_vehicle_id_list

    def query(self,
              user_prompt: str,
              num_of_vehicles: int = 3,
              is_log: bool = False) -> SummaryVo:
        """
        云端下发查询，返回结果报告
        :param num_of_vehicles: 执行任务的车辆数量
        :param user_prompt: 用户输入
        :param is_log: 是否打印日志
        :return: 结果报告
        """
        # 生成本次任务的 uuid，用于 Memory 存储
        task_uuid = "task-" + str(uuid.uuid4()).replace("-", "")

        # 1. 云端 LLM 解析用户输入
        parse_user_prompt_vo = self.__parse_user_prompt(user_prompt, is_log)

        # 2. 云端下发广播，所有车辆返回自身的 agent_card
        agent_cards = self.__get_agent_cards(parse_user_prompt_vo.location, is_log)

        # 3. 挑选最优的车辆执行任务
        best_vehicle_id_list = self.__get_best_best_vehicle_id_list(parse_user_prompt_vo.location ,agent_cards, num_of_vehicles, is_log)
        if not best_vehicle_id_list:
            raise Exception("没有找到合适的车辆执行任务")

        best_vehicle_id_set = set(best_vehicle_id_list)
        task_description = parse_user_prompt_vo.task
        # 4. 多智能体协作执行任务，返回最终总结报告，这里用流程编排器来实现，可以切换不同的执行流程
        final_report = self.coordinator.multi_agent_execute(
            best_vehicle_id_set,
            task_uuid,
            task_description,
            self.llm,
            is_log
        )

        return final_report