import json
import random
import uuid

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from env_utils.llm_args import *
from vc_guard.common.constants import VEHICLE_SIMPLE_REPORT_KEY
from vc_guard.common.models import ParseUserPromptVo, SummaryVo, BestVehicleListVo
from vc_guard.common.prompts import parse_user_prompt_template, multi_view_understanding_summary_template, \
    get_best_vehicle_id_list_template, llm_judge_template
from vc_guard.edge.vechicle import AgentCard
from vc_guard.globals.executor import vehicle_executor
from vc_guard.globals.memory import long_term_memory_store
from vc_guard.globals.vehicles import vehicles
from vc_guard.grid.map import latlon_to_grid, get_quadrant
from vc_guard.globals.grid import map_simulator


class CloudSolver:
    """
    云端计算
    """
    def __init__(self, max_iter_num=3):
        self.llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model)
        self.max_iter_num = max_iter_num    # 流程重试最大次数

    def _parse_user_prompt(self, user_prompt: str, is_log: bool) -> ParseUserPromptVo:
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

    def _get_agent_cards(self, location: tuple[float, float], is_log: bool) -> list[str]:
        """
        获取任务执行地点附近所有车辆的 agent_card
        :param location: 任务执行地点
        :param is_log: 是否打印日志
        :return: agent_card 列表
        """
        # TODO 这里的 vehicle_list 实际上需要从模拟器中利用 location 获取附近的车辆，这里直接模拟
        car_id_set: set[str] = set([vehicle.car_id for vehicle in vehicles])

        # 线程池并行处理
        agent_cards: list[str] = vehicle_executor.execute_tasks(
            vehicle_list=vehicles,
            best_vehicle_id_set=car_id_set,
            method_name='get_agent_card',
            return_results=True
        )

        if is_log:
            print(f"get_agent_cards ===> ")
            # print(f"location: {location}")
            print(f"agent_cards: {agent_cards}")
            print(f"get_agent_cards <===")
            print()

        return agent_cards

    def _init_map_simulator(self,
                            location: tuple[float, float],
                            agent_card_dict: dict[str, AgentCard],
                            is_log: bool) -> None:
        """
        初始化地图模拟器
        :param location: 目标位置
        :param agent_card_dict: 车辆信息字典
        :param is_log: 是否打印日志
        :return:
        """

        # 将任务地点坐标映射到模拟的网格地图上
        task_location = latlon_to_grid(
            location[0], location[1],
            map_simulator.width, map_simulator.height
        )

        # 测试：中心点为目标点
        map_simulator.add_target_point((15, 15))

        # 将车辆坐标映射到模拟的网格地图上
        for car_id in agent_card_dict.keys():
            agent_card_model = agent_card_dict[car_id]
            # 坐标映射
            agent_card_model.location = latlon_to_grid(
                agent_card_model.location[0], agent_card_model.location[1],
                map_simulator.width, map_simulator.height
            )

        # 网格图中添加车辆
        map_simulator.add_vehicle_id_position(agent_card_dict)

        if is_log:
            map_simulator.visualize()

    def _get_nearby_vehicle_id_list(self,
                                    radius: int,
                                    is_log: bool) -> tuple[list[str], set[int]]:
        """
        获取附近车辆的 id 列表
        :param radius: 搜索半径
        :param is_log: 是否打印日志
        :return: 附近车辆的 id 列表，车辆涉及的象限列表
        """

        # 拿到附近车辆的 id - 位置字典
        nearby_vehicle_id_position_dict: dict[str, tuple[int, int]] = map_simulator.get_nearby_vehicle_id_position_dict(radius)

        # 整理象限集合
        quadrants = set()
        for car_id, position in nearby_vehicle_id_position_dict.items():
            quadrants.add(get_quadrant(position[0], position[1], map_simulator.target_point[0], map_simulator.target_point[1]))

        nearby_vehicle_id_list = [car_id for car_id in nearby_vehicle_id_position_dict.keys()]

        if is_log:
            print(f"get_nearby_vehicle_id_list ===> ")
            print(f"nearby_vehicle_id_list: {nearby_vehicle_id_list}")
            print(f"get_nearby_vehicle_id_list <===")
            print()

        return nearby_vehicle_id_list, quadrants

    def _get_final_report_from_nearby_observation(self,
                                                  nearby_vehicle_id_list: list[str],
                                                  task_location: str,
                                                  task_description: str,
                                                  task_uuid: str,
                                                  is_log: bool) -> SummaryVo:
        """
        从附近车辆的观察中获取最终报告
        :param nearby_vehicle_id_list: 附近车辆 id 列表
        :param task_location: 任务地点描述
        :param task_description: 任务描述
        :param task_uuid: 任务 id
        :param is_log: 是否开启日志
        :return: 最终报告
        """
        vehicle_executor.execute_tasks(
            best_vehicle_id_set=set(nearby_vehicle_id_list),
            method_name='execute_task',
            args=(task_location, task_description, task_uuid, is_log)
        )

        simple_report_list = long_term_memory_store.get_list(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid))

        return self._multi_view_understand(
            simple_report_list=simple_report_list,
            task_description=task_description,
            task_uuid=task_uuid,
            is_log=is_log
        )

    def _get_best_vehicle_id_list(self,
                                  num_of_vehicles: int,
                                  agent_card_dict: dict,
                                  quadrants: set[int],
                                  is_log: bool) -> BestVehicleListVo:
        """
        挑选最优的车辆执行任务
        :param num_of_vehicles: 执行任务的车辆数量
        :param agent_card_dict: 车辆信息字典
        :param quadrants: 检测过的象限集合
        :return: 车辆 id 列表以及终点坐标
        """
        # 使用大模型筛选车辆
        agent = create_agent(model=self.llm, tools=[], response_format=ToolStrategy(BestVehicleListVo))

        agent_card_models = agent_card_dict.values()

        prompt = get_best_vehicle_id_list_template.format(
            grid_matrix=map_simulator.grid_matrix,
            observed_quadrant=quadrants,
            task_location=map_simulator.target_point,
            num_of_vehicles=num_of_vehicles,
            agent_card_models=agent_card_models
        )

        response = agent.invoke({"messages": [prompt]})

        # 拿到最优车辆的 id 列表
        best_vehicle_id_list_vo: BestVehicleListVo = response["structured_response"]

        if is_log:
            print(f"get_best_vehicle_id_list ===> ")
            print(f"observed_quadrant: {quadrants}")
            print(f"best_vehicle_id_list: {best_vehicle_id_list_vo.best_vehicle_id_list}")
            print(f"best_vehicle_position_list: {best_vehicle_id_list_vo.best_vehicle_target_points_list}")
            print(f"get_best_vehicle_id_list <===")
            print()

        return best_vehicle_id_list_vo

    def _plan_path(self,
                   selected_vehicle_locations: list[tuple[int, int]],
                   is_log: bool,
                   target: tuple[int, int] = None) -> None:
        """
        路径规划，使用 A* 算法
        :param selected_vehicle_locations: 选中的车辆坐标
        :param is_log: 是否打印日志
        :param target: 目标位置
        :return:
        """
        vehicle_paths = {}

        for i, position in enumerate(selected_vehicle_locations):
            vehicle_id = f"Vehicle-{i + 1}"
            path = map_simulator.find_path(position, map_simulator.target_point if not target else target)
            vehicle_paths[vehicle_id] = path

        if is_log:
            map_simulator.visualize(vehicle_paths=vehicle_paths)

    def _multi_view_understand(self,
                               simple_report_list: list,
                               task_description: str,
                               task_uuid: str,
                               is_log: bool,
                               ) -> SummaryVo:
        """
        云端多视角理解
        :param simple_report_list: 车辆简易报告
        :param task_description: 任务描述
        :param task_uuid: 任务 id
        :param is_log: 是否打印日志
        :return: 最终报告
        """
        prompt = multi_view_understanding_summary_template.format(
            simple_report_list=simple_report_list,
            task_id=task_uuid,
            task_description=task_description
        )

        agent = create_agent(model=self.llm, tools=[], response_format=ToolStrategy(SummaryVo))

        # 6. 解析输出
        response = agent.invoke({"messages": [prompt]})

        if is_log:
            print(f"summary ===> ")
            print(f"prompt: {prompt}")
            print(f"summary_list: {simple_report_list}")
            print(f"summary <===")
            print()

        final_report = response["structured_response"]
        return final_report

    def _llm_judge(self, task_description: str, final_report: SummaryVo) -> str:
        """
        云端 LLM 判断
        :param final_report: 最终报告
        :return: 是否执行任务
        """
        prompt = llm_judge_template.format(
            task_description=task_description,
            final_report=final_report
        )

        agent = create_agent(model=self.llm, tools=[])

        response = agent.invoke({"messages": [prompt]})
        # 解析输出
        content = response["messages"][-1].content_blocks
        status = content[0]['text']

        return status

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
        parse_user_prompt_vo: ParseUserPromptVo = self._parse_user_prompt(user_prompt, is_log)
        task_description = parse_user_prompt_vo.task
        location = parse_user_prompt_vo.location
        task_location = parse_user_prompt_vo.task_location

        # 2. 云端下发广播，所有车辆返回自身的 agent_card，并对 agent_card 进行实例化以及字典化
        agent_cards = self._get_agent_cards(location, is_log)
        agent_card_dict: dict[str, AgentCard] = {}
        for agent_card in agent_cards:
            agent_card_model = AgentCard(**json.loads(agent_card))
            agent_card_dict[agent_card_model.car_id] = agent_card_model

        # 3. 初始化地图模拟器：车辆坐标映射，添加目标点
        self._init_map_simulator(location, agent_card_dict, is_log)

        final_report = None

        # 4. 根据目标位置，搜索附近的车辆，直接获取历史记录
        nearby_vehicle_id_list, quadrants = self._get_nearby_vehicle_id_list(3, is_log)
        if nearby_vehicle_id_list:
            final_report = self._get_final_report_from_nearby_observation(nearby_vehicle_id_list, task_location, task_description, task_uuid, is_log)

        if final_report:
            # 大模型评估，查看是否能够分析出根因
            status = self._llm_judge(task_description, final_report)

            # 仅靠历史观测就能分析出根因，直接返回对应的报告
            if status != "CONTINUE":
                return final_report

        # 删除作废车辆
        for car_id in nearby_vehicle_id_list:
            del agent_card_dict[car_id]

        map_simulator.remove_vehicle_id_position(nearby_vehicle_id_list)

        # 5. 当附近车辆的历史记录没办法完成任务时，主动调度车辆前往
        is_continue = True
        iter_num = 0
        while is_continue and iter_num < self.max_iter_num:

            # 挑选最优的车辆执行任务
            best_vehicle_vo = self._get_best_vehicle_id_list(num_of_vehicles, agent_card_dict, quadrants, is_log)

            best_vehicle_id_list = best_vehicle_vo.best_vehicle_id_list
            best_vehicle_target_list = best_vehicle_vo.best_vehicle_target_points_list

            if not best_vehicle_id_list or not best_vehicle_target_list or len(best_vehicle_id_list) != len(best_vehicle_target_list):
                raise Exception("没有找到合适的车辆执行任务")

            for idx, vehicle_id in enumerate(best_vehicle_id_list):
                target = best_vehicle_target_list[idx]
                selected_position = agent_card_dict[vehicle_id].location
                # 路径规划
                self._plan_path([selected_position], is_log, target)
                # 模拟车辆到达预设地点
                for vehicle in vehicles:
                    if vehicle.car_id == vehicle_id:
                        vehicle.location = target
                quadrant_idx = get_quadrant(target[0], target[1], map_simulator.target_point[0], map_simulator.target_point[1])
                quadrants.add(quadrant_idx)

            # 6. 执行任务，线程池并行模拟
            vehicle_executor.execute_tasks(
                best_vehicle_id_set=set(best_vehicle_id_list),
                method_name='execute_task',
                args=(task_location, task_description, task_uuid, is_log)
            )

            # 获取车辆的简单报告列表
            simple_report_list = long_term_memory_store.get_list(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid))

            # 7. 多视角理解
            final_report = self._multi_view_understand(
                simple_report_list=simple_report_list,
                task_description=task_description,
                task_uuid=task_uuid,
                is_log=is_log
            )

            # 8. 大模型评估，查看是否能够分析出根因
            status = self._llm_judge(task_description, final_report)
            is_continue = True if status == "CONTINUE" else False

            # 无法分析根因，需要选择其他车辆来进行辅助，这里剔除掉已经选择过的车辆
            if is_continue:
                for vehicle_id in best_vehicle_id_list:
                    del agent_card_dict[vehicle_id]
                map_simulator.remove_vehicle_id_position(best_vehicle_id_list)

            iter_num += 1

        # 9. 返回最终报告
        return final_report