import time
import uuid, json, heapq

from demo.globals.vehicles import vehicle_list
from demo.globals.memory import long_term_memory
from demo.vehicle.AgentCard import AgentCard

"""
该文件存放一些测试代码
"""

def test_car_choose():
    target_location = (0, 0)

    agent_cards = [vehicle.get_agent_card() for vehicle in vehicle_list]

    agent_card_models = [AgentCard(**json.loads(agent_card)) for agent_card in agent_cards]
    print(agent_card_models)

    # (距离, 速度) 优先短距离，慢速度
    scores = []
    score_heap = []
    for agent_card in agent_card_models:
        distance = (agent_card.location[0] - target_location[0]) ** 2 + (
                    agent_card.location[1] - target_location[1]) ** 2
        score = -distance * 0.7 - agent_card.speed * 0.3
        scores.append(-score)
        heapq.heappush(score_heap, score)
        if len(score_heap) > 3:
            heapq.heappop(score_heap)

    print(sorted(scores))

    while score_heap:
        score = heapq.heappop(score_heap)
        print(-score)

def test_memory():
    long_term_memory.rpush("test", "test")
    long_term_memory.rpush("test", "test1")
    print(long_term_memory.get_list("test"))

def test_executor():
    from demo.vehicle.VehicleExecutor import VehicleExecutor
    from demo.globals.vehicles import vehicle_list

    t1 = time.time()
    vehicle_executor = VehicleExecutor()
    vehicle_executor.execute_tasks(vehicle_list)
    t2 = time.time()

    t11 = time.time()
    for vehicle in vehicle_list:
        vehicle.exec()
    t22 = time.time()
    print(f"执行时间: {t2 - t1}")
    print(f"执行时间: {t22 - t11}")

test_executor()