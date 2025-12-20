import base64
import os
import time
import uuid, json, heapq

from demo.globals.executor import vehicle_executor
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
    t1 = time.time()
    car_id_set: set[str] = set([vehicle.car_id for vehicle in vehicle_list])
    card_list = vehicle_executor.execute_tasks(
        car_id_set,
        'get_agent_card',
        return_results=True
    )
    t2 = time.time()
    print(card_list)
    print(t2 - t1)
    print("===>")
    t1 = time.time()
    card_list = []
    for vehicle in vehicle_list:
        card = vehicle.get_agent_card()
        card_list.append(card)
    t2 = time.time()
    print(card_list)
    print(t2 - t1)

def test_file_read():
    # 获取当前脚本所在目录的绝对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # 获取项目根路径（CityGuard 层级）
    project_root = os.path.dirname(os.path.dirname(current_script_dir))  # 向上两级

    # 拼接自定义文件夹路径
    custom_folder = os.path.join(project_root, "your_folder_name", "abc")

    print(f"当前脚本路径: {current_script_dir}")
    print(f"项目根路径: {project_root}")
    print(f"自定义文件夹路径: {custom_folder}")

def test_image_to_base64(file_name):
    # 获取当前脚本所在目录的绝对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # 向上两级获取项目根路径（CityGuard 层级）
    project_root = os.path.dirname(os.path.dirname(current_script_dir))

    # 拼接目标目录路径
    target_dir = os.path.join(project_root, "datasets", "garbage", file_name)

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

if __name__ == '__main__':
    lst = test_image_to_base64("car3")
    print(lst)