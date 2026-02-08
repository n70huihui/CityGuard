import base64
import json
from collections import defaultdict
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolRuntime

from env_utils.llm_args import *
from guard.common.model import Monitor, MonitorReport, Camera, CameraReport, RootAnalyzeData
from guard.common.prompt import monitor_executor_sys_prompt, camera_executor_sys_prompt


def load_monitors(file_path: str) -> dict[str, Monitor]:
    """加载监控信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {key: Monitor(**monitor_dict) for key, monitor_dict in data.items()}

def load_cameras(file_path: str) -> tuple[dict[str, Camera], dict[str, list[Camera]]]:
    """加载车载摄像头信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    area_camera_dict = defaultdict(list)
    name_camera_dict = {key: Camera(**camera_dict) for key, camera_dict in data.items()}

    for camera in name_camera_dict.values():
        area_camera_dict[camera.camera_area].append(camera)
    return name_camera_dict, area_camera_dict

def load_root_analyze_info(file_path: str) -> dict[str, list[RootAnalyzeData]]:
    """加载根因分析信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {key: [RootAnalyzeData(**root_analyze_data_dict) for root_analyze_data_dict in root_analyze_data_lst] for key, root_analyze_data_lst in data.items()}

monitors = load_monitors('../meta/monitor_info.json')
name_camera_dict, area_camera_dict = load_cameras('../meta/camera_info.json')
root_analyze_info = load_root_analyze_info('../meta/root_analyze_info.json')

monitor_executor = create_agent(
    model=ChatOpenAI(model=visual_model, base_url=base_url, api_key=api_key),
    tools=[],
    response_format=ToolStrategy(MonitorReport)
)

camera_executor = create_agent(
model=ChatOpenAI(model=visual_model, base_url=base_url, api_key=api_key),
    tools=[],
    response_format=ToolStrategy(CameraReport)
)


@dataclass
class PlannerContext:
    """规划器工具调用上下文"""
    type_name: str
    id: int

@tool
def get_monitor_report(monitor_name: str, task_description: str, runtime: ToolRuntime[PlannerContext]) -> MonitorReport:
    """
    获取监控视角对应的分析报告
    :param monitor_name: 监控名称
    :param task_description: 市民举报信息
    :param runtime: 工具运行时上下文
    :return: 监控视角分析报告
    """
    # print(f"get_monitor_info: {monitor_name}")
    # 提取相关的举报信息
    type_name = runtime.context.type_name
    type_id = str(runtime.context.id)

    # 提取监控编号
    monitor_id = monitor_name.split('_')[1]

    # 获取项目根路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 构建两种可能的图片路径
    # 优先路径：datasets/{type_name}/{type_id}/monitor/{monitor_id}.jpg
    image_path_priority = os.path.join(
        project_root, 'datasets', type_name, type_id, 'monitor', f"{monitor_id}.jpg"
    )
    # 备选路径：datasets/base/monitor/{monitor_id}.jpg
    image_path_fallback = os.path.join(
        project_root, 'datasets', 'base', 'monitor', f"{monitor_id}.jpg"
    )

    # 优先检查新路径，不存在则使用备选路径
    if os.path.exists(image_path_priority):
        image_path = image_path_priority
    elif os.path.exists(image_path_fallback):
        image_path = image_path_fallback
    else:
        # 两个路径都不存在时抛出异常
        raise FileNotFoundError(f"监控图片不存在: {image_path_priority} or {image_path_fallback}")

    # 读取图片并转换为 base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # 智能体分析监控画面
    prompt = monitor_executor_sys_prompt.format(
        monitor=monitors[monitor_name],
        task_description=task_description
    )

    message_content = [
        {"type": "text", "text": prompt.content},
        {"type": "image", "source_type": "base64", "data": encoded_string, "mime_type": "image/jpeg"}
    ]

    inputs = {"messages": [HumanMessage(content=message_content)]}

    response = monitor_executor.invoke(inputs)

    return response["structured_response"]

@tool
def get_camera_report(camera_area: str, task_description: str, runtime: ToolRuntime[PlannerContext]) -> CameraReport:
    """
    获取车载摄像头视角对应的分析报告
    :param camera_area: 车载摄像头所在的区域，示例：area_1
    :param task_description: 市民举报信息
    :param runtime: 工具运行时上下文
    :return: 车载摄像头视角分析报告
    """
    # print(f"get_camera_report:{camera_area}")
    # 提取相关的举报信息
    type_name = runtime.context.type_name
    type_id = str(runtime.context.id)

    # 拿到当前区域的摄像头列表
    camera_lst = area_camera_dict[camera_area]
    camera_content_lst = []

    # 获取项目跟路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 按顺序拿到摄像头视角
    for camera in camera_lst:
        # 拿到区域编号和摄像头编号
        split_camera_name = camera.camera_name.split('_')
        area_id = split_camera_name[1]
        camera_id = split_camera_name[3]

        # 构建两种可嫩的图片路径
        # 优先路径: datasets/{type_name}/{type_id}/cameras/{area_id}/{camera_id}.jpg
        image_path_priority = os.path.join(
            project_root, 'datasets', type_name, type_id, 'cameras', str(area_id), f"{camera_id}.jpg"
        )
        # 备选路径: datasets/base/cameras/{area_id}/{camera_id}.jpg
        image_path_fallback = os.path.join(
            project_root, 'datasets', 'base', 'cameras', str(area_id), f"{camera_id}.jpg"
        )

        # 优先检查新路径，不存在则使用备选路径
        if os.path.exists(image_path_priority):
            image_path = image_path_priority
        elif os.path.exists(image_path_fallback):
            image_path = image_path_fallback
        else:
            # 两个路径都不存在时抛出异常
            raise FileNotFoundError(f"监控图片不存在: {image_path_priority} or {image_path_fallback}")

        # 读取图片并转换为 base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            camera_content_lst.append(encoded_string)

    # 智能体分析摄像头画面
    prompt = camera_executor_sys_prompt.format(
        camera_lst=camera_lst,
        task_description=task_description
    )

    # 初始化消息正文
    message_content = [
        {"type": "text", "text": prompt.content}
    ]

    # 插入图片消息
    for camera_content in camera_content_lst:
        message_content.append({
            "type": "image",
            "source_type": "base64",
            "data": camera_content,
            "mime_type": "image/jpeg"
        })

    inputs = {"messages": [HumanMessage(content=message_content)]}

    response = camera_executor.invoke(inputs)

    return response["structured_response"]

if __name__ == '__main__':
    print(root_analyze_info)