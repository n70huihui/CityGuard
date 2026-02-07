from pydantic import BaseModel, Field

"""
该文件存放整个项目所有的数据模型
"""

class ParseUserPromptVo(BaseModel):
    task_location: str = Field(description="位置地点描述")
    task_description: str = Field(description="任务描述")

class Task(BaseModel):
    task_location: str = Field(description="位置地点描述")
    task_description: str = Field(description="任务描述")
    task_uuid: str = Field(description="任务 id")

class DataUnit(BaseModel):
    type_name: str = Field(description="数据类型名称")
    file_id: str = Field(description="文件 id")
    file_name: str = Field(description="文件名称")

class BestVehicleListVo(BaseModel):
    best_vehicle_id_list: list[str] = Field(description="最优车辆 id 列表")
    best_vehicle_target_points_list: list[tuple[int, int]] = Field(description="车辆对应的终点坐标")

class HandleObservationVo(BaseModel):
    task_id: str = Field(description="本次观测任务 id")
    car_id: str = Field(description="提供报告的车辆 id")
    car_direction: float = Field(description="车辆方向")
    observation_timestamp: int = Field(description="观测的时间戳")
    result: str = Field(description="结果")
    evidence: str = Field(description="证据")

class SummaryVo(BaseModel):
    task_id: str = Field(description="本次任务 id")
    task_description: str = Field(description="任务描述")
    car_id_list: list[str] = Field(description="提供报告的车辆 id 列表")
    observation_timestamp_list: list[int] = Field(description="观测的时间戳列表")
    summary: str = Field(description="总结")