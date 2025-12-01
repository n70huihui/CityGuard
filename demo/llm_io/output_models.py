from pydantic import BaseModel, Field

"""
该文件存放整个 demo 项目所有的 LLM 输出模型
变量名统一为 FunctionNameVo
"""

class ParseUserPromptVo(BaseModel):
    location: tuple[float, float] = Field(description="位置经纬度")
    task: str = Field(description="任务描述")

class HandleObservationVo(BaseModel):
    task_id: str = Field(description="本次观测任务 id")
    car_id: str = Field(description="提供报告的车辆 id")
    observation_timestamp: int = Field(description="观测的时间戳")
    result: str = Field(description="结果")
    evidence: str = Field(description="证据")