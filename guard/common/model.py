from pydantic import BaseModel, Field


class Monitor(BaseModel):
    """监控"""
    monitor_name: str = Field(description="监控名称")
    monitor_area: list[str] = Field(description="监控区域")

class MonitorReport(BaseModel):
    """监控视角分析报告"""
    monitor_name: str = Field(description="监控名称")
    monitor_area: list[str] = Field(description="监控区域")
    monitor_content: str = Field(description="监控画面内容")
    monitor_report: str = Field(description="监控视角分析报告")

class Camera(BaseModel):
    """车载摄像头"""
    camera_name: str = Field(description="摄像头名称")
    camera_area: str = Field(description="摄像头所在区域")
    camera_location: str = Field(description="摄像头具体位置")

class CameraReport(BaseModel):
    """车载摄像头分析报告"""
    camera_name_lst: list[str] = Field(description="摄像头名称")
    camera_area_lst: list[str] = Field(description="摄像头所在区域")
    camera_location_lst: list[str] = Field(description="摄像头具体位置")
    camera_content_lst: list[str] = Field(description="摄像头画面内容")
    camera_report_lst: list[str] = Field(description="摄像头视角分析报告")

class RootAnalyzeData(BaseModel):
    """根因分析数据"""
    type_name: str = Field(description="类型名称")
    id: int = Field(description="id 编号")
    user_prompt: str = Field(description="用户提示词")
    root_cause: str = Field(description="根因")

class RootAnalyzeReport(BaseModel):
    """根因分析报告"""
    type_name: str = Field(description="类型名称")
    id: int = Field(description="id 编号")
    response: str = Field(description="智能体报告")
    step: int = Field(description="推理步数")
    score: float = Field(description="推理得分")