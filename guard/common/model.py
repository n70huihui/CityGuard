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