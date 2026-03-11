from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    """任务请求模型"""
    user_prompt: str = Field(..., description="市民举报信息")
    type_name: str = Field(default="garbage", description="异常类型名称")
    type_id: int = Field(default=1, description="类型下的具体案例ID")
    task_uuid: str | None = Field(default=None, description="任务UUID，用于会话追踪")


class TaskResponse(BaseModel):
    """任务响应模型（非流式）"""
    task_uuid: str
    reasoning_process: str = Field(..., description="推理过程")
    final_report: dict = Field(..., description="最终格式化报告")
    steps: int


class StreamEvent(BaseModel):
    """流式事件模型"""
    event: str = Field(..., description="事件类型: reasoning, tool_call, tool_message, final_report")
    data: dict = Field(..., description="事件数据")
    step: int | None = Field(default=None, description="当前步骤数")
    event_type: str = Field(..., description="渲染类型: reasoning=推理过程(蓝色), final_report=最终报告(绿色)")


class FinalReportData(BaseModel):
    """最终报告数据"""
    analyze_goal: str = Field(..., description="分析目标")
    reasoning_process_report: str = Field(..., description="推理过程报告")
    final_report: str = Field(..., description="最终结果报告")
