from pydantic import BaseModel, Field

"""
该文件存放整个 demo 项目所有的 LLM 输出模型
变量名统一为 FunctionNameVo
"""

class ParseUserPromptVo(BaseModel):
    location: tuple[float, float] = Field(description="位置经纬度")
    task: str = Field(description="任务描述")