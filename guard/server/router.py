import uuid

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from guard.server.schemas import TaskRequest, TaskResponse
from guard.server.service import PlannerService, get_planner_service, VerifierService, get_verifier_service


router = APIRouter(prefix="/api/v1", tags=["planner"])


@router.post("/task", response_model=TaskResponse)
async def create_task(
    request: TaskRequest,
    service: PlannerService = Depends(get_planner_service),
) -> TaskResponse:
    """
    创建任务（非流式）
    返回推理过程和最终格式化报告
    """
    task_uuid = request.task_uuid or str(uuid.uuid4())
    reasoning_process, final_report, steps = service.run(
        user_prompt=request.user_prompt,
        type_name=request.type_name,
        type_id=request.type_id,
        task_uuid=task_uuid,
    )

    return TaskResponse(
        task_uuid=task_uuid,
        reasoning_process=reasoning_process,
        final_report=final_report,
        steps=steps
    )


@router.post("/task/stream")
async def create_task_stream(
    request: TaskRequest,
    service: PlannerService = Depends(get_planner_service),
) -> StreamingResponse:
    """
    创建任务（流式响应）
    使用 Server-Sent Events (SSE) 进行流式输出
    """
    task_uuid = request.task_uuid or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        for event in service.run_stream(
            user_prompt=request.user_prompt,
            type_name=request.type_name,
            type_id=request.type_id,
            task_uuid=task_uuid,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/verify/stream")
async def verify_stream(
    file: UploadFile = File(..., description="CSV 文件"),
    service: VerifierService = Depends(get_verifier_service),
) -> StreamingResponse:
    """
    评估验证（流式响应）
    上传 CSV 文件，串行调用验证模型，通过 SSE 逐条返回评估结果
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="请上传 CSV 文件")

    content = await file.read()
    try:
        rows = service.parse_csv(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV 解析失败: {str(e)}")

    if not rows:
        raise HTTPException(status_code=400, detail="CSV 文件内容为空")

    async def event_generator() -> AsyncGenerator[str, None]:
        for event in service.run_stream(rows):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "CityGuard Planner"}
