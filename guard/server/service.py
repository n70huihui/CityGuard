import uuid
from typing import Generator
import json
import csv
import io

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from guard.agent.planner import Planner
from guard.agent.executor import (
    get_monitor_report,
    get_camera_report,
    PlannerContext,
    monitors,
)
from guard.agent.generator import generator as final_report_generator
from guard.agent.verifier import server_verify
from guard.common.prompt import planner_sys_prompt, generator_sys_prompt
from guard.common.model import FinalReport, VerifyReport


class PlannerService(Planner):
    """继承 Planner 的 Web 服务类"""

    def __init__(self, type_name: str = "garbage"):
        """
        初始化 Planner 服务
        :param type_name: 默认异常类型名称
        """
        super().__init__(
            type_name=type_name,
            tools=[get_monitor_report, get_camera_report],
            system_prompt=planner_sys_prompt.format(monitor_info=monitors),
        )

    def run_stream(self, user_prompt: str, type_name: str, type_id: int, task_uuid: str | None = None) -> Generator[str, None, None]:
        """
        流式执行智能体规划流程
        :param user_prompt: 用户举报信息
        :param type_name: 异常类型名称
        :param type_id: 类型下的案例ID
        :param task_uuid: 任务UUID
        :return: SSE 流式事件
        """
        if task_uuid is None:
            task_uuid = str(uuid.uuid4())

        all_messages = []  # 收集所有消息

        # 发送任务开始事件
        yield self._format_sse_event(
            "reasoning",
            {"message": "任务开始", "task_uuid": task_uuid},
            step=0,
            event_type="reasoning"
        )

        step_count = 0

        for chunk in self.planner.stream(
            {"messages": [HumanMessage(content=f"市民举报信息如下：{user_prompt}")]},
            {"configurable": {"thread_id": task_uuid}},
            context=PlannerContext(type_name=type_name, id=type_id),
            stream_mode="updates"
        ):
            for step, data in chunk.items():
                step_count += 1
                response = data.get('messages', [None])[-1]

                # 收集消息
                if response is not None:
                    all_messages.append(response)

                if isinstance(response, AIMessage):
                    # AI 消息事件 - 推理过程
                    content = response.content if isinstance(response.content, str) else str(response.content)

                    yield self._format_sse_event(
                        "reasoning",
                        {"content": content, "tool_calls": response.tool_calls or []},
                        step=step_count,
                        event_type="reasoning"
                    )

                    # 工具调用事件 - 推理过程
                    if response.tool_calls is not None and len(response.tool_calls) > 0:
                        for tool_call in response.tool_calls:
                            yield self._format_sse_event(
                                "tool_call",
                                {"tool_name": tool_call.get('name'), "tool_args": tool_call.get('args', {})},
                                step=step_count,
                                event_type="reasoning"
                            )

                elif isinstance(response, ToolMessage):
                    # 工具返回结果事件 - 推理过程
                    tool_content = response.content
                    # 如果是结构化对象，转换为字典
                    if hasattr(tool_content, 'model_dump'):
                        tool_content = tool_content.model_dump()

                    yield self._format_sse_event(
                        "tool_message",
                        {"tool_name": response.name, "content": tool_content},
                        step=step_count,
                        event_type="reasoning"
                    )

                elif isinstance(response, HumanMessage):
                    # 用户消息事件
                    yield self._format_sse_event(
                        "human_message",
                        {"content": response.content},
                        step=step_count,
                        event_type="reasoning"
                    )

            # 步骤完成事件
            yield self._format_sse_event(
                "step",
                {"message": f"步骤 {step_count} 完成"},
                step=step_count,
                event_type="reasoning"
            )

        # 发送推理完成事件
        yield self._format_sse_event(
            "reasoning_complete",
            {"message": "推理完成，开始生成报告", "total_steps": step_count},
            step=step_count,
            event_type="reasoning"
        )

        # 使用 generator 生成最终报告（参考 run_with_final_report）
        prompt = generator_sys_prompt.format(user_prompt=user_prompt, agent_response=all_messages)
        final_report_response = final_report_generator.invoke({"messages": [prompt]})
        final_report: FinalReport = final_report_response["structured_response"]

        # 发送最终报告事件 - 前端用绿色渲染
        yield self._format_sse_event(
            "final_report",
            {
                "analyze_goal": final_report.analyze_goal,
                "reasoning_process_report": final_report.reasoning_process_report,
                "final_report": final_report.final_report,
            },
            step=step_count,
            event_type="final_report"
        )

    def run(self, user_prompt: str, type_name: str, type_id: int, task_uuid: str | None = None) -> tuple[str, FinalReport, int]:
        """
        执行智能体规划流程（非流式）
        :param user_prompt: 用户举报信息
        :param type_name: 异常类型名称
        :param type_id: 类型下的案例ID
        :param task_uuid: 任务UUID
        :return: 推理过程、最终报告和步骤数
        """
        if task_uuid is None:
            task_uuid = str(uuid.uuid4())

        response = self.planner.invoke(
            {"messages": [HumanMessage(content=f"市民举报信息如下：{user_prompt}")]},
            {"configurable": {"thread_id": task_uuid}},
            context=PlannerContext(type_name=type_name, id=type_id)
        )

        messages = response["messages"]
        reasoning_content = messages[-1].content_blocks if messages[-1].content_blocks else ""

        # 使用 generator 生成最终报告（参考 run_with_final_report）
        prompt = generator_sys_prompt.format(user_prompt=user_prompt, agent_response=messages)
        final_report_response = final_report_generator.invoke({"messages": [prompt]})
        final_report: FinalReport = final_report_response["structured_response"]

        return reasoning_content, final_report, len(messages)

    @staticmethod
    def _format_sse_event(event: str, data: dict | str, step: int | None = None, event_type: str = "reasoning") -> str:
        """格式化 SSE 事件"""
        json_data = json.dumps({
            "event": event,
            "data": data,
            "step": step,
            "event_type": event_type
        }, ensure_ascii=False)
        return f"data: {json_data}\n\n"


class VerifierService:
    """评估验证服务"""

    @staticmethod
    def _format_sse_event(event: str, data: dict | str, step: int | None = None, event_type: str = "verify") -> str:
        """格式化 SSE 事件"""
        json_data = json.dumps({
            "event": event,
            "data": data,
            "step": step,
            "event_type": event_type,
        }, ensure_ascii=False)
        return f"data: {json_data}\n\n"

    @staticmethod
    def parse_csv(file_content: bytes) -> list[dict]:
        """解析 CSV 文件内容，提取 type_name, id, response 三列"""
        text = file_content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        required_fields = {"type_name", "id", "response"}
        for row in reader:
            if not required_fields.issubset(row.keys()):
                raise ValueError(f"CSV 缺少必要列，需要: {required_fields}")
            rows.append({
                "type_name": row["type_name"].strip(),
                "id": int(row["id"].strip()),
                "response": row["response"].strip(),
            })
        return rows

    def run_stream(self, rows: list[dict]) -> Generator[str, None, None]:
        """
        串行评估每条数据，通过 SSE 流式返回结果
        :param rows: 解析后的 CSV 行列表，每行包含 type_name, id, response
        :return: SSE 事件生成器
        """
        total = len(rows)

        yield self._format_sse_event(
            "verify_start",
            {"message": "开始评估", "total": total},
            step=0,
            event_type="verify",
        )

        for i, row in enumerate(rows):
            try:
                report: VerifyReport = server_verify(
                    type_name=row["type_name"],
                    id=row["id"],
                    response=row["response"],
                )
                yield self._format_sse_event(
                    "verify_item",
                    {
                        "index": i,
                        "type_name": row["type_name"],
                        "id": row["id"],
                        "report": report.model_dump(),
                    },
                    step=i + 1,
                    event_type="verify",
                )
            except Exception as e:
                yield self._format_sse_event(
                    "verify_error",
                    {
                        "index": i,
                        "type_name": row["type_name"],
                        "id": row["id"],
                        "error": str(e),
                    },
                    step=i + 1,
                    event_type="verify",
                )

        yield self._format_sse_event(
            "verify_complete",
            {"message": "评估完成", "total": total},
            step=total,
            event_type="verify",
        )


# 全局服务实例
_planner_service: PlannerService | None = None


def get_planner_service() -> PlannerService:
    """获取 Planner 服务实例"""
    global _planner_service
    if _planner_service is None:
        _planner_service = PlannerService()
    return _planner_service


_verifier_service: VerifierService | None = None


def get_verifier_service() -> VerifierService:
    """获取 Verifier 服务实例"""
    global _verifier_service
    if _verifier_service is None:
        _verifier_service = VerifierService()
    return _verifier_service
