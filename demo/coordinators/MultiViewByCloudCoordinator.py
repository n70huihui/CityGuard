from langchain_core.language_models import BaseChatModel

from demo.coordinators.AgentCoordinator import AgentCoordinator
from demo.llm_io.output_models import SummaryVo

class MultiViewByCloudCoordinator(AgentCoordinator):
    """
    流程编排，由云端大模型来做多视角理解
    """
    def multi_agent_execute(self,
                            best_vehicle_id_set: set[str],
                            task_uuid: str,
                            task_description: str,
                            llm: BaseChatModel,
                            is_log: bool) -> SummaryVo:
        pass