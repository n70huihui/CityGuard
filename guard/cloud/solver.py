import uuid
from guard.agent.planner import Planner


# class CloudSolver:
#     def __init__(self, user_prompt: str):
#         self.planner = Planner(user_prompt=user_prompt)
#
#     def run(self) -> None:
#         # 生成任务 uuid
#         task_uuid = uuid.uuid4().__str__()
#
#         self.planner.run(task_uuid=task_uuid)
#
# if __name__ == "__main__":
#     user_prompt = "road_1_1 附近有异味"
#     solver = CloudSolver(user_prompt=user_prompt)
#     solver.run()