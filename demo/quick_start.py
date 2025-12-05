from demo.cloud.CloudSolver import CloudSolver
from demo.coordinators.MultiViewByVehicleCoordinator import MultiViewByVehicleCoordinator

# 初始化云端服务
solver = CloudSolver(coordinator=MultiViewByVehicleCoordinator())

# 云端下发查询
user_prompt = "请帮我查询长沙市岳麓区阜埠河路附近的单车违停情况"
report = solver.query(user_prompt, is_log=True)
print(report)