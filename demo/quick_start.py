from demo.cloud.CloudSolver import CloudSolver

# 初始化云端服务
solver = CloudSolver()

# 云端下发查询
# user_prompt = "请帮我查询长沙市岳麓区阜埠河路附近的单车违停情况"
user_prompt = "请帮我查询长沙市岳麓区阜埠河路附近的垃圾违规堆放情况"
report = solver.query(user_prompt, num_of_vehicles=1, is_log=True)
print(report)