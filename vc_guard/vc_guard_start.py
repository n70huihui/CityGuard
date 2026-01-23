from vc_guard.cloud.solver import CloudSolver

solver = CloudSolver()

user_prompt = "经市民举报，长沙市岳麓区阜埠河路附近存在较大异味，请查询根因。"
report = solver.query(user_prompt, num_of_vehicles=2, is_log=True)
print(report)