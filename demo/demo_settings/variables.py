from demo.edge.Memory import Memory
from demo.vehicle.Vehicle import Vehicle

# 初始化车辆，数量为 5
vehicle_list = [Vehicle() for _ in range(5)]

# 初始化第三方存储
memory = Memory()