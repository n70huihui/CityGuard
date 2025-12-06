from demo.observations.ImageObservationHandler import ImageObservationHandler
from demo.vehicle.Vehicle import Vehicle

# 初始化车辆，数量为 5
vehicle_list = [Vehicle(observation_handler=ImageObservationHandler()) for _ in range(5)]