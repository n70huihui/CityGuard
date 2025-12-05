from concurrent.futures import ThreadPoolExecutor

from demo.vehicle.Vehicle import Vehicle


class VehicleExecutor:
    """
    使用线程池模拟车辆并行执行任务
    """
    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_tasks(self, vehicles: list[Vehicle]):
        futures = []
        for vehicle in vehicles:
            future = self.executor.submit(vehicle.exec)
            futures.append(future)

        # 等待所有任务执行完成
        for future in futures:
            future.result()