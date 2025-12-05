from concurrent.futures import ThreadPoolExecutor, as_completed

from demo.globals.vehicles import vehicle_list

class VehicleExecutor:
    """
    使用线程池模拟车辆并行执行任务
    """
    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_tasks(
            self,
            best_vehicle_id_set: set[str],
            method_name: str,
            args: tuple = None,
            return_results: bool = False
    ) -> list[object] | None:
        """
        执行车辆对象的指定方法

        :param best_vehicle_id_set: 最佳车辆 id 集合
        :param method_name: 要执行的方法名称
        :param args: 车辆方法的参数元组，默认为 None 表示无参数
        :param return_results: 是否返回结果，默认为 False
        :return: 如果 return_results=True，返回结果列表；否则返回 None
        """
        if args is None:
            args = ()

        futures = []
        for vehicle in vehicle_list:
            if vehicle.car_id not in best_vehicle_id_set:
                continue
            method = getattr(vehicle, method_name)
            future = self.executor.submit(method, *args)
            futures.append(future)

        # 等待所有任务完成
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                if return_results:
                    results.append(result)
            except Exception as e:
                print(f"任务执行失败: {str(e)}")
                if return_results:
                    results.append(None)

        return results if return_results else None