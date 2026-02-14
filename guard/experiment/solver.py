import os
import csv

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from guard.agent.executor import root_analyze_info, get_camera_report, get_monitor_report, monitors
from guard.agent.planner import Planner
from guard.agent.verifier import verify
from guard.common.model import RootAnalyzeReport, RootAnalyzeData
from guard.common.prompt import ablation_monitor_sys_prompt, ablation_camera_sys_prompt, ablation_random_sys_prompt


class ExperimentSolver:
    """
    实验代码
    """
    def __init__(self, planner: Planner, experiment_name: str):
        """
        初始化
        :param planner: 自定义规划器
        :param experiment_name: 实验名称，用于保存结果
        """
        self.planner: Planner = planner
        self.experiment_name: str = experiment_name
        self.data: list[RootAnalyzeData] = root_analyze_info[planner.type_name]

    def _process_single_task(self, idx: int) -> tuple[int, RootAnalyzeReport]:
        """
        处理单个任务
        :param idx: 样例索引
        :return: 索引，报告
        """
        result, step = self.planner.run_with_step(
            task_uuid=f"uuid-{idx}",
            user_prompt=self.data[idx].user_prompt,
            type_id=self.data[idx].id
        )
        return idx, RootAnalyzeReport(
            type_name=self.planner.type_name,
            id=self.data[idx].id,
            response=result,
            step=step,
            score=0.0
        )

    def _planner_execute(self, start_id: int) -> list[RootAnalyzeReport]:
        """
        执行规划器，串行执行
        :param start_id: 样例起始 id
        :return: 根因分析报告列表
        """
        start_idx = start_id - 1
        reports = []
        for i in tqdm(range(start_idx, len(self.data)), desc='planner_execute'):
            _, report = self._process_single_task(i)
            reports.append(report)
        return reports

    def _planner_execute_multi(self, start_id: int, max_workers: int = 5) -> list[RootAnalyzeReport]:
        """
        执行规划器（线程池并行执行）
        :param start_id: 样例起始 id
        :param max_workers: 线程池最大工作线程数，默认 5
        :return: 根因分析报告列表
        """
        start_idx = start_id - 1
        # 收集所有需要处理的索引
        indices = list(range(start_idx, len(self.data)))

        # 使用线程池并行执行
        results: dict[int, RootAnalyzeReport] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(self._process_single_task, i): i for i in indices}

            # 收集结果
            for future in tqdm(as_completed(future_to_idx), total=len(indices), desc='planner_execute_multi'):
                idx, report = future.result()
                results[idx] = report

        # 按原始顺序返回报告
        reports = [results[i] for i in indices]
        return reports

    def _report_verify(self, reports: list[RootAnalyzeReport]) -> None:
        """
        验证报告
        :param reports: 根因分析报告列表
        :return: 将报告的得分字段进行赋值
        """
        # 这里因为大模型打分很快，就直接串行执行了:D
        for report in tqdm(reports, desc='report_verify'):
            data_idx = report.id - 1
            # 拿到对应的根因
            root_cause = self.data[data_idx].root_cause

            # 验证报告
            score = verify(report=report.response, answer=root_cause)
            report.score = score

    def _save_report_as_csv(self, reports: list[RootAnalyzeReport]) -> None:
        """
        将报告保存到 csv 文件
        :param reports: 报告对象列表
        :return: 默认保存到 本目录 / results / experiment_name / type_name.csv
        """
        # 构建目录路径
        dir_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results",
            self.experiment_name
        )

        # 创建目录（如果不存在）
        os.makedirs(dir_path, exist_ok=True)

        # 构建文件路径
        file_path = os.path.join(dir_path, f"{self.planner.type_name}.csv")

        # 定义 CSV 表头
        fieldnames = ["type_name", "id", "response", "step", "score"]

        # 检查文件是否存在以确定是否写入表头
        file_exists = os.path.exists(file_path)

        # 写入报告到 CSV
        with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writeheader()

            # 写入每条报告记录
            for report in reports:
                writer.writerow({
                    "type_name": report.type_name,
                    "id": report.id,
                    "response": report.response,
                    "step": str(report.step),  # 将步骤列表转换为字符串
                    "score": str(report.score)
                })

    def solve(self, start_id: int = 1, max_workers: int = 5, is_multi: bool = True) -> None:
        """
        处理实验
        :param start_id: 样例起始 id
        :param max_workers: 线程池最大工作线程数，默认 5
        :param is_multi: 是否使用多线程执行，默认 True
        :return: 无
        """
        # 1. 执行规划器
        if is_multi:
            reports = self._planner_execute_multi(start_id=start_id, max_workers=max_workers)
        else:
            reports = self._planner_execute(start_id=start_id)

        # 2. 验证报告，拿到得分
        self._report_verify(reports=reports)

        # 3. 保存报告到 csv 文件
        self._save_report_as_csv(reports=reports)

class CityGuardSolver(ExperimentSolver):
    """CityGuard 实验代码"""
    def __init__(self, type_name: str):
        super().__init__(
            planner=Planner(type_name=type_name),
            experiment_name="cityguard"
        )

class MotivationSolver(ExperimentSolver):
    """Motivation 实验代码"""
    def __init__(self, type_name: str):
        super().__init__(
            planner=Planner(type_name=type_name),
            experiment_name="motivation"
        )

class AblationMonitorSolver(ExperimentSolver):
    """消融实验，去掉 Monitor"""
    def __init__(self, type_name: str):
        super().__init__(
            planner=Planner(
                type_name=type_name,
                tools=[get_camera_report],
                system_prompt=ablation_monitor_sys_prompt.format()
            ),
            experiment_name="ablation_monitor"
        )

class AblationCameraSolver(ExperimentSolver):
    """消融实验，去掉 Camera"""
    def __init__(self, type_name: str):
        super().__init__(
            planner=Planner(
                type_name=type_name,
                tools=[get_monitor_report],
                system_prompt=ablation_camera_sys_prompt.format(monitor_info=monitors)
            ),
            experiment_name="ablation_camera"
        )

class AblationRandomSolver(ExperimentSolver):
    """消融实验，Random 策略"""
    def __init__(self, type_name: str):
        super().__init__(
            planner=Planner(
                type_name=type_name,
                system_prompt=ablation_random_sys_prompt.format(monitor_info=monitors)
            ),
            experiment_name="ablation_random"
        )

if __name__ == '__main__':
    solver = AblationMonitorSolver(type_name='garbage')
    solver.solve(start_id=1, max_workers=5, is_multi=True)