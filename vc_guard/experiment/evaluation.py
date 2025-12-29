import csv
import json
from pathlib import Path

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from env_utils.llm_args import *
from vc_guard.common.prompts import evaluate_summary_template


class BatchLLMJudge:
    """
    大模型评估（支持起始索引和类型名称）
    """
    def __init__(self,
                 input_csv: str,
                 output_csv: str,
                 batch_size: int = 5,
                 start_idx: int = 0):
        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv)

        self.batch_size = batch_size
        self.start_idx = start_idx

        self.fieldnames = ['type_name', 'file_id', 'score']

        self.type_name_answer = {
            "garbage": "路段周围垃圾违规堆放造成异味",
            "illegal_parking": "违规停车造成车辆拥堵"
            # TODO 其他情况
        }

        self._init_output_file()
        self.agent = create_agent(
            model=ChatOpenAI(base_url=base_url, api_key=api_key, model=model),
            tools=[]
        )

    def _init_output_file(self):
        """
        初始化输出文件并写入表头
        """
        # 只有当 start_idx 为 0 时才创建新文件并写入表头
        if self.start_idx == 0:
            with open(self.output_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def _read_batches(self):
        """
        分批读取输入 CSV 文件，支持跳过起始行
        """
        if not self.input_csv.exists():
            raise FileNotFoundError(f"输入文件不存在: {self.input_csv}")

        with open(self.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            batch = []
            current_idx = 0  # 当前行索引

            for row in reader:
                # 跳过起始索引之前的行
                if current_idx < self.start_idx:
                    current_idx += 1
                    continue

                batch.append(row)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

                current_idx += 1

            # 返回最后一批数据
            if batch:
                yield batch

    def _evaluate_summary(self, type_name: str, summary: str) -> float:
        """
        LLM as a judge
        :param type_name: 检测的场景类型
        :param summary: 需要评估的结果
        :return: 评估得分
        """

        # 构建提示词
        prompt = evaluate_summary_template.format(
            target_task=type_name,
            summary=summary
        )

        response = self.agent.invoke({"messages": [prompt]})

        # 解析输出
        content = response["messages"][-1].content_blocks
        score = content[0]['text']

        return score

    def _process_batch(self, batch: list[dict]):
        """
        处理单个批次的数据
        :param batch: 批次数据
        """
        evaluated_batch = []

        for row in batch:
            # 调用评估函数时传入type_name
            score = self._evaluate_summary(
                type_name=row['type_name'],
                summary=row.get('summary', '')
            )

            evaluated_batch.append({
                'type_name': row['type_name'],
                'file_id': row['file_id'],
                'score': score
            })

        return evaluated_batch

    def _save_batch(self, batch: list[dict]):
        """
        保存单个批次的评估结果
        """
        with open(self.output_csv, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(batch)

    def run(self):
        """
        执行评估流程
        """
        total_batches = 0
        total_rows = 0

        # 获取实际开始处理的行号
        actual_start_idx = self.start_idx
        if actual_start_idx != 0:
            print(f"开始处理，跳过前 {actual_start_idx} 行数据")

        for i, batch in enumerate(self._read_batches(), 1):
            evaluated_batch = self._process_batch(batch)
            self._save_batch(evaluated_batch)

            total_batches = i
            total_rows += len(batch)

            print(f"已处理批次 {i}，包含 {len(batch)} 行，累计 {total_rows} 行")

        print(f"评估完成！共处理 {total_batches} 个批次，{total_rows} 行数据")
        print(f"结果已保存至: {self.output_csv}")

        # 返回最后处理的行号，便于后续续传
        return actual_start_idx + total_rows


if __name__ == '__main__':
    input_csv = '../../results/executor/single_vehicle_output.csv'
    output_csv = 'single_vehicle_evaluation.csv'
    batch_size = 5

    # start_idx = 100

    evaluator = BatchLLMJudge(
        input_csv=input_csv,
        output_csv=output_csv,
        batch_size=batch_size,
        # start_idx=start_idx
    )

    last_processed_idx = evaluator.run()
    print(f"最后处理的行索引: {last_processed_idx}")