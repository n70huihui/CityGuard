"""
仅作测试用
"""
import os
import csv

from guard.agent.executor import root_analyze_info
from guard.agent.verifier import verify


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "new_verify")

# METHOD_NAMES = ["baseline", "counterfactual_only", "delayed_decision_only", "cityguard"]
METHOD_NAMES = ["cityguard"]
TYPE_NAMES = ["accident", "garbage", "noise", "water"]


def re_verify_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for method_name in METHOD_NAMES:
        print(f"\n===== 正在重新打分: {method_name} =====")
        method_dir = os.path.join(RESULTS_DIR, method_name)

        # key 为行索引 (0-9), value 为 {type_name: (response, root_cause)}
        row_data: dict[int, dict[str, tuple[str, str]]] = {}

        for type_name in TYPE_NAMES:
            csv_path = os.path.join(method_dir, f"{type_name}.csv")
            if not os.path.exists(csv_path):
                print(f"  警告: {csv_path} 不存在，跳过")
                continue

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row["id"]) - 1
                    response = row["response"]
                    root_cause = root_analyze_info[type_name][idx].root_cause

                    if idx not in row_data:
                        row_data[idx] = {}
                    row_data[idx][type_name] = (response, root_cause)

        # 对每一行（每个样例）依次打分
        rows_result = []
        for idx in sorted(row_data.keys()):
            result_row = {}
            for type_name in TYPE_NAMES:
                if type_name in row_data[idx]:
                    response, root_cause = row_data[idx][type_name]
                    print(f"  [{method_name}] {type_name} # {idx + 1} ...", end=" ")
                    score = verify(report=response, answer=root_cause)
                    print(f"{score:.1f}")
                    result_row[type_name] = score
                else:
                    result_row[type_name] = ""
            rows_result.append(result_row)

        # 写入 CSV
        out_path = os.path.join(OUTPUT_DIR, f"{method_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TYPE_NAMES)
            writer.writeheader()
            writer.writerows(rows_result)
        print(f"  已保存: {out_path}")


if __name__ == '__main__':
    re_verify_all()
