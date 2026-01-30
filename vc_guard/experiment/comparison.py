from vc_guard.observations.handlers import get_project_root
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def single_view_vs_multi_view():
    files = [
        ('bike_illegal_parking', 'multi', 'results/evaluation/bike_illegal_parking/multi_vehicle_bike_illegal_parking_evaluation.csv'),
        ('bike_illegal_parking', 'single', 'results/evaluation/bike_illegal_parking/single_vehicle_bike_illegal_parking_evaluation.csv'),
        ('garbage', 'multi', 'results/evaluation/garbage/multi_vehicle_evaluation.csv'),
        ('garbage', 'single', 'results/evaluation/garbage/single_vehicle_evaluation.csv'),
        ('illegal_parking', 'multi', 'results/evaluation/illegal_parking/multi_vehicle_illegal_parking_evaluation.csv'),
        ('illegal_parking', 'single', 'results/evaluation/illegal_parking/single_vehicle_illegal_parking_evaluation.csv'),
        ('waste_incineration', 'multi', 'results/evaluation/waste_incineration/multi_vehicle_waste_incineration_evaluation.csv'),
        ('waste_incineration', 'single', 'results/evaluation/waste_incineration/single_vehicle_waste_incineration_evaluation.csv'),
        ('fallen_leaves_and_accumulated_water', 'single', 'results/evaluation/fallen_leaves_and_accumulated_water/single_vehicle_fallen_leaves_and_accumulated_water_evaluation.csv'),
        ('fallen_leaves_and_accumulated_water', 'multi', 'results/evaluation/fallen_leaves_and_accumulated_water/multi_vehicle_fallen_leaves_and_accumulated_water_evaluation.csv'),
        ('road_occupation_for_business_and_construction', 'single', 'results/evaluation/road_occupation_for_business_and_construction/single_vehicle_road_occupation_for_business_and_construction_evaluation.csv'),
        ('road_occupation_for_business_and_construction', 'multi', 'results/evaluation/road_occupation_for_business_and_construction/multi_vehicle_road_occupation_for_business_and_construction_evaluation.csv'),
    ]



    # 设置项目根目录
    BASE_DIR = get_project_root()

    # 1. 读取并合并所有数据
    all_data = []
    for violation_type, perspective, rel_path in files:
        file_path = os.path.join(BASE_DIR, rel_path)
        try:
            df = pd.read_csv(file_path)
            df['perspective'] = perspective
            df['violation_type'] = violation_type
            all_data.append(df)
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")

    combined_df = pd.concat(all_data)

    # 2. 创建4个子图
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('single_view vs multi_view', fontsize=16)

    # 违规类型列表
    violation_types = ['bike_illegal_parking', 'garbage',
                       'illegal_parking', 'waste_incineration',
                       'fallen_leaves_and_accumulated_water', 'road_occupation_for_business_and_construction']

    # 3. 绘制每个违规类型的箱线图
    for i, v_type in enumerate(violation_types):
        ax = axs[i // 2, i % 2]

        # 筛选当前类型的数据
        type_data = combined_df[combined_df['violation_type'] == v_type]

        # 准备多视角和单视角数据
        multi = type_data[type_data['perspective'] == 'multi']['score']
        single = type_data[type_data['perspective'] == 'single']['score']

        # 绘制箱线图
        bp = ax.boxplot([single, multi],
                        patch_artist=True,
                        labels=['single_view', 'multi_view'])

        # 设置颜色
        colors = ['#1f77b4', '#ff7f0e']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # 设置标题和标签
        ax.set_title(f'{v_type.replace("_", " ").title()}')
        ax.set_ylabel('score')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 11)  # 统一Y轴范围

    # 4. 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留空间
    plt.savefig('perspective_comparison.png', dpi=300)
    plt.show()


def inactive_vs_active():

    BASE_DIR = get_project_root()

    metrics = [
        ('low_score_ratio', 'Low Score Ratio (<3)', lambda s: (s < 3).mean()),
        ('high_score_ratio', 'High Score Ratio (>=7)', lambda s: (s >= 7).mean()),
        ('mean_score', 'Mean Score', np.mean)
    ]

    active_files = [
        ('Bike Parking', 'results/evaluation/active/active_bike_illegal_parking_evaluation.csv'),
        ('Fallen Leaves', 'results/evaluation/active/active_fallen_leaves_and_accumulated_water_evaluation.csv'),
        ('Garbage', 'results/evaluation/active/active_garbage_evaluation.csv'),
        ('Illegal Parking', 'results/evaluation/active/active_illegal_parking_evaluation.csv'),
        ('Road Occupation',
         'results/evaluation/active/active_road_occupation_for_business_and_construction_evaluation.csv'),
        ('Waste Incineration', 'results/evaluation/active/active_waste_incineration_evaluation.csv')
    ]

    results = []
    for violation_type, file_path in active_files:
        df = pd.read_csv(os.path.join(BASE_DIR, file_path))
        for col in ['score1', 'score2']:
            for metric_id, _, func in metrics:
                value = func(df[col])
                results.append({
                    'violation_type': violation_type,
                    'score_type': col,
                    'metric': metric_id,
                    'value': value
                })

    result_df = pd.DataFrame(results)

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Impact of Active Scheduling Module', fontsize=16)

    for ax, (metric_id, metric_name, _) in zip(axes, metrics):
        plot_data = result_df[result_df['metric'] == metric_id]
        pivot_data = plot_data.pivot(
            index='violation_type',
            columns='score_type',
            values='value'
        ).reset_index()

        pivot_data['improvement'] = pivot_data['score2'] - pivot_data['score1']

        x = np.arange(len(pivot_data))
        width = 0.35

        rects1 = ax.bar(x - width / 2, pivot_data['score1'], width,
                        label='Without Scheduling', color='#1f77b4')
        rects2 = ax.bar(x + width / 2, pivot_data['score2'], width,
                        label='With Scheduling', color='#ff7f0e')

        for i, (_, row) in enumerate(pivot_data.iterrows()):
            ax.text(i, max(row['score1'], row['score2']) + 0.05,
                    f"+{row['improvement']:.2f}",
                    ha='center', fontsize=9)

        ax.set_title(metric_name)
        ax.set_ylabel('Ratio' if 'ratio' in metric_id else 'Score')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_data['violation_type'])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('active_scheduling_impact.png', dpi=300)
    plt.show()


def random_vs_quadrant():
    """
    比较随机调车与象限调车的性能差异
    使用两个关键指标展示象限调车的优越性：
    1. 平均得分对比
    2. 高优化率（提升超过2分）的比例
    """
    BASE_DIR = get_project_root()

    # 定义文件路径
    files = [
        ('Fallen Leaves', 'results/evaluation/quadrant/fallen_leaves_and_accumulated_water_evaluation.csv'),
        ('Bike Parking', 'results/evaluation/quadrant/random_bike_illegal_parking_evaluation.csv'),
        ('Garbage', 'results/evaluation/quadrant/random_garbage_evaluation.csv'),
        ('Illegal Parking', 'results/evaluation/quadrant/random_illegal_parking_evaluation.csv'),
        ('Road Occupation',
         'results/evaluation/quadrant/random_road_occupation_for_business_and_construction_evaluation.csv'),
        ('Waste Incineration', 'results/evaluation/quadrant/random_waste_incineration_evaluation.csv')
    ]

    # 存储每个违规类型的结果
    results = []

    for violation_type, rel_path in files:
        file_path = os.path.join(BASE_DIR, rel_path)
        try:
            df = pd.read_csv(file_path)
            # 计算关键指标
            improvement = df['score2'] - df['score1']

            # 高优化率 (improvement > 2)
            high_improvement_ratio = (improvement > 2).mean()

            # 存储结果
            results.append({
                'violation_type': violation_type,
                'mean_random': df['score1'].mean(),
                'mean_quadrant': df['score2'].mean(),
                'mean_improvement': improvement.mean(),
                'high_improvement_ratio': high_improvement_ratio
            })

        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except KeyError:
            print(f"Columns 'score1' or 'score2' not found in {file_path}")

    if not results:
        print("No valid data found")
        return

    result_df = pd.DataFrame(results)

    # 创建图表（1行2列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Random vs Quadrant Scheduling Performance Comparison', fontsize=16)

    # 子图1: 平均得分对比
    x = np.arange(len(result_df))
    width = 0.35

    # 绘制随机调度和象限调度的平均得分柱状图
    ax1.bar(x - width / 2, result_df['mean_random'], width,
            label='Random Scheduling', color='#1f77b4', alpha=0.8)
    ax1.bar(x + width / 2, result_df['mean_quadrant'], width,
            label='Quadrant Scheduling', color='#ff7f0e', alpha=0.8)

    # 添加优化值标签
    for i, row in result_df.iterrows():
        y_pos = max(row['mean_random'], row['mean_quadrant']) + 0.1
        ax1.text(i, y_pos, f"+{row['mean_improvement']:.2f}",
                 ha='center', fontsize=10, fontweight='bold')

    # 设置图表属性
    ax1.set_title('Average Score Comparison', fontsize=14)
    ax1.set_ylabel('Average Score', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(result_df['violation_type'], rotation=15, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, loc='best')

    # 添加水平参考线
    ax1.axhline(y=7.5, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=5.0, color='gray', linestyle='--', alpha=0.3)

    # 子图2: 高优化率对比
    # 绘制高优化率柱状图
    bars = ax2.bar(result_df['violation_type'],
                   result_df['high_improvement_ratio'] * 100,
                   color='#2ca02c', alpha=0.8)

    # 在每个柱子上添加百分比标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # 设置图表属性
    ax2.set_title('High Improvement Rate (>2 points)', fontsize=14)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_xticklabels(result_df['violation_type'], rotation=15, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # 添加水平参考线
    ax2.axhline(y=30, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留空间
    plt.savefig('quadrant_scheduling_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    random_vs_quadrant()