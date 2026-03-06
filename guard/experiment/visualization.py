import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 使用 __file__ 获取当前脚本所在目录，动态计算路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # guard/experiment 的父目录是 guard
BASE_PATH = os.path.join(PROJECT_ROOT, "experiment", "results")

# 设置绘图风格和字体（论文常用字体）
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'  # 论文常用字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_experiment_data(base_path):
    """加载实验数据并合并"""
    experiment_groups = {
        "Full Workflow": os.path.join(base_path, "cityguard"),
        "Ablation: Camera": os.path.join(base_path, "ablation_camera"),
        "Ablation: Monitor": os.path.join(base_path, "ablation_monitor")
    }

    all_data = []
    for exp_name, exp_path in experiment_groups.items():
        for file in os.listdir(exp_path):
            if file.endswith(".csv"):
                file_path = os.path.join(exp_path, file)
                df = pd.read_csv(file_path)
                df['Experiment'] = exp_name  # 添加实验组标签
                all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['Response Length'] = combined_df['response'].apply(len)
    combined_df.dropna(subset=['score', 'step'], inplace=True)
    return combined_df


def load_category_data(base_path):
    """加载分类别实验数据"""
    experiment_groups = {
        "Full Workflow": os.path.join(base_path, "cityguard"),
        "Ablation: Camera": os.path.join(base_path, "ablation_camera"),
        "Ablation: Monitor": os.path.join(base_path, "ablation_monitor")
    }

    categories = ['accident', 'garbage', 'noise', 'water']
    all_data = []

    for exp_name, exp_path in experiment_groups.items():
        for cat in categories:
            file_path = os.path.join(exp_path, f"{cat}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Experiment'] = exp_name
                df['Category'] = cat.capitalize()
                all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['Response Length'] = combined_df['response'].apply(len)
    combined_df.dropna(subset=['score', 'step'], inplace=True)
    return combined_df


def plot_score_distribution(df, output_path):
    """绘制得分分布箱线图"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Experiment', y='score', data=df, palette="Set2")
    plt.title('Score Distribution Comparison Across Experiments', fontsize=14)
    plt.xlabel('Experiment Group', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'score_distribution.png'), dpi=300)
    plt.close()


def plot_average_scores(df, output_path):
    """绘制平均得分条形图"""
    plt.figure(figsize=(10, 6))
    mean_scores = df.groupby('Experiment')['score'].mean()
    std_scores = df.groupby('Experiment')['score'].std()

    ax = mean_scores.plot(kind='bar', yerr=std_scores, capsize=5, color=['#4C72B0', '#55A868', '#C44E52'])
    plt.title('Average Scores Across Experiment Groups', fontsize=14)
    plt.xlabel('Experiment Group', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(rotation=0)

    for i, v in enumerate(mean_scores):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'average_scores.png'), dpi=300)
    plt.close()


def plot_step_distribution(df, output_path):
    """绘制推理步数分布直方图"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='step', hue='Experiment',
                 element='step', stat='density', common_norm=False,
                 palette="Set2", alpha=0.5)
    plt.title('Distribution of Reasoning Steps', fontsize=14)
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'step_distribution.png'), dpi=300)
    plt.close()


def plot_step_score_relationship(df, output_path):
    """绘制步数与得分关系散点图"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='step', y='score', hue='Experiment',
                    palette="Set2", alpha=0.7, s=80)
    plt.title('Relationship Between Reasoning Steps and Scores', fontsize=14)
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'step_score_relationship.png'), dpi=300)
    plt.close()


def plot_response_length(df, output_path):
    """绘制回复长度分布密度图"""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Response Length', hue='Experiment',
                palette="Set2", common_norm=False, fill=True, alpha=0.3)
    plt.title('Distribution of Response Lengths', fontsize=14)
    plt.xlabel('Response Length (characters)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'response_length.png'), dpi=300)
    plt.close()


def statistical_test(df):
    """执行统计显著性检验"""
    full_data = df[df['Experiment'] == 'Full Workflow']['score']
    camera_ablation = df[df['Experiment'] == 'Ablation: Camera']['score']
    monitor_ablation = df[df['Experiment'] == 'Ablation: Monitor']['score']

    camera_t, camera_p = stats.ttest_ind(full_data, camera_ablation, equal_var=False)
    monitor_t, monitor_p = stats.ttest_ind(full_data, monitor_ablation, equal_var=False)

    return (camera_t, camera_p), (monitor_t, monitor_p)


def save_statistical_report(camera_result, monitor_result, output_path):
    """保存统计检验报告"""
    camera_t, camera_p = camera_result
    monitor_t, monitor_p = monitor_result

    with open(os.path.join(output_path, "statistical_report.txt"), "w") as f:
        f.write("实验组间得分统计显著性检验报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"完整工作流 vs 摄像头消融:\n")
        f.write(f"  t统计量 = {camera_t:.3f}, p值 = {camera_p:.4f}\n")
        f.write(f"  显著性水平: {'显著' if camera_p < 0.05 else '不显著'}\n\n")
        f.write(f"完整工作流 vs 监控消融:\n")
        f.write(f"  t统计量 = {monitor_t:.3f}, p值 = {monitor_p:.4f}\n")
        f.write(f"  显著性水平: {'显著' if monitor_p < 0.05 else '不显著'}\n")
        f.write("=" * 50 + "\n")
        f.write("注: p < 0.05 表示统计显著（95%置信水平）")

    print("=" * 50)
    print("统计显著性检验结果:")
    print(f"完整工作流 vs 摄像头消融: t = {camera_t:.3f}, p = {camera_p:.4f}")
    print(f"完整工作流 vs 监控消融: t = {monitor_t:.3f}, p = {monitor_p:.4f}")
    print("=" * 50)


def aggregate_analysis():
    """总体性数据分析：所有样例汇总后的统计分析"""
    output_path = os.path.join(BASE_PATH, "visual", "analysis_plots")
    os.makedirs(output_path, exist_ok=True)

    print("=" * 50)
    print("开始总体性数据分析...")
    print(f"输出目录: {output_path}")

    # 加载数据
    df = load_experiment_data(BASE_PATH)
    print(f"加载数据量: {len(df)} 条记录")

    # 绘制可视化图表
    plot_score_distribution(df, output_path)
    plot_average_scores(df, output_path)
    plot_step_distribution(df, output_path)
    plot_step_score_relationship(df, output_path)
    plot_response_length(df, output_path)

    # 统计检验
    camera_result, monitor_result = statistical_test(df)
    save_statistical_report(camera_result, monitor_result, output_path)

    print(f"总体性数据分析完成！图表已保存到 {output_path}")


def per_category_analysis():
    """细粒度数据分析：按类别（accident, garbage, noise, water）分别对比"""
    output_path = os.path.join(BASE_PATH, "visual", "category_plots")
    os.makedirs(output_path, exist_ok=True)

    print("=" * 50)
    print("开始细粒度分类别数据分析...")
    print(f"输出目录: {output_path}")

    # 加载分类别数据
    df = load_category_data(BASE_PATH)
    categories = ['Accident', 'Garbage', 'Noise', 'Water']
    print(f"加载数据量: {len(df)} 条记录")

    # 创建 2x2 子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = ['#4C72B0', '#55A868', '#C44E52']  # Full Workflow, Ablation Camera, Ablation Monitor

    for idx, category in enumerate(categories):
        cat_df = df[df['Category'] == category]

        # 计算各类别的统计数据
        mean_scores = cat_df.groupby('Experiment')['score'].mean().reindex(
            ['Full Workflow', 'Ablation: Camera', 'Ablation: Monitor']
        )
        std_scores = cat_df.groupby('Experiment')['score'].std().reindex(
            ['Full Workflow', 'Ablation: Camera', 'Ablation: Monitor']
        )

        # 绘制条形图
        ax = axes[idx]
        bars = ax.bar(range(3), mean_scores.values, yerr=std_scores.values,
                      capsize=5, color=colors, alpha=0.8)

        ax.set_xticks(range(3))
        ax.set_xticklabels(['Full\nWorkflow', 'Ablation:\nCamera', 'Ablation:\nMonitor'], fontsize=9)
        ax.set_title(f'{category}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0, 1.1)

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, mean_scores.values)):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', fontsize=10)

    plt.suptitle('Score Comparison Across Categories', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_path, 'category_comparison.png'), dpi=300)
    plt.close()

    # 绘制各类别箱线图对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, category in enumerate(categories):
        cat_df = df[df['Category'] == category]
        ax = axes[idx]
        sns.boxplot(x='Experiment', y='score', data=cat_df, palette="Set2", ax=ax)
        ax.set_title(f'{category}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=11)
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Score Distribution by Category', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_path, 'category_boxplot.png'), dpi=300)
    plt.close()

    # 各类别的统计检验
    print("\n各类别统计显著性检验:")
    with open(os.path.join(output_path, "category_statistical_report.txt"), "w") as f:
        f.write("分类别得分统计显著性检验报告\n")
        f.write("=" * 50 + "\n\n")

        for category in categories:
            cat_df = df[df['Category'] == category]
            full = cat_df[cat_df['Experiment'] == 'Full Workflow']['score']
            camera = cat_df[cat_df['Experiment'] == 'Ablation: Camera']['score']
            monitor = cat_df[cat_df['Experiment'] == 'Ablation: Monitor']['score']

            if len(full) > 0 and len(camera) > 0 and len(monitor) > 0:
                camera_t, camera_p = stats.ttest_ind(full, camera, equal_var=False)
                monitor_t, monitor_p = stats.ttest_ind(full, monitor, equal_var=False)

                print(f"{category}:")
                print(f"  Full vs Camera: t={camera_t:.3f}, p={camera_p:.4f} {'*' if camera_p < 0.05 else ''}")
                print(f"  Full vs Monitor: t={monitor_t:.3f}, p={monitor_p:.4f} {'*' if monitor_p < 0.05 else ''}")

                f.write(f"{category}:\n")
                f.write(f"  完整工作流 vs 摄像头消融: t={camera_t:.3f}, p={camera_p:.4f}, "
                       f"{'显著' if camera_p < 0.05 else '不显著'}\n")
                f.write(f"  完整工作流 vs 监控消融: t={monitor_t:.3f}, p={monitor_p:.4f}, "
                       f"{'显著' if monitor_p < 0.05 else '不显著'}\n\n")

        f.write("=" * 50 + "\n")
        f.write("注: * 表示 p < 0.05，统计显著")

    print(f"\n细粒度数据分析完成！图表已保存到 {output_path}")


if __name__ == "__main__":
    # 执行两种分析
    aggregate_analysis()
    print("\n")
    per_category_analysis()
