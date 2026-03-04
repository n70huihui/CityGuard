import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置绘图风格和字体
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'SimHei'  # 中文字体支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 数据加载与预处理
# ============================================================
# 定义实验组目录路径
base_path = "D:\Pycharm\pycharm\Pyspace\CityGuard\guard\experiment\\results"
experiment_groups = {
    "Full Workflow": os.path.join(base_path, "cityguard"),
    "Ablation: Camera": os.path.join(base_path, "ablation_camera"),
    "Ablation: Monitor": os.path.join(base_path, "ablation_monitor")
}

# 读取所有CSV文件并合并
all_data = []
for exp_name, exp_path in experiment_groups.items():
    # 遍历目录中的所有CSV文件
    for file in os.listdir(exp_path):
        if file.endswith(".csv"):
            file_path = os.path.join(exp_path, file)
            df = pd.read_csv(file_path)
            df['Experiment'] = exp_name  # 添加实验组标签
            all_data.append(df)

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)

# 数据预处理
# 添加回复长度特征
combined_df['Response Length'] = combined_df['response'].apply(len)
# 处理可能的缺失值
combined_df.dropna(subset=['score', 'step'], inplace=True)

# 2. 可视化分析
# ============================================================
# 创建图表保存目录
os.makedirs("results/visual/experiment/analysis_plots", exist_ok=True)

# 图1: 得分分布箱线图（实验组间比较）
plt.figure(figsize=(10, 6))
sns.boxplot(x='Experiment', y='score', data=combined_df, palette="Set2")
plt.title('Score Distribution Comparison Across Experiments', fontsize=14)
plt.xlabel('Experiment Group', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('visual/experiment/analysis_plots/score_distribution.png', dpi=300)
plt.close()

# 图2: 平均得分条形图（带误差线）
plt.figure(figsize=(10, 6))
mean_scores = combined_df.groupby('Experiment')['score'].mean()
std_scores = combined_df.groupby('Experiment')['score'].std()

ax = mean_scores.plot(kind='bar', yerr=std_scores, capsize=5, color=['#4C72B0', '#55A868', '#C44E52'])
plt.title('Average Scores Across Experiment Groups', fontsize=14)
plt.xlabel('Experiment Group', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.xticks(rotation=0)

# 添加数值标签
for i, v in enumerate(mean_scores):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('visual/experiment/analysis_plots/average_scores.png', dpi=300)
plt.close()

# 图3: 推理步数分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(data=combined_df, x='step', hue='Experiment',
             element='step', stat='density', common_norm=False,
             palette="Set2", alpha=0.5)
plt.title('Distribution of Reasoning Steps', fontsize=14)
plt.xlabel('Number of Steps', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.tight_layout()
plt.savefig('visual/experiment/analysis_plots/step_distribution.png', dpi=300)
plt.close()

# 图4: 得分与步数关系散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_df, x='step', y='score', hue='Experiment',
                palette="Set2", alpha=0.7, s=80)
plt.title('Relationship Between Reasoning Steps and Scores', fontsize=14)
plt.xlabel('Number of Steps', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.tight_layout()
plt.savefig('visual/experiment/analysis_plots/step_score_relationship.png', dpi=300)
plt.close()

# 图5: 模型回复长度分布
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, x='Response Length', hue='Experiment',
            palette="Set2", common_norm=False, fill=True, alpha=0.3)
plt.title('Distribution of Response Lengths', fontsize=14)
plt.xlabel('Response Length (characters)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.tight_layout()
plt.savefig('visual/experiment/analysis_plots/response_length.png', dpi=300)
plt.close()

# 3. 统计显著性检验
# ============================================================
# 准备数据
full_data = combined_df[combined_df['Experiment'] == 'Full Workflow']['score']
camera_ablation = combined_df[combined_df['Experiment'] == 'Ablation: Camera']['score']
monitor_ablation = combined_df[combined_df['Experiment'] == 'Ablation: Monitor']['score']

# 执行独立样本t检验
camera_t, camera_p = stats.ttest_ind(full_data, camera_ablation, equal_var=False)
monitor_t, monitor_p = stats.ttest_ind(full_data, monitor_ablation, equal_var=False)

# 打印统计结果
print("=" * 50)
print("统计显著性检验结果:")
print(f"完整工作流 vs 摄像头消融: t = {camera_t:.3f}, p = {camera_p:.4f}")
print(f"完整工作流 vs 监控消融: t = {monitor_t:.3f}, p = {monitor_p:.4f}")
print("=" * 50)

# 生成统计结果报告
with open("results/visual/experiment/analysis_plots/statistical_report.txt", "w") as f:
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

print("分析完成！所有图表和报告已保存到 visual/experiment/analysis_plots/ 目录")
