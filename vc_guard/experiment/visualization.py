import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 确保输出目录存在
output_dir = '../../results/evaluation'
os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）

# 2. 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 3. 读取数据
single_df = pd.read_csv(
    '../../results/evaluation/single_and_multi_view/road_occupation_for_business_and_construction/single_vehicle_road_occupation_for_business_and_construction_evaluation.csv')
multi_df = pd.read_csv(
    '../../results/evaluation/single_and_multi_view/road_occupation_for_business_and_construction/multi_vehicle_road_occupation_for_business_and_construction_evaluation.csv')

# 4. 添加视角类型列
single_df['视角'] = '单视角'
multi_df['视角'] = '多视角'

# 5. 合并数据
combined_df = pd.concat([single_df, multi_df])

# 6. 创建图表
plt.figure(figsize=(14, 10))

# 7. 修复Seaborn警告（添加hue参数）
# 箱线图比较分布
plt.subplot(2, 2, 1)
sns.boxplot(x='视角', y='score', hue='视角', data=combined_df, palette='Set2', legend=False)
plt.title('得分分布比较')
plt.ylabel('得分')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 8. 修复Seaborn警告（添加hue参数）
# 小提琴图展示分布密度
plt.subplot(2, 2, 2)
sns.violinplot(x='视角', y='score', hue='视角', data=combined_df, palette='Set2',
               inner='quartile', legend=False)
plt.title('得分密度分布')
plt.ylabel('得分')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 9. 折线图比较每个样本的得分变化
plt.subplot(2, 1, 2)
# 创建合并数据用于折线图
comparison_df = single_df.merge(
    multi_df,
    on=['type_name', 'file_id'],
    suffixes=('_single', '_multi')
)
comparison_df = comparison_df.sort_values('file_id')

# 10. 绘制折线图
plt.plot(comparison_df['file_id'], comparison_df['score_single'],
         'o-', label='单视角', color='#1f77b4')
plt.plot(comparison_df['file_id'], comparison_df['score_multi'],
         's-', label='多视角', color='#ff7f0e')

# 11. 添加差异标记
for i, row in comparison_df.iterrows():
    diff = row['score_multi'] - row['score_single']
    color = 'green' if diff > 0 else 'red' if diff < 0 else 'gray'
    plt.annotate(f"{diff:+.1f}",
                 (row['file_id'], max(row['score_single'], row['score_multi']) + 0.3),
                 ha='center', color=color, fontsize=9)

plt.title('各样本得分对比')
plt.xlabel('样本ID')
plt.ylabel('得分')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(comparison_df['file_id'])

# 12. 添加统计信息表格
stats_table = pd.DataFrame({
    '视角': ['单视角', '多视角'],
    '平均值': [single_df['score'].mean(), multi_df['score'].mean()],
    '中位数': [single_df['score'].median(), multi_df['score'].median()],
    '标准差': [single_df['score'].std(), multi_df['score'].std()],
    '最小值': [single_df['score'].min(), multi_df['score'].min()],
    '最大值': [single_df['score'].max(), multi_df['score'].max()]
})

# 13. 在图表下方添加统计表格
plt.figtext(0.5, 0.01,
            stats_table.to_string(index=False),
            ha='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5))

# 14. 调整布局
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为表格留出空间

# 15. 添加总标题
plt.suptitle('单视角与多视角得分对比分析', fontsize=16)

# 16. 保存图表（使用完整路径）
output_path = os.path.join(output_dir, 'score_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至: {os.path.abspath(output_path)}")

# 17. 显示图表
plt.show()

# 18. 输出统计结果
print("\n统计摘要:")
print(stats_table)

# 19. 计算得分差异
comparison_df['差异'] = comparison_df['score_multi'] - comparison_df['score_single']
print("\n各样本得分差异:")
print(comparison_df[['type_name', 'file_id', 'score_single', 'score_multi', '差异']])