files = [
    ('bike_illegal_parking', 'multi', 'results/evaluation/bike_illegal_parking/multi_vehicle_bike_illegal_parking_evaluation.csv'),
    ('bike_illegal_parking', 'single', 'results/evaluation/bike_illegal_parking/single_vehicle_bike_illegal_parking_evaluation.csv'),
    ('garbage', 'multi', 'results/evaluation/garbage/multi_vehicle_evaluation.csv'),
    ('garbage', 'single', 'results/evaluation/garbage/single_vehicle_evaluation.csv'),
    ('illegal_parking', 'multi', 'results/evaluation/illegal_parking/multi_vehicle_illegal_parking_evaluation.csv'),
    ('illegal_parking', 'single', 'results/evaluation/illegal_parking/single_vehicle_illegal_parking_evaluation.csv'),
    ('waste_incineration', 'multi', 'results/evaluation/waste_incineration/multi_vehicle_waste_incineration_evaluation.csv'),
    ('waste_incineration', 'single', 'results/evaluation/waste_incineration/single_vehicle_waste_incineration_evaluation.csv')
]

import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置项目根目录
BASE_DIR = 'd:/Pycharm/pycharm/Pyspace/CityGuard'

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
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('single_view vs multi_view', fontsize=16)

# 违规类型列表
violation_types = ['bike_illegal_parking', 'garbage',
                   'illegal_parking', 'waste_incineration']

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
