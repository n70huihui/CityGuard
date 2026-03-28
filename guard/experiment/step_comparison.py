import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------- 路径配置 ----------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "visual", "step_comparison")

METHODS = {
    "Baseline": os.path.join(RESULTS_DIR, "baseline"),
    "Counterfactual": os.path.join(RESULTS_DIR, "counterfactual_only"),
    "Delayed Decision": os.path.join(RESULTS_DIR, "delayed_decision_only"),
    "CityGuard": os.path.join(RESULTS_DIR, "cityguard"),
}

TYPE_NAMES = ["accident", "garbage", "noise", "water"]
TYPE_LABELS = {
    "accident": "Accident",
    "garbage": "Garbage",
    "noise": "Noise",
    "water": "Water",
}

COLORS = ["#5B8FF9", "#5AD8A6", "#F6BD16", "#E8684A"]


# ---------- 数据加载 ----------
def load_steps(csv_path: str) -> list[int]:
    """从 CSV 文件中读取 step 列"""
    steps = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
    return steps


def build_data() -> dict:
    """
    构建数据字典，结构为:
    {
        type_name: {
            method_name: [step, step, ...]
        }
    }
    """
    data = {}
    for tn in TYPE_NAMES:
        data[tn] = {}
        for mname, mdir in METHODS.items():
            csv_path = os.path.join(mdir, f"{tn}.csv")
            if os.path.exists(csv_path):
                data[tn][mname] = load_steps(csv_path)
    return data


# ---------- 绘图：平均步数柱状图 ----------
def plot_mean_steps(data: dict):
    """绘制各事件类型下四种方法的平均推理步数柱状图"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    })

    method_names = list(METHODS.keys())
    x = np.arange(len(TYPE_NAMES))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, mname in enumerate(method_names):
        means = []
        for tn in TYPE_NAMES:
            steps = data[tn].get(mname, [])
            means.append(np.mean(steps) if steps else 0)
        offset = (i - len(method_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=mname,
                      color=COLORS[i], alpha=0.75, edgecolor="white")
        for bar, m in zip(bars, means):
            ax.annotate(
                f"{m:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
            )

    ax.set_xlabel("Event Type", fontsize=13)
    ax.set_ylabel("Mean Steps", fontsize=13)
    ax.set_title("Mean Reasoning Steps Comparison Across Methods by Event Type",
                 fontsize=16, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[tn] for tn in TYPE_NAMES], fontsize=12)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "mean_steps.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[平均步数柱状图] 已保存: {out_path}")


# ---------- 绘图：标准差柱状图 ----------
def plot_std_steps(data: dict):
    """绘制各事件类型下四种方法的步数标准差对比柱状图"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    })

    method_names = list(METHODS.keys())
    x = np.arange(len(TYPE_NAMES))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, mname in enumerate(method_names):
        stds = []
        for tn in TYPE_NAMES:
            steps = data[tn].get(mname, [])
            stds.append(np.std(steps) if steps else 0)
        offset = (i - len(method_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, stds, width, label=mname,
                      color=COLORS[i], alpha=0.75, edgecolor="white")
        for bar, s in zip(bars, stds):
            ax.annotate(
                f"{s:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
            )

    ax.set_xlabel("Event Type", fontsize=13)
    ax.set_ylabel("Standard Deviation", fontsize=13)
    ax.set_title("Step Dispersion (Std Dev) Comparison Across Methods by Event Type",
                 fontsize=16, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[tn] for tn in TYPE_NAMES], fontsize=12)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=12, loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "std_steps.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[标准差柱状图] 已保存: {out_path}")


# ---------- 绘图：箱线图 ----------
def plot_box_steps(data: dict):
    """绘制各事件类型下四种方法的推理步数分布箱线图（2x2 子图）"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    })

    method_names = list(METHODS.keys())
    n_methods = len(method_names)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, tn in enumerate(TYPE_NAMES):
        ax = axes[idx]
        steps_list = [data[tn].get(m, []) for m in method_names]
        means = [np.mean(s) if s else 0 for s in steps_list]

        bp = ax.boxplot(
            steps_list,
            tick_labels=method_names,
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=7),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker="o", markersize=4, alpha=0.6),
        )

        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor(color)

        for i, (m, s) in enumerate(zip(means, steps_list)):
            ax.annotate(
                f"{m:.1f}",
                xy=(i + 1, max(s) if s else m),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold",
            )

        ax.set_title(TYPE_LABELS[tn], fontsize=16, fontweight="bold", pad=10)
        ax.set_ylabel("Steps", fontsize=13)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelsize=11, rotation=0)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.65, edgecolor=c)
               for c in COLORS]
    fig.legend(
        handles, method_names,
        loc="lower center", ncol=n_methods, fontsize=13,
        frameon=False, bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle("Reasoning Steps Distribution (Box Plot) by Event Type",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "step_boxplot.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[箱线图] 已保存: {out_path}")


# ---------- 绘图：小提琴图（全局步数分布） ----------
def plot_violin_steps(data: dict):
    """绘制全局推理步数的小提琴图（按方法分组，不区分事件类型）"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    })

    method_names = list(METHODS.keys())

    # 汇总所有事件类型的步数
    all_steps = []
    for mname in method_names:
        steps = []
        for tn in TYPE_NAMES:
            steps.extend(data[tn].get(mname, []))
        all_steps.append(steps)

    fig, ax = plt.subplots(figsize=(10, 6))

    parts = ax.violinplot(
        all_steps, positions=range(1, len(method_names) + 1),
        showmeans=True, showmedians=True, showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i])
        pc.set_alpha(0.65)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("gray")
    parts["cmeans"].set_linestyle("--")
    parts["cmedians"].set_linestyle("-")

    # 标注均值
    for i, steps in enumerate(all_steps):
        m = np.mean(steps)
        ax.annotate(f"{m:.1f}", xy=(i + 1, m), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(range(1, len(method_names) + 1))
    ax.set_xticklabels(method_names, fontsize=12)
    ax.set_ylabel("Steps", fontsize=13)
    ax.set_title("Reasoning Steps Distribution (Violin Plot) - All Event Types",
                 fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "violin_steps.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[小提琴图] 已保存: {out_path}")


# ---------- 绘图：核密度估计图（2x2 子图） ----------
def plot_kde_steps(data: dict):
    """绘制各事件类型下四种方法的推理步数核密度估计图（2x2 子图）"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    })

    method_names = list(METHODS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, tn in enumerate(TYPE_NAMES):
        ax = axes[idx]
        x_max = 0
        for i, mname in enumerate(method_names):
            steps = np.array(data[tn].get(mname, []))
            if len(steps) < 2:
                continue
            x_range = np.linspace(0, steps.max() + 5, 200)
            kde = gaussian_kde(steps, bw_method=0.5)
            y_vals = kde(x_range)
            ax.plot(x_range, y_vals, color=COLORS[i], linewidth=2, label=mname)
            ax.fill_between(x_range, y_vals, alpha=0.15, color=COLORS[i])
            x_max = max(x_max, steps.max() + 5)

        ax.set_title(TYPE_LABELS[tn], fontsize=16, fontweight="bold", pad=10)
        ax.set_xlabel("Steps", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_xlim(0, x_max)
        ax.tick_params(labelsize=11)
        ax.grid(linestyle="--", alpha=0.4)

    handles = [plt.Line2D([0], [0], color=c, linewidth=2) for c in COLORS]
    fig.legend(handles, method_names, loc="lower center", ncol=4,
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Reasoning Steps Density (KDE) by Event Type",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "kde_steps.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[核密度估计图] 已保存: {out_path}")


# ---------- 绘图：Step-Score 散点图 ----------
def plot_step_score_scatter(data: dict):
    """绘制步数-得分散点图，展示步数与得分的关系"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    })

    method_names = list(METHODS.keys())

    # 同时加载 scores
    score_data = {}
    for mname, mdir in METHODS.items():
        for tn in TYPE_NAMES:
            csv_path = os.path.join(mdir, f"{tn}.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            steps = [int(r["step"]) for r in rows]
            scores = [float(r["score"]) for r in rows]
            if mname not in score_data:
                score_data[mname] = {"steps": [], "scores": []}
            score_data[mname]["steps"].extend(steps)
            score_data[mname]["scores"].extend(scores)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, tn in enumerate(TYPE_NAMES):
        ax = axes[i]
        for j, mname in enumerate(method_names):
            csv_path = os.path.join(METHODS[mname], f"{tn}.csv")
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            steps = [int(r["step"]) for r in rows]
            scores = [float(r["score"]) for r in rows]
            ax.scatter(steps, scores, color=COLORS[j], alpha=0.7,
                       s=50, edgecolors="white", linewidth=0.5, label=mname)

        ax.set_title(TYPE_LABELS[tn], fontsize=16, fontweight="bold", pad=10)
        ax.set_xlabel("Steps", fontsize=13)
        ax.set_ylabel("Score", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(linestyle="--", alpha=0.4)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=8, label=m)
               for c, m in zip(COLORS, method_names)]
    fig.legend(handles, method_names, loc="lower center", ncol=4,
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Steps vs Score Scatter Plot by Event Type",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "step_score_scatter.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[步数-得分散点图] 已保存: {out_path}")


# ---------- 主入口 ----------
if __name__ == "__main__":
    data = build_data()

    for tn in TYPE_NAMES:
        print(f"\n--- {tn} ---")
        for mname, steps in data[tn].items():
            mean_s = np.mean(steps)
            std_s = np.std(steps)
            print(f"  {mname}: mean={mean_s:.1f}, std={std_s:.2f}, n={len(steps)}")

    plot_mean_steps(data)
    plot_std_steps(data)
    plot_box_steps(data)
    plot_violin_steps(data)
    plot_kde_steps(data)
    plot_step_score_scatter(data)
