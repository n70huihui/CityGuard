import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------- 路径配置 ----------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "visual", "score_comparison")

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
def load_scores(csv_path: str) -> list[float]:
    """从 CSV 文件中读取 score 列"""
    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["score"]))
    return scores


def build_data() -> dict:
    """
    构建数据字典，结构为:
    {
        type_name: {
            method_name: [score, score, ...]
        }
    }
    """
    data = {}
    for tn in TYPE_NAMES:
        data[tn] = {}
        for mname, mdir in METHODS.items():
            csv_path = os.path.join(mdir, f"{tn}.csv")
            if os.path.exists(csv_path):
                data[tn][mname] = load_scores(csv_path)
    return data


# ---------- 绘图：平均分柱状图 ----------
def plot_mean_scores(data: dict):
    """绘制各事件类型下四种方法的平均得分柱状图"""
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
        stds = []
        for tn in TYPE_NAMES:
            scores = data[tn].get(mname, [])
            means.append(np.mean(scores) if scores else 0)
            stds.append(np.std(scores) if scores else 0)
        offset = (i - len(method_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=mname,
                      color=COLORS[i], alpha=0.75, edgecolor="white",
                      capsize=4, error_kw={"linewidth": 1})
        # 在柱子顶部标注均值
        for bar, m in zip(bars, means):
            ax.annotate(
                f"{m:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
            )

    ax.set_xlabel("Event Type", fontsize=13)
    ax.set_ylabel("Mean Score", fontsize=13)
    ax.set_title("Mean Score Comparison Across Methods by Event Type",
                 fontsize=16, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[tn] for tn in TYPE_NAMES], fontsize=12)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=12, loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "mean_scores.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[平均分柱状图] 已保存: {out_path}")


# ---------- 绘图：箱线图 ----------
def plot_box_scores(data: dict):
    """绘制各事件类型下四种方法的得分分布箱线图（2x2 子图）"""
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
        scores_list = [data[tn].get(m, []) for m in method_names]
        means = [np.mean(s) if s else 0 for s in scores_list]

        # 箱线图
        bp = ax.boxplot(
            scores_list,
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

        # 为每个箱体填充颜色
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor(color)

        # 在箱线图上方标注均值
        for i, (m, s) in enumerate(zip(means, scores_list)):
            ax.annotate(
                f"{m:.2f}",
                xy=(i + 1, max(s) if s else m),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold",
            )

        ax.set_title(TYPE_LABELS[tn], fontsize=16, fontweight="bold", pad=10)
        ax.set_ylabel("Score", fontsize=13)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelsize=11, rotation=0)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 底部共享图例
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.65, edgecolor=c)
               for c in COLORS]
    fig.legend(
        handles, method_names,
        loc="lower center", ncol=n_methods, fontsize=13,
        frameon=False, bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle("Score Distribution (Box Plot) by Event Type",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "score_boxplot.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[箱线图] 已保存: {out_path}")


# ---------- 主入口 ----------
if __name__ == "__main__":
    data = build_data()

    # 打印统计摘要
    for tn in TYPE_NAMES:
        print(f"\n--- {tn} ---")
        for mname, scores in data[tn].items():
            mean_s = np.mean(scores)
            std_s = np.std(scores)
            print(f"  {mname}: mean={mean_s:.2f}, std={std_s:.2f}, n={len(scores)}")

    # 分别输出两张图
    plot_mean_scores(data)
    plot_box_scores(data)
