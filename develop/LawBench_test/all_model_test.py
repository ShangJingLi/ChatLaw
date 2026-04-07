import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================
# 0. Matplotlib 中文显示设置
# =========================
plt.rcParams["font.sans-serif"] = ["SimHei"]      # 黑体
plt.rcParams["axes.unicode_minus"] = False        # 解决负号显示问题


# =========================
# 1. 文件路径
# =========================
one_shot_path = os.path.join("develop", "resources", "lawbench_result", "results_one_shot.csv")
zero_shot_path = os.path.join("develop", "resources", "lawbench_result", "results_zero_shot.csv")
my_one_shot_path = os.path.join("develop", "resources", "lawbench_result", "qwen3_4b_rag_oneshot_results.csv")
my_zero_shot_path = os.path.join("develop", "resources", "lawbench_result", "qwen3_4b_rag_zeroshot_results.csv")


# =========================
# 2. 读取并统计所有模型平均分
# =========================
def prepare_all_model_scores(base_csv, my_csv):
    base_df = pd.read_csv(base_csv)
    my_df = pd.read_csv(my_csv)

    # 只保留必要列并合并
    df = pd.concat(
        [
            base_df[["task", "model_name", "score"]],
            my_df[["task", "model_name", "score"]],
        ],
        ignore_index=True
    )

    # 按模型统计平均分
    avg_df = (
        df.groupby("model_name", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "avg_score"})
    )

    # 转为百分制
    avg_df["avg_score_pct"] = avg_df["avg_score"] * 100

    # 按平均分降序排列
    avg_df = avg_df.sort_values("avg_score_pct", ascending=False).reset_index(drop=True)

    return avg_df


# =========================
# 3. 绘图函数
# =========================
def plot_all_model_scores(avg_df, title, ylabel, output_file, my_model_name="qwen3_4b_rag"):
    plt.figure(figsize=(20, 8))

    # 本文模型高亮，其他模型统一颜色
    colors = [
        "#7b3294" if model == my_model_name else "#5ab89a"
        for model in avg_df["model_name"]
    ]

    plt.bar(avg_df["model_name"], avg_df["avg_score_pct"], color=colors)

    plt.title(title, fontsize=20)
    plt.xlabel("模型名称", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    plt.xticks(rotation=75, ha="right", fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles = [
        Patch(color="#7b3294", label="本文模型"),
        Patch(color="#5ab89a", label="其他模型")
    ]
    plt.legend(handles=legend_handles, fontsize=13, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# 4. 统计所有模型
# =========================
one_shot_all_avg = prepare_all_model_scores(one_shot_path, my_one_shot_path)
zero_shot_all_avg = prepare_all_model_scores(zero_shot_path, my_zero_shot_path)

print("One-shot 测试中所有模型的平均得分：")
print(one_shot_all_avg)

print("\nZero-shot 测试中所有模型的平均得分：")
print(zero_shot_all_avg)


# =========================
# 5. 保存统计表
# =========================
one_shot_all_avg.to_csv("all_models_one_shot.csv", index=False, encoding="utf-8-sig")
zero_shot_all_avg.to_csv("all_models_zero_shot.csv", index=False, encoding="utf-8-sig")


# =========================
# 6. 绘图
# =========================
plot_all_model_scores(
    one_shot_all_avg,
    title="所有模型在 One-shot 测试中的平均得分",
    ylabel="平均得分（%）",
    output_file="all_models_one_shot.png"
)

plot_all_model_scores(
    zero_shot_all_avg,
    title="所有模型在 Zero-shot 测试中的平均得分",
    ylabel="平均得分（%）",
    output_file="all_models_zero_shot.png"
)