import os
import re
import numpy as np
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
# 2. 从模型名中提取参数量（单位：B）
# 例如：
# qwen-7b-chat-hf -> 7
# chatglm2-6b-hf -> 6
# qwen3_4b_rag -> 4
# =========================
def extract_param_size(model_name: str):
    model_name = str(model_name).lower()
    match = re.search(r'(\d+(?:\.\d+)?)\s*[_-]?b(?=[^a-z0-9]|$)', model_name)
    if match:
        return float(match.group(1))
    return np.nan


# =========================
# 3. 读取并整理数据
# =========================
def prepare_average_scores(base_csv, my_csv, max_param_b=7):
    base_df = pd.read_csv(base_csv)
    my_df = pd.read_csv(my_csv)

    # 只保留需要的列，然后合并
    df = pd.concat(
        [
            base_df[["task", "model_name", "score"]],
            my_df[["task", "model_name", "score"]],
        ],
        ignore_index=True
    )

    # 按模型求平均分
    avg_df = (
        df.groupby("model_name", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "avg_score"})
    )

    # 提取参数量
    avg_df["param_b"] = avg_df["model_name"].apply(extract_param_size)

    # 只保留参数量 <= 7B 的模型
    avg_df = avg_df[avg_df["param_b"].notna() & (avg_df["param_b"] <= max_param_b)].copy()

    # 转成百分制
    avg_df["avg_score_pct"] = avg_df["avg_score"] * 100

    # 排序
    avg_df = avg_df.sort_values("avg_score_pct", ascending=False).reset_index(drop=True)

    return avg_df


# =========================
# 4. 画图函数
# =========================
def plot_avg_scores(avg_df, title, ylabel, output_file, my_model_name="qwen3_4b_rag"):
    plt.figure(figsize=(18, 7))

    # 你的模型高亮，其余模型统一颜色
    colors = ["#7b3294" if name == my_model_name else "#5ab89a"
              for name in avg_df["model_name"]]

    plt.bar(avg_df["model_name"], avg_df["avg_score_pct"], color=colors)

    plt.title(title, fontsize=20)
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel("模型名称", fontsize=16)

    plt.xticks(rotation=75, ha="right", fontsize=11)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # 图例
    legend_handles = [
        Patch(color="#7b3294", label="本文模型"),
        Patch(color="#5ab89a", label="其他参数量不超过 7B 的模型")
    ]
    plt.legend(handles=legend_handles, fontsize=14, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# 5. 生成 one-shot 和 zero-shot 平均分表
# =========================
one_shot_avg = prepare_average_scores(one_shot_path, my_one_shot_path, max_param_b=7)
zero_shot_avg = prepare_average_scores(zero_shot_path, my_zero_shot_path, max_param_b=7)

print("One-shot 平均分结果：")
print(one_shot_avg)

print("\nZero-shot 平均分结果：")
print(zero_shot_avg)


# =========================
# 6. 绘图
# =========================
plot_avg_scores(
    one_shot_avg,
    title="参数量不超过 7B 的模型在 One-shot 测试中的平均得分",
    ylabel="One-shot 平均得分（%）",
    output_file="similar_params_models_one_shot.png"
)

plot_avg_scores(
    zero_shot_avg,
    title="参数量不超过 7B 的模型在 Zero-shot 测试中的平均得分",
    ylabel="Zero-shot 平均得分（%）",
    output_file="similar_params_models_zero_shot.png"
)