# Qwen3-4B 指令微调（PyTorch + Transformers）

脚本：`train_qwen3_finetune.py`

## 启动方式（按你的要求）
不使用命令行参数，直接运行：

```bash
python develop/fine_tune/train_qwen3_finetune.py
```

## 数据集位置与格式
- 数据集文件放在**脚本同目录**：`develop/fine_tune/`
- 脚本优先查找：
  - `train.json`
  - `train.jsonl`
  - `dataset.json`
  - `dataset.jsonl`
- 若上述文件都不存在，会自动使用该目录下第一个 `.json` 或 `.jsonl` 文件。

每条数据示例：

```json
{
  "reference": ["法条1", "法条2"],
  "question": "用户问题",
  "answer": "标准答案"
}
```

## 结果保存位置
训练后的参数会保存在脚本同目录自动创建的：

- `develop/fine_tune/qwen3_4b_finetuned/`

## 说明
- 使用 `reference + question` 作为 user 输入，`answer` 作为 assistant 监督目标。
- 训练时会把 user 部分 label 置为 `-100`，仅对 assistant 回复计算损失。
- 如需调整训练超参数，直接修改脚本顶部配置常量。
