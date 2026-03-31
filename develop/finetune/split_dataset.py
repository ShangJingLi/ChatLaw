import os
import json

input_file = os.path.join("develop", "resources", "finetuning_dataset", "dataset.json")
train_file = os.path.join("develop", "resources", "finetuning_dataset", "train.json")
test_file = os.path.join("develop", "resources", "finetuning_dataset", "test.json")

# 读取数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

train_data = []
test_data = []

# 每19个训练 + 1个测试
for i, item in enumerate(data):
    if i % 20 == 19:
        test_data.append(item)
    else:
        train_data.append(item)

# 写入训练集
with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

# 写入测试集
with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

# 输出统计信息
print("原始数据量:", len(data))
print("训练集数量:", len(train_data))
print("测试集数量:", len(test_data))
print("训练/测试比例:", f"{len(train_data)}:{len(test_data)}")