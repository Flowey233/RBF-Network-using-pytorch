import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据集
file_path = 'your_file_path'
df = pd.read_csv(file_path)

# 使用 train_test_split 函数划分数据集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存训练集和测试集到新文件
train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)
