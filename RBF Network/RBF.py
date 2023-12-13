import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.nn.parameter import Parameter
from Pattern import Pattern
from Data import Data

class RBFNetwork(nn.Module):
    def __init__(self, no_of_input, no_of_hidden, no_of_output, data):
        super(RBFNetwork, self).__init__()
        self.no_of_input = no_of_input
        self.no_of_hidden = no_of_hidden
        self.no_of_output = no_of_output
        self.data = data
        self.learningRate=0.0262
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化中心点
        self.centroid = nn.Parameter(torch.rand((self.no_of_hidden, self.no_of_input)).to(self.device))

        # 初始化σ值
        self.sigma = Parameter(torch.rand(self.no_of_hidden).to(self.device))

        # 初始化隐藏层到输出层的权重
        self.hidden_to_output_weight = Parameter(torch.rand((self.no_of_hidden, self.no_of_output)).to(self.device))
        # 初始化输出层偏置
        self.output_bias = Parameter(torch.rand(self.no_of_output).to(self.device).to(self.device))

        
    @staticmethod
    def euclidean_distance(x, y, device):
        """Compute Euclidean distance between tensors x and y"""
        return torch.norm(x.to(device) - y.to(device))
        
    def pass_to_hidden_node(self):
        """将输入传递到隐藏层"""
        self.hidden_output = torch.zeros(self.no_of_hidden, device=self.device)
        for i in range(len(self.hidden_output)):
            euclid_distances = self.euclidean_distance(self.input, self.centroid[i], self.device) ** 2
            self.hidden_output[i] = torch.exp(- (euclid_distances / (2 * self.sigma[i] * self.sigma[i])))

    def pass_to_output_node(self):
        """将输入传递到输出层"""
        self.output = torch.matmul(self.hidden_output, self.hidden_to_output_weight) + self.output_bias.to(self.device)
        self.output = self.output / torch.sum(self.output)
   
    
    def train(self, n, batch_size, save_interval, resume_path=None):
        optimizer = optim.SGD(self.parameters(), lr=self.learningRate)
        criterion = nn.MSELoss()

        # 如果提供了 resume_path，则加载之前保存的模型参数
        if resume_path:
            self.load_state_dict(torch.load(resume_path))
            print(f"Resuming training from {resume_path}")

        for epoch in range(n):
            all_error = 0
            all_index = [i for i in range(len(self.data.patterns))]

            # 打乱数据集的顺序
            random.shuffle(all_index)

            # 计算总批次数
            num_batches = math.ceil(len(all_index) / batch_size)

            for batch_num in range(num_batches):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, len(all_index))
                batch_indices = all_index[start_index:end_index]

                for i in batch_indices:
                    pattern = self.data.patterns[i]

                    # Convert to PyTorch tensors
                    self.input = torch.tensor(pattern.input, dtype=torch.float32).to(self.device)
                    self.actual_target_values = torch.tensor(pattern.output, dtype=torch.float32).to(self.device)

                    # Forward pass
                    self.pass_to_hidden_node()
                    self.pass_to_output_node()

                    # Compute loss and backpropagate
                    optimizer.zero_grad()
                    loss = criterion(self.output, self.actual_target_values)
                    loss.backward()
                    optimizer.step()

                    all_error += loss.item()

                # 保存模型参数
                if (batch_num + 1) % save_interval == 0:
                    torch.save(self.state_dict(), f'\\epoch_{epoch + 1}_batch_{batch_num + 1}.pth')

                print(f"Epoch {epoch + 1}, Batch {batch_num + 1}/{num_batches}, Mean Squared Error: {all_error / len(batch_indices)}")

            all_error = all_error / len(self.data.patterns)
            print(f"Epoch {epoch + 1}, Mean Squared Error: {all_error}")

        print("\nFinish training\n")
        return all_error



# 余下的代码保持不变
file_path = 'your_dataset_path'
df = pd.read_csv(file_path)
label_encoder = LabelEncoder()
# 转换为float类型，将非数值的值转换为NaN
df = df.apply(lambda x: label_encoder.fit_transform(x) if x.dtype == 'O' else x)

# 删除包含NaN值的行
df = df.dropna()

# 提取特征值和输出值列
features = df.iloc[:, :9].values  # 提取前 9 列作为特征值
output_values = df.iloc[:, 9].values  # 假设输出值在第 10 列

# 创建 Pattern 对象
patterns = [Pattern(i, features[i], [output_values[i]]) for i in range(len(features))]

# 创建 Data 对象
class_labels = ['your_labels']
data = Data(patterns, class_labels)
# 创建 RBFNetwork 实例
rbf = RBFNetwork(9, 6, 1, data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rbf = rbf.to(device)

# 执行训练
mse = rbf.train(1,20000,10,resume_path='your_pth')
print("Last MSE ", mse)
