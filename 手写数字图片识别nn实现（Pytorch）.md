# 代码
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


print(torch.cuda.is_available())

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
print(torch.cuda.get_device_name(0))

from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')

data_labels = data['label']
data_images = data.drop(columns=['label'])

train_labels, test_labels, train_images, test_images = train_test_split(data_labels, data_images,
                                                    train_size=0.8,
                                                    random_state=42,
                                                    shuffle=True)


# 转换成张量
train_labels = torch.tensor(train_labels.to_numpy(), dtype=torch.float)
train_images = torch.tensor(train_images.to_numpy(), dtype=torch.float)
test_labels = torch.tensor(test_labels.to_numpy(), dtype=torch.float)
test_images = torch.tensor(test_images.to_numpy(), dtype=torch.float)

print(train_images.size())
print(train_images)

if torch.cuda.is_available():
    # 转换成GPU版本
    train_labels = train_labels.cuda()
    train_images = train_images.cuda()
    test_labels = test_labels.cuda()
    test_images = test_images.cuda()
    
    
    
from torch import nn
from torch.autograd import Variable

# 设置超参数
batch_size, lr, epochs = 100, 0.001, 20

train = torch.utils.data.TensorDataset(train_images,train_labels)
test = torch.utils.data.TensorDataset(test_images,test_labels)

# 将数据转为dataloader
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)


# # 建立模型
# class Net(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size) 
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out
    
# input_size = 784
# hidden_size = 500
# output_size = 10

# network = Net(input_size, hidden_size, output_size)

network = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
)

if torch.cuda.is_available():
    network.to(torch.device("cuda:0"))

# 损失函数
loss_func = nn.CrossEntropyLoss()
# 优化算法
optimizer = torch.optim.SGD(network.parameters(), lr=lr)

# 训练模型
print(len(train_loader))
total_step = len(train_loader)
for t in range(epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.view(-1, 28* 28).to(device)    # -1 是指模糊控制的意思，即固定784列，不知道多少行
        labels = labels.to(device)
        
        outputs = network(images)
        labels=labels.to(torch.int64)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(t+1, epochs, i+1, total_step, loss.item()))


```
# 结果
![image](https://user-images.githubusercontent.com/116483698/225670835-55595a10-9860-4efe-80a7-e67f583f56f7.png)
![image](https://user-images.githubusercontent.com/116483698/225670946-f472da72-3db6-4b94-8cfb-5d4563997c06.png)
