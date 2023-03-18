# 导入库
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
```
![image](https://user-images.githubusercontent.com/116483698/226077944-62cd98c1-d0b1-4453-a938-938092398f1c.png)

> 结果为True，并返回使用的device以及GPU名称


# 分离数据集并处理数据
```python
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')

data_labels = data['label']
data_images = data.drop(columns=['label'])

train_labels, test_labels, train_images, test_images = train_test_split(data_labels, data_images,
                                                    train_size=0.80,
                                                    shuffle=True)

# 转换成张量
train_labels = torch.tensor(train_labels.to_numpy(), dtype=torch.float)
train_images = torch.tensor(train_images.to_numpy(), dtype=torch.float)
test_labels = torch.tensor(test_labels.to_numpy(), dtype=torch.float)
test_images = torch.tensor(test_images.to_numpy(), dtype=torch.float)


if torch.cuda.is_available():
    # 转换成GPU版本
    train_labels = train_labels.cuda()
    train_images = train_images.cuda()
    test_labels = test_labels.cuda()
    test_images = test_images.cuda()
```

# 转换数据格式
```python
from torch import nn
from torch.autograd import Variable

# 设置超参数
batch_size, lr, epochs = 100, 0.001, 50

train = torch.utils.data.TensorDataset(train_images,train_labels)
test = torch.utils.data.TensorDataset(test_images,test_labels)

# 将数据转为dataloader
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
```

# 建立模型（两种方法）
## 方法一
```python
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
input_size = 784
hidden_size = 500
output_size = 10

network = Net(input_size, hidden_size, output_size)

if torch.cuda.is_available():
    network.to(torch.device("cuda:0"))
```

## 方法二
```python
network = nn.Sequential(
    nn.Linear(784, 500),
    nn.ReLU(),
    nn.Linear(500, 10)
)

if torch.cuda.is_available():
    network.to(torch.device("cuda:0"))

```

# 损失函数和优化算法
```python
# 损失函数
loss_func = nn.CrossEntropyLoss()
# 优化算法
optimizer = torch.optim.SGD(network.parameters(), lr=lr)
```

# 训练模型
```python
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

# 计算准确率
```python
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = network(images)
        _, predicted = torch.max(outputs.data, 1)  
        total += labels.size(0)  ##更新测试图片的数量   size(0),返回行数
        correct += (predicted == labels).sum().item() ##更新正确分类的图片的数量

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
```
> 对于 _, predicted = torch.max(outputs.data, 1)，这里返回两组数据，最大image_data和最大值索引，可以用torch.argmax（）更为直观；这里去理解其作用为返回最大索引，即预测出来的类别。
> _ , predicted是python的一种常用的写法，表示后面的函数其实会返回两个值，但是我们对第一个值不感兴趣，就写个_在那里，把它赋值给_就好，我们只关心第二个值predicted
> 比如 _ ,a = 1,2 这中赋值语句在python中是可以通过的，你只关心后面的等式中的第二个位置的值是多少

运行结果（迭代50次，只截取开头和结尾的部分迭代）：
![image](https://user-images.githubusercontent.com/116483698/226078475-74def5e7-a09b-4987-977d-0e107b9826df.png)
![image](https://user-images.githubusercontent.com/116483698/226078486-9d6fc16d-c9b9-4c91-a9e6-565e4fd4cd07.png)
![image](https://user-images.githubusercontent.com/116483698/226078501-6afd2c09-d47a-4a98-bf88-74ced72fb00c.png)

