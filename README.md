# 手写数字图片识别不同方法实现（knn、mlp、nn）

## knn
代码：
```python
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('train.csv')

train_labels = data['label']
train_images = data.drop(columns=['label'])

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.25)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

score2 = accuracy_score(y_test, prediction)
print("KNN :", score2)

cv_scores2 = cross_val_score(knn, train_images, train_labels, cv=10)
print("KNN(cross):", np.mean(cv_scores2))
```

运行结果：

![image](https://user-images.githubusercontent.com/116483698/225035566-d698132b-630e-4387-b61f-bc7078e9b5cc.png)


## mlp(sklearn)
代码：
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# 导入数据
train_data = pd.read_csv('train.csv')

# 处理数据
train_labels = train_data['label']
train_images = train_data.drop(columns=['label'])

# 数据分类
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.25)

# 建立模型
mlp = MLPClassifier()

# 训练数据
mlp.fit(X_train, y_train)

# 评价模型
score = mlp.score(X_test, y_test)
print(score)

```

运行结果：
![image](https://user-images.githubusercontent.com/116483698/225287961-74ebd627-1bbb-4b6e-8d24-ea991298f0f4.png)


## 神经网络(tensorflow)
代码：
```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 分离数据的标签与特征
train_labels = train_data['label']
train_images = train_data.drop(columns=['label'])

# 将数据转为ndarray格式
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_data)

# 数据信息展示
n = 90
print('训练数据维度', train_images.shape)
print('测试图片个数', len(test_images))
print('训练集第', n, '个数字图像的识别结果: ', train_labels[n])

# 数据图片展示
plt.imshow(train_images[n].reshape(28, 28))
plt.show()

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 建立模型
network = Sequential()
# 添加一个层，激活函数为ReLU函数，输入数据形状为28*28
network.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
# 激活函数为softmax
network.add(Dense(10, activation='softmax'))
# 学习率的设置
adam = Adam(lr=0.001)

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# 对数据进行预处理，将其变换成网络要求的形状，缩放到[0, 1]之间
train_images = train_images.reshape((42000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((28000, 28 * 28))
test_images = test_images.astype('float32') /255

# 对标签进行分类编码
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)

# 训练数据
network.fit(train_images, train_labels, epochs=10, batch_size=128)

# 预测训练集
predict = []
predict_test = network.predict(test_images)
predict = np.argmax(predict_test, 1)  # axis = 1是取行的最大值的索引，0是列的最大值的索引

print(predict)

```

结果展示：
![image](https://user-images.githubusercontent.com/116483698/225334433-96d9924c-8ac7-4ab0-9851-99e3d0770c1d.png)

![image](https://user-images.githubusercontent.com/116483698/225335264-64fbba37-1e5d-4694-8b68-29b38615ead0.png)

## 神经网络（pytorch）
### 导入库
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


### 分离数据集并处理数据
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

### 转换数据格式
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

### 建立模型（两种方法）
#### 方法一
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

#### 方法二
```python
network = nn.Sequential(
    nn.Linear(784, 500),
    nn.ReLU(),
    nn.Linear(500, 10)
)

if torch.cuda.is_available():
    network.to(torch.device("cuda:0"))

```

### 损失函数和优化算法
```python
# 损失函数
loss_func = nn.CrossEntropyLoss()
# 优化算法
optimizer = torch.optim.SGD(network.parameters(), lr=lr)
```

### 训练模型
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

### 计算准确率
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

