# 导入库
```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

# 导入数据and数据信息处理
```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 分离数据的标签与特征
train_labels = train_data['label']
train_images = train_data.drop(columns=['label'])

# 将数据转为ndarray格式
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_data)
```

# 数据信息展示
```python
n = 90
print('训练数据维度', train_images.shape)
print('测试图片个数', len(test_images))
print('训练集第', n, '个数字图像的识别结果: ', train_labels[n])

# 数据图片展示
plt.imshow(train_images[n].reshape(28, 28))
plt.show()
```
![image](https://user-images.githubusercontent.com/116483698/225342039-e50207d3-e41a-4e7a-9c6f-9532d87a0e20.png)


# 建立模型
```python
# 导入keras库
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
```
# （补充）Dense层常见参数
```python
model.add(Dense(units, #输出的大小（神经元个数）
                activation=None, #激活函数
                use_bias=True, #是否添加偏置
                kernel_initializer='glorot_uniform', #权重矩阵初始化
                bias_initializer='zeros', #偏置初始化
                kernel_regularizer=None, #权重矩阵的正则函数
                bias_regularizer=None,) #偏置的的正则函数
          )
```

# 对数据进行预处理，将其变换成网络要求的形状，缩放到[0, 1]之间
```python
train_images = train_images.reshape((42000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((28000, 28 * 28))
test_images = test_images.astype('float32') /255
```

# 对标签进行分类编码
```python
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
```

# 训练数据
```python
network.fit(train_images, train_labels, epochs=10, batch_size=128)
```

# 预测训练集
```python
predict = []
predict_test = network.predict(test_images)
predict = np.argmax(predict_test, 1)  # axis = 1是取行的最大值的索引，0是列的最大值的索引

print(predict)
```
![image](https://user-images.githubusercontent.com/116483698/225344414-c686b871-ef98-48fc-b289-7f84ee4d121b.png)


# 完整代码：
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

完整结果展示：
![image](https://user-images.githubusercontent.com/116483698/225334433-96d9924c-8ac7-4ab0-9851-99e3d0770c1d.png)

![image](https://user-images.githubusercontent.com/116483698/225335264-64fbba37-1e5d-4694-8b68-29b38615ead0.png)
