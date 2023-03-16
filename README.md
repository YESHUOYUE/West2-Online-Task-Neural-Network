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
