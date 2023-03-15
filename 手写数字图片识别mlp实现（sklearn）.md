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
