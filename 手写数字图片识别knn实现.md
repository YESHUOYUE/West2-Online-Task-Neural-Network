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
