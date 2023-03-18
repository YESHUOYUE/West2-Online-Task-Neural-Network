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
