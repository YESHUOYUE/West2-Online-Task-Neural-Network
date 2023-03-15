Dense层常见参数
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
