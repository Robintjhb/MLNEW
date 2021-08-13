import torch
import numpy as np
from torch.nn import functional as F  # 函数接口
import torch.nn as nn  # 函数接口

x = torch.randn(1, 16, 14, 14)
layer = nn.MaxPool2d(2, stride=2) #n.MaxPool2d

out = layer(x)
print(out.shape) # torch.Size([1, 16, 7, 7])

out = F.avg_pool2d(x, 2, stride=2) #F.avg_pool2d

print(out.shape) # torch.Size([1, 16, 7, 7])

out = F.interpolate(x, scale_factor=2, mode='nearest')  # 放大2倍，nearest就是采用临近复制的模式，channel通道不会改变

print(out.shape) # torch.Size([1, 16, 28, 28])

layer=nn.ReLU(inplace=True)  # inplace=True使用变量的内存空间,ReLU: out最小值都变成0，因为负数都被过滤了
out=layer(x)
print(out.shape) # torch.Size([1, 16, 28, 28])
# print(x) # torch.Size([1, 16, 28, 28])
print(out) # torch.Size([1, 16, 28, 28])
