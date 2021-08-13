import torch
import numpy as np
from torch.nn import functional as F  # 函数接口
import torch.nn as nn  # 函数接口

# 张量 概念

# https://zhuanlan.zhihu.com/p/265394674

# torch.tensor
arr = np.ones((3, 3))  # 3x3 都是1的数组
print("arr", arr)
print("ndarray的数据类型：", arr.dtype)
# 创建存放在 GPU 的数据
t = torch.tensor(arr, device='cuda')
# t= torch.tensor(arr)
print(t)

# 从 numpy 创建 tensor。
# 利用这个方法创建的 tensor 和原来的 ndarray 共享内存，
# 当修改其中一个数据，另外一个也会被改动。
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)

t = torch.from_numpy(arr)  # 从 numpy 创建 tensor----

# 修改 array，tensor 也会被修改
# print("\n修改arr")
arr[0, 0] = 0
print("numpy array: ", arr)
print("tensor : ", t)

# 修改 tensor，array 也会被修改
print("\n修改tensor")
t[0, 0] = -1
print("numpy array: ", arr)
print("tensor : ", t)

# 根据数值创建 Tensor
# torch.zeros()
# torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 功能：根据 size 创建全 0 张量

# size: 张量的形状
# out: 输出的张量，如果指定了 out，那么torch.zeros()返回的张量和 out 指向的是同一个地址
# layout: 内存中布局形式，有 strided，sparse_coo 等。当是稀疏矩阵时，设置为 sparse_coo 可以减少内存占用。
# device: 所在设备，cuda/cpu
# requires_grad: 是否需要梯度

out_t = torch.tensor([1])
# 这里制定了 out
t = torch.zeros((3, 3), out=out_t)
print(t, '\n', out_t)
# id 是取内存地址。最终 t 和 out_t 是同一个内存地址
print(id(t), id(out_t), id(t) == id(out_t))

# torch.zeros_like 功能：根据 input 形状创建全 0 张量

# torch.full()，torch.full_like() 创建自定义数值的张量
# size: 张量的形状，如 (3,3)
# fill_value: 张量中每一个元素的值

t = torch.full((3, 3), 1)
print(t)

# torch.arange()功能：创建等差的 1 维张量。注意区间为[start, end)。

# start: 数列起始值
# end: 数列结束值，开区间，取不到结束值
# step: 数列公差，默认为 1
# torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
t = torch.arange(2, 10, 2)
print(t)  # tensor([2, 4, 6, 8])

# 根据概率创建 Tensor
# torch.normal() 
# #功能：生成正态分布 (高斯分布)
# mean: 均值
# std: 标准差
# torch.normal(mean, std, *, generator=None, out=None)

# 我们从一个标准正态分布N～(0,1)，提取一个2x2的矩阵
mean = torch.zeros((4,))
t_normal = torch.normal(mean, 0.02).reshape(2, 2)
print(t_normal)

# mean：张量 std: 标量
mean = torch.arange(1, 5, dtype=torch.float)
std = 1
t_normal = torch.normal(mean, std)
print("mean:{}\nstd:{}".format(mean, std))
print(t_normal)

# # dim
# dim=1，指定列，也就是行不变，列之间的比较，
# 所以原来的a有三行，最后argmax()出来的应该也是三个值，
# 第一行的时候，同一列之间比较，最大值是0.8338，索引是0，同理，第二行，最大值的索引是1……
a = np.array([[0.3300, 0.5988, 0.4680, 0.8495],
              [0.2334, 0.7431, 0.1800, 0.1857],
              [0.6215, 0.5906, 0.2555, 0.1662]])

print(a)
t = torch.tensor(a, device='cuda')
b = torch.argmax(t, dim=1)  ##指定列，也就是行不变，列之间的比较
print(b)
print(b[0])
print(a[0, 3])
print(a[1, 1])
print(a[2, 0])

# [[0.33   0.5988 0.468  0.8495]
#  [0.2334 0.7431 0.18   0.1857]
#  [0.6215 0.5906 0.2555 0.1662]]
# tensor([3, 1, 0], device='cuda:0') #行不变，列之间比较，找到了最大得列得标号
# tensor(3, device='cuda:0')
# 0.33

# shape属性
print(t.shape)
x = torch.tensor([
    [0, 2],
    [3, 4],
    [9, 8]
])
print(x.shape)

x = torch.tensor([
    [
        [0, 2],
        [0, 8],
        [2, 7]
    ],
    [
        [2, 5],
        [9, 3],
        [7, 3]
    ]
])
print(x)
print(x.shape)  # 2行，每行3行x2列
print(x[1, 1, 0])

# 对角线矩阵
t = torch.eye(3)
print(t)
# #tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

# randperm 随机打散
t = torch.randperm(10)
print(t)
# tensor([8, 0, 3, 1, 4, 2, 7, 5, 6, 9])

# 相同的索引
a = torch.rand(4, 3)
b = torch.rand(4, 3)

idx = torch.randperm(3)

print(idx)
print(a)
print(a[idx])  # 输出按行的标号的矩阵
print(b[idx])

# tensor([1, 0, 2]) 
# tensor([[0.2349, 0.9668, 0.9344],#0
#         [0.8428, 0.5992, 0.2636],#1
#         [0.3811, 0.2809, 0.4656],#2
#         [0.7883, 0.3680, 0.3094]])#3
# tensor([[0.8428, 0.5992, 0.2636],#1
#         [0.2349, 0.9668, 0.9344],#0
#         [0.3811, 0.2809, 0.4656]])#3

# 索引及切片
a = torch.rand(4, 3, 28, 28)
print(a[0].shape)
print(a[0, 0].shape)
print(a[0, 0, 2, 4])  # 标量

# torch.Size([3, 28, 28])
# torch.Size([28, 28])
# tensor(0.0403) # 标量

# #从第几维挑选数据----在想想看吧
# dim: 要查找的维度
# index：表示从第一个参数维度中的哪个位置挑选数据，类型为torch.Tensor类的实例；
# index_select(
#     dim,
#     index
# )
print(a.index_select(2, torch.arange(8)).shape)

t = torch.arange(24).reshape(2, 3, 4)  # 初始化一个tensor，从0到23，形状为（2,3,4）
print("t--->", t)
#
index = torch.tensor([1, 2])  # 要选取数据的位置
print("index--->", index)
#
data1 = t.index_select(1, index)  # 第一个参数:从第1维挑选， 第二个参数:从该维中挑选的位置
print("data1--->", data1)
#
data2 = t.index_select(2, index)  # 第一个参数:从第2维挑选， 第二个参数:从该维中挑选的位置
print("data2--->", data2)

# 维度变换
a = torch.rand(4, 1, 28, 28)
print(a.shape)
a1 = a.view(4, 28 * 28)
print(a1.shape)

# Broadcast ,不同维度进行操作，需要对其中一个进行扩展

a = torch.tensor([1, 2, 3])
print("a-->", a.size())
print("a-->", a)

b = torch.tensor([[2, 2, 2], [3, 3, 3]])
print("b-->", b.size())
print("b-->", b)

c = a.expand_as(b)

print(c)
print("c-->", c.size())

# a = torch.rand(4, 32,28, 28)
# b=torch.rand(3,1,1)
# #b=b.unsqueeze(0) #在第1维增加一个维度，使其维度变为（1，3，1,1）
# # b.unsqueeze(0).shape

# c = b.expand_as(a)  -------还是有问题
# print(a+b)

# 拼接与拆分
# ▪ Cat--合并
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
c = torch.cat([a, b], dim=0)
print(c.shape)  # torch.Size([9, 32, 8])

# ▪ Stack
a = torch.rand(32, 8)
b = torch.rand(32, 8)
c = torch.stack([a, b], dim=0)
print(c.shape)  # torch.Size([2, 32, 8])，建立新的维度，2维度--》32x8

# ▪ Split 按长度进行拆分

aa, bb = c.split(1, dim=0)
print(aa.shape, bb.shape)

# ▪ Chunk 按数量 进行 拆分

aa, bb = c.chunk(2, dim=0)  # 按2个数量 进行拆分---》拆分成两个
print(aa.shape, bb.shape)

# 基本运算

a = torch.rand(3, 4)
b = torch.rand(4)

print(a + b)

# 矩阵乘法 .matmul  @

x = torch.rand(4, 784)
w = torch.rand(512, 784)
print((x @ w.t()).shape)  # torch.Size([4, 512])

# 次方运算：pow **

a = torch.full([2, 2], 3)
print(a ** 2)  # 平方
# tensor([[9., 9.],
#         [9., 9.]])


print((a ** 2).sqrt())  # 开方

# torch.clamp
# 将输入input张量每个元素的夹紧到区间 [min,max]内，并返回结果到一个新张量。

a = torch.randint(low=0, high=10, size=(10, 1))
print(a)
a = torch.clamp(a, 3, 9)  # 会把低于3的变为3，把高于9的变为9，数组数量不变
print(a)

# 统计属性

#  范数 
a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print(a, b, c)
# 1-范数
print(a.norm(1), b.norm(1), c.norm(1))
# 2-范数
print(a.norm(2), b.norm(2), c.norm(2))
# 1-范数，在dim维度1上进行
print(b.norm(1, dim=1), c.norm(2, dim=1))

# argmax argmin
a = torch.tensor([[0.4169, 0.4982, 0.7874],
                  [0.6238, 0.5957, 0.7438],
                  [0.4239, 0.6937, 0.6177],
                  [0.5326, 0.9344, 0.8445]])
print(a)
print(a.argmax())  # 求最大值的位置，是打平的维度--为1维度的数组
# tensor(9)
print(a.argmax(dim=1))  # 求最大值的位置，指定的dim，这样就可以找到，在那个维度上的了
# tensor([9, 5, 7, 2])

print(a.max(dim=1))  # 求最大值,返回值及位置
# (tensor([0.7874, 0.7438, 0.6937, 0.9344]), tensor([2, 2, 1, 1]))

print(a.max(dim=1, keepdim=True))  # 保留维度，来求最大值,返回值及位置
# (tensor([[0.7874],
#         [0.7438],
#         [0.6937],
#         [0.9344]]), tensor([[2],
#         [2],
#         [1],
#         [1]]))

# 激活函数
# σ(sigmoid)
a = torch.linspace(-100, 100, 10)
print(a)

# (sigmoid)
sigmoid_σ = torch.sigmoid(a)

print(sigmoid_σ)

# ReLU函数

a = torch.linspace(-1, 1, 10)
relu = torch.relu(a)

print(relu)
# tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1111, 0.3333, 0.5556, 0.7778,
#         1.0000])

# loss
# 均方差MSE:求导
x = torch.ones(1)
w = torch.full([1], 2)
# loss 值
mse = F.mse_loss(torch.ones(1), x * w)  # torch.one(1)预测值，x*w计算值

w.requires_grad_()  # 更新求导
mse = F.mse_loss(torch.ones(1), x * w)  # 重新计算

autograd = torch.autograd.grad(mse, [w])
print(autograd)  # (tensor([2.]),)

# 第二种方法
mse = F.mse_loss(torch.ones(1), x * w)  # 重新计算
mse.backward()  # 向后传播
w_grad = w.grad
print(w_grad)

# 分类问题--softmax函数
a = torch.rand(3)
a.requires_grad_()  # 更新求导

p = F.softmax(a, dim=0)  # softmax 函数：预测值：a，dim对那个维度进行

autograd = torch.autograd.grad(p[1], [a], retain_graph=True)  # p[1]:对第i个进行，[a]:变量 retain_graph声明求导

print(autograd)  # (tensor([-0.0736,  0.1604, -0.0868]),)

autograd = torch.autograd.grad(p[2], [a])  # retain_graph声明求导
print(autograd)  # (tensor([-0.1586, -0.0868,  0.2455]),)

# 单层 感知机 模型：实现
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)

o = torch.sigmoid(x @ w.t())  # 激活函数

print(o)

loss = F.mse_loss(torch.ones(1, 1), o)  # loss函数 mse

loss.backward()  # 后向传播

w_grad = w.grad
print(w_grad)  # 得到w的导数

# 多层 感知机
x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)  # 2层的

o = torch.sigmoid(x @ w.t())  # 激活函数

print(o)

loss = F.mse_loss(torch.ones(1, 2), o)  # loss函数 mse

loss.backward()  # 后向传播

w_grad = w.grad
print(w_grad)  # 得到w的导数 2x10 的导数 结果

# 链式法则
x = torch.tensor(1.)
w1 = torch.tensor(2., requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2., requires_grad=True)
b2 = torch.tensor(1.)

y1 = x * w1 + b1
y2 = y1 * w2 + b2

# 求导
dy2_dy1 = torch.autograd.grad(y2, [y1])[0]
dy1_dw1 = torch.autograd.grad(y1, [w1])[0]

dy2_dw1 = dy2_dy1 * dy1_dw1  # 手动 链式求导
print(dy2_dw1)

# dy2_dw1_n = torch.autograd.grad(y2, [w1])[0]  # 直接使用torch提供的求导
#
# print(dy2_dw1_n)


# 交叉熵

x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x @ w.t()
pred = F.softmax(logits, dim=1)
pred_log = torch.log(pred)

ce1 = F.nll_loss(pred_log, torch.tensor([3]))
print(ce1)
ce2 = F.cross_entropy(logits, torch.tensor([3]))
print(ce2)

# BatchNorm  批规范化-->批量正态分布：

x = torch.rand(100, 16, 784)
layer = nn.BatchNorm1d(16) # BatchNorm
out = layer(x)

print(layer.running_mean)
print(layer.running_var)

# 二维 BatchNorm2d

x = torch.rand(1, 16, 28, 28)  # 这里是28*28的数据

# 二维直接使用.BatchNorm2d
# 因为Batch Norm的参数直接是由channel数量得来的，
# 因此这里直接给定了channel的数量为16，后续会输出16个channel的统计信息
layer = nn.BatchNorm2d(16)


out = layer(x)


# 进行权值计算并输出
print("进行权值计算并输出:",layer.weight)
print(layer.bias)


print(layer.running_mean)
print(layer.running_var)