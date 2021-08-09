import torch
import numpy as np

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
