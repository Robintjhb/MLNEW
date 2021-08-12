import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import torch


# 2d 函数
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)  # 先指定范围
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()  # plt显示

# [1., 0.], [-4, 0.], [4, 0.]
x = torch.tensor([-4., 0.], requires_grad=True)  # 初始化值

optimizer = torch.optim.Adam([x], lr=1e-3)  # 使用torch内置的优化函数

for step in range(20000):  # 迭代

    pred = himmelblau(x)  # 每次迭代值

    optimizer.zero_grad()
    pred.backward()  # 后向传播，求导
    optimizer.step()  # 迭代求导

    if step % 2000 == 0:
        print(pred)
        print('step {}: x = {}, f(x) = {}'
              .format(step, x.tolist(), pred.item()))  # 打印求导结果
