import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 200
learning_rate = 0.01
epochs = 10

# 训练数据
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

# 测试数据
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)

# 各层的权重
w1, b1 = torch.randn(200, 784, requires_grad=True), \
         torch.zeros(200, requires_grad=True)

w2, b2 = torch.randn(200, 200, requires_grad=True), \
         torch.zeros(200, requires_grad=True)

w3, b3 = torch.randn(10, 200, requires_grad=True), \
         torch.zeros(10, requires_grad=True)

# 初始化 --在做新的方法的时候，没有好的结果，初始化很重要******************
# 现在使用的何凯明的初始化方法
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

# 网络函数
def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)  # 使用了relu 整型的线性单元函数，结果为0或者x
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x


optimizer = optim.SGD([w1, b1, w2, b2, w3, b3],
                      lr=learning_rate)  # torch 内置的迭代函数--优化器 https://www.jianshu.com/p/f8955dbc3553
criteon = nn.CrossEntropyLoss()  # 与F.cross_entropy 相同

for epoch in range(epochs):  # 数据训练 10次

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)  # 数据值

        logits = forward(data)  # 网络层函数 加载数据后，得到分类值
        loss = criteon(logits, target)  # 计算loss ，使用的是 交叉熵 值

        optimizer.zero_grad()  # 优化器
        loss.backward()  # 后向网络
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()  # 优化器迭代

        if batch_idx % 100 == 0:
            # Train Epoch: 9[40000 / 60000(67 %)] Loss: 0.083235
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0
    for data, target in test_loader:  # 测试数据
        data = data.view(-1, 28 * 28)
        logits = forward(data)  # 预测值
        test_loss += criteon(logits, target).item()  # 交叉熵 叠加

        pred = logits.data.max(1)[1]  # 预测值最大值
        correct += pred.eq(target.data).sum()  # 预测值 与 测试值 是否eq相同，集合

    test_loss /= len(test_loader.dataset)

    # 平均loss，准确率
    # Test set: Average loss: 0.0007, Accuracy: 9557 / 10000(95 %)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
