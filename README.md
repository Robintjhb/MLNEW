# MLNEW

重新学习机器学习~加油
保持好，学习的激情与深度，加油

#  Pytorch

昨天很高兴，看到了这个，现在重新学习机器学习方面或者说人工智能方面的吧，通过不断的锤炼，做出好用的产品

## pytorch初步：

    看着这个进行学习吧，加油：
    https://www.bilibili.com/video/BV1U54y1E7i6
1.安装

![20210806111825.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210806111825.png)

(1)Anacoda5.3.1---到清华镜像下载
Anacoda安装目录下面就有python.exe 查看python版本
![20210806142714.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210806142714.png)

(2)cuda版本 查看：
    
https://blog.csdn.net/zz2230633069/article/details/103725229

安装包下载地址(先装cuda10吧)：

https://developer.nvidia.com/cuda-10.0-download-archive

安装注意点：

自定义安装：

https://www.pianshen.com/article/4550345762/

![20210806134943.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210806134943.png)

运行nvcc -v出现 nvcc fatal : No input files specified; use option --help for more information

    nvcc -V

 就可以了

(3)安装pytorch：

    conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch

https://pytorch.org/get-started/previous-versions/





(4)pycharm下载：选择社区免费版本下载

https://www.jetbrains.com/zh-cn/pycharm/download/#section=windows

![20210806135736.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210806135736.png)

(5) 在vscode中运行py文件：

    直接安装python插件，运行，ctrl+alt+n，运行

![20210806172019.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210806172019.png)


![20210806144904.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210806144904.png)

对py文件自动补全：

https://blog.csdn.net/XU18829898203/article/details/106042078



![20210806171855.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210806171855.png)

对vscode中python执行，
    
（1）进行settings.json中配置：

    // 设置Python的路径
    "python.pythonPath": "C:/ProgramData/Anaconda3/python",
    "python.terminal.executeInFileDir": true,
   
    //万能运行环境
    "code-runner.executorMap": {
        
        //"python": "set PYTHONIOENCODING=utf8 && C:/ProgramData/Anaconda3/python"
        "python": "set PYTHONIOENCODING=utf8 && python",
    },
    // 设置Python的代码格式化
    "python.formatting.provider": "yapf",
    // 设置Python的代码检查
    "python.linting.flake8Path": "pycodestyle",
    "python.autoComplete.extraPaths": [
        "C:/ProgramData/Anaconda3/",
        "C:/ProgramData/Anaconda3/Lib/",
        "C:/ProgramData/Anaconda3/Lib/site-packages/",
        "C:/ProgramData/Anaconda3/DLLs/"
    ],
    "python.autoComplete.addBrackets": true,
    "notebook.cellToolbarLocation": {
        "default": "right",
        "jupyter-notebook": "left"
    },
    "python.analysis.extraPaths": [
        "C:/ProgramData/Anaconda3/",
        "C:/ProgramData/Anaconda3/Lib/",
        "C:/ProgramData/Anaconda3/Lib/site-packages/",
        "C:/ProgramData/Anaconda3/DLLs/"
    ],
    "python.analysis.completeFunctionParens": true,
    "python.linting.pylintPath": "C:\\ProgramData\\Anaconda3\\pkgs\\pylint-2.1.1-py37_0\\Scripts\\pylint",
    "code-runner.fileDirectoryAsCwd": true,

（2）在.vscode中添加launch.json文件，并配置：

        {
            "configurations": [
                {
                    "cwd":"${fileDirname}"
                }
            ]
        
        }
解决相对路径问题：        
![20210809092450.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809092450.png)



(6) 安装成功验证：

    print(torch.__version__)  #torch 版本

    print(torch.cuda.is_available())  #是否支持gpu



# 基础概念

## -张量 torch.tensor

https://zhuanlan.zhihu.com/p/265394674

Tensor 的概念
Tensor 中文为张量。

张量的意思是一个多维数组，它是标量、向量、矩阵的高维扩展。

标量可以称为 0 维张量，向量可以称为 1 维张量，矩阵可以称为 2 维张量，RGB 图像可以表示 3 维张量。你可以把张量看作多维数组。

![20210809094100.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809094100.png)

Tensor 创建的方法

直接创建 Tensor

torch.tensor()

torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)

    data: 数据，可以是 list，numpy

    dtype: 数据类型，默认与 data 的一致

    device: 所在设备，cuda/cpu

    requires_grad: 是否需要梯度

    pin_memory: 是否存于锁页内存

代码示例：

    arr = np.ones((3, 3))
    print("ndarray的数据类型：", arr.dtype)
    # 创建存放在 GPU 的数据
    # t = torch.tensor(arr, device='cuda')
    t= torch.tensor(arr)
    print(t)
    
输出为：



索引与切片

![20210809143446.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809143446.png)


![20210809144314.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809144314.png)

间隔索引：

![20210809144803.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809144803.png)

index_select--？这个在想想看吧

![20210809155055.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809155055.png)

...:

![20210809155539.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809155539.png)

![20210809155817.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809155817.png)

![20210809155926.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809155926.png)

维度变换：

![20210809160057.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809160057.png)


![20210809160642.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809160642.png)


![20210809161718.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809161718.png)

维度扩张：
unsqueeze(正数)：在前面加上维度：

![20210809161959.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809161959.png)


维度删减：

![20210809162547.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809162547.png)


维度扩展：设置为-1时候，不变换

![20210809163006.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809163006.png)

repeat：

![20210809172853.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809172853.png)


矩阵转置：.t

![20210809173100.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210809173100.png)

Transpose

![20210810093056.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810093056.png)



permute:--维度转换，提前或者后移：

![20210810093342.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810093342.png)



Broadcast自动扩展：

        Expand ，
        without copying data
 
维度变换后，相加：       
![20210810094650.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810094650.png)


    a = torch.rand(4, 32, 14, 14)
    b = torch.rand(1, 32, 1, 1)
    c = torch.rand(32, 1, 1)

    # b [1, 32, 1, 1]=>[4, 32, 14, 14]
    print((a + b).shape)
    print((a+c).shape)

拼接/合并：
torch.cat

![20210810110222.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810110222.png)


![20210810110907.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810110907.png)

torch.stack:

![20210810111232.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810111232.png)

拆分：

▪ Split 按长度进行拆分

    aa, bb = c.split(1, dim=0)
    print(aa.shape, bb.shape)


![20210810112543.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810112543.png)



▪ Chunk 按数量 进行 拆分

    aa, bb = c.chunk(2, dim=0)  # 按2个数量 进行拆分---》拆分成两个
    print(aa.shape, bb.shape)


基本运算：

![20210810113215.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810113215.png)

加减乘除：

![20210810113551.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810113551.png)

矩阵乘法：.matmul  @

![20210810113829.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810113829.png)

示例分析：
把4x784 转换为4x512矩阵，
--可以把神经网络学习理解tensor的不断变换的过程--tensor flow的流动

![20210810114601.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810114601.png)

![20210810120000.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810120000.png)

次方运算：pow

![20210810133225.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810133225.png)


幂次方：exp，log

![20210810134316.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810134316.png)

其他：
![20210810134437.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810134437.png)


torch.clamp

将输入input张量每个元素的夹紧到区间 [min,max]内，并返回结果到一个新张量。

    a = torch.randint(low=0, high=10, size=(10, 1))
    print(a)
    a = torch.clamp(a, 3, 9)  # 会把低于3的变为3，把高于9的变为9，数组数量不变
    print(a)


统计属性：

![20210810140737.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810140737.png)

![20210810141559.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810141559.png)

Tensor求和以及按索引求和：torch.sum() torch.Tensor.indexadd()

Tensor元素乘积：torch.prod(input) ---

对Tensor求均值、方差、极值：

torch.mean() torch.var()

torch.max() torch.min()

dim,keepdim:维度，
求
max，返回指定维度的最大值，及 索引。
argmax：返回指定维度的最大值的索引。

![20210810145541.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810145541.png)

topk kthvalue：

![20210810150239.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810150239.png)

比较：

![20210810150722.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810150722.png)

其他：
where：

![20210810151412.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810151412.png)

gather：收集--就是通过索引号，取值

![20210810161134.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810161134.png)

## 梯度：-----重要

什么是梯度：所有偏微分的总和。偏微分：自变量的导数

![20210810161750.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810161750.png)

梯度是有长度，及方向的向量；表现了f(x,y) 在x，y的数值变化，增长的速率：
![20210810162216.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810162216.png)


![20210810164353.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810164353.png)

![20210810164656.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810164656.png)

![20210810164851.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810164851.png)

局部最小点较多--通过连接网络，达到极小值：

![20210810165220.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810165220.png)


马鞍形：一个方向为最小值，一个方向为最大值：鞍点

![20210810165416.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810165416.png)


找到极值点的影响因素：

![20210810165848.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810165848.png)

初始状态：

    何恺明，初始化状态： https://arxiv.org/pdf/1502.01852.pdf

![20210810170249.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810170249.png)



学习率

![20210810171558.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810171558.png)



动速
如何逃出局部最小值：

![20210810171739.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810171739.png)


常见函数梯度：

常见函数：

![20210810171920.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810171920.png)

线性函数 y=wx+b---对w，b进行偏微分求解--》x+1

![20210810172112.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810172112.png)

![20210810172304.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810172304.png)

![20210810172406.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810172406.png)


![20210810172628.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810172628.png)

![20210810172649.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810172649.png)

![20210810172752.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810172752.png)


##  激活函数及梯度


![20210810173113.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810173113.png)

这样的激活函数不可导

![20210810173232.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210810173232.png)

解决不可导，提出了这样的函数：

sigmoid函数：

![20210811092323.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811092323.png)

σ(sigmoid)的倒数是 σ'=σ*(1-σ)
![20210811092635.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811092635.png)

但在正负无穷的时候，σ' 趋近于0，导致导数迷散

![20210811093040.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811093040.png)

torch 中使用：


    from torch.nn import functional as F #函数接口

    # σ(sigmoid)
    a = torch.linspace(-100, 100, 10)
    print(a)

    # (sigmoid)
    sigmoid_σ = torch.sigmoid(a)

    print(sigmoid_σ)




![20210811093137.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811093137.png)


tanh函数：在RNN中使用

![20210811094340.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811094340.png)

tanh的导数：x从正负无穷，y从-1，1

![20210811094526.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811094526.png)

在torch中使用

![20210811094710.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811094710.png)

Rectified Linear Unit 整型的线性单元---ReLU

![20210811095032.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811095032.png)

![20210811095238.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811095238.png)
 
 torch中使用：
    # ReLU函数

    a=torch.linspace(-1, 1, 10)
    relu = torch.relu(a)

    print(relu)
    # tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1111, 0.3333, 0.5556, 0.7778,
    #         1.0000])


![20210811095401.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811095401.png)

## LOSS及其梯度：

![20210811095735.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811095735.png)

经典loss
![20210811095944.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811095944.png)

均方差MSE:
有3种

![20210811100247.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811100247.png)

求导：

![20210811101537.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811101537.png)

torch自动求导：
        
    # 均方差MSE:求导
    x = torch.ones(1)
    w = torch.full([1], 2)
    # loss 值
    mse = F.mse_loss(torch.ones(1), x * w)  # torch.one(1)预测值，x*w计算值

    w.requires_grad_()  # 更新求导
     # 第1种方法
    mse = F.mse_loss(torch.ones(1), x * w)  # 重新计算   
    autograd = torch.autograd.grad(mse, [w])
    print(autograd)  # (tensor([2.]),)

    # 第二种方法
    mse = F.mse_loss(torch.ones(1), x * w)  # 重新计算
    mse.backward()  # 向后传播
    w_grad = w.grad
    print(w_grad)


![20210811102417.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811102417.png)

第2种方法
![20210811103901.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811103901.png)

![20210811104553.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811104553.png)

分类问题--求出概率
softmax: soft version of max

![20210811105353.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811105353.png)

i=j

![20210811105850.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811105850.png)

i不等于j

![20210811110114.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811110114.png)

![20210811110411.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811110411.png)

torch实现：

    # 分类问题--softmax函数
    a = torch.rand(3)
    a.requires_grad_()  # 更新求导

    p = F.softmax(a, dim=0)  # softmax 函数：预测值：a，dim对那个维度进行

    autograd = torch.autograd.grad(p[1], [a], retain_graph=True)# p[1]:对第i个进行，[a]:变量 retain_graph声明求导

    print(autograd)  # (tensor([-0.0736,  0.1604, -0.0868]),)

    autograd = torch.autograd.grad(p[2], [a])  # retain_graph声明求导
    print(autograd)  # (tensor([-0.1586, -0.0868,  0.2455]),)



![20210811112302.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811112302.png)

## 感知机：

![20210811114124.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811114124.png)

单层 感知机 模型：

![20210811114352.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811114352.png)

![20210811115449.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811115449.png)

![20210811120025.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811120025.png)

![20210811134442.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811134442.png)

torch实现：


    # 单层 感知机 模型：实现
    x = torch.randn(1, 10)
    w = torch.randn(1, 10, requires_grad=True)

    o = torch.sigmoid(x @ w.t())  # 激活函数

    print(o)

    loss = F.mse_loss(torch.ones(1, 1), o)  # loss函数 mse

    loss.backward()  # 后向传播

    w_grad = w.grad
    print(w_grad)  # 得到w的导数


![20210811135003.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811135003.png)


多层感知机：多层网络

![20210811141808.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811141808.png)

![20210811142016.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811142016.png)

torch实现：
        
    #多层 感知机
    x = torch.randn(1, 10)
    w = torch.randn(2, 10, requires_grad=True) # 2层的

    o = torch.sigmoid(x @ w.t())  # 激活函数

    print(o)

    loss = F.mse_loss(torch.ones(1, 2), o)  # loss函数 mse

    loss.backward()  # 后向传播

    w_grad = w.grad
    print(w_grad)  # 得到w的导数 2x10 的导数 结果


## 链式法则

链式法则是微积分中的求导法则，用以求一个复合函数的导数。
所谓的复合函数，是指以一个函数作为另一个函数的自变量。

如设
f(x)=3x，g(x)=x+3，
g(f(x))就是一个复合函数，
并且g(f(x))=3x+3 链式法则(chain rule)

若h(x)=f(g(x))

则h'(x)=f'(g(x))g'(x)

链式法则用文字描述，
就是“由两个函数凑起来的复合函数，其导数等于里边函数代入外边函数的值之导数，乘以里边函数的导数。


![20210811164400.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811164400.png)

![20210811164649.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811164649.png)

各类rule原则：
![20210811164935.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811164935.png)

![20210811165112.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811165112.png)

乘积原则：
![20210811165228.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811165228.png)

![20210811165326.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811165326.png)

![20210811165637.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811165637.png)

链式法则过程：

![20210811165920.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811165920.png)

torch实现：
        
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

## Multi-Layer Perceptron(MLP) 多层网络 反向传播：

![20210811172410.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811172410.png)


![20210811172605.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811172605.png)

![20210812085749.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812085749.png)

### 多层网络求导：当前层导数信息 乘以 后面层的导数信息之和。


![20210811173052.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811173052.png)

![20210812090335.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812090335.png)

![20210812090540.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812090540.png)

![20210812091220.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812091220.png)

![20210812091614.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812091614.png)


## 2d 函数优化

![20210812091838.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812091838.png)

示例：

![20210812091932.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812091932.png)

![20210812092113.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812092113.png)

python 实现函数，绘制图形：

![20210812092301.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812092301.png)

torch迭代，找到最小值：

![20210812093207.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812093207.png)
   
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


## 逻辑回归/回归分析 Logistic Regression

![20210812094141.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812094141.png)

![20210812094639.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812094639.png)

![20210812094742.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812094742.png)

![20210812095137.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812095137.png)

不使用最大准确率：

![20210812100650.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812100650.png)

为什么叫逻辑回归：

![20210812101007.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812101007.png)

2分类：

![20210812101055.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812101055.png)

多分类：

![20210812101157.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812101157.png)

softmax引入，更好的找到概率大小的值：

![20210812101322.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812101322.png)

![20210812101354.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812101354.png)

## 经典LOSS:

![20210812101737.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812101737.png)



## 交叉熵 Cross Entropy Loss

![20210812101632.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812101632.png)

熵：不确定性，惊喜度。

    熵越高，惊喜度越低，越稳定；熵越低，越不稳定，惊喜度越高

![20210812102052.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812102052.png)

![20210812102409.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812102409.png)

二分类的交叉熵：

![20210812103010.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812103010.png)

![20210812103315.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812103315.png)

示例：

![20210812103607.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812103607.png)

对于分类问题，使用交叉熵，不使用MSE均方差：

![20210812104037.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812104037.png)

![20210812104145.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812104145.png)

torch 实现：

![20210812104458.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812104458.png)

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


## 多分类问题 实战：

![20210812105124.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812105124.png)

![20210812105156.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812105156.png)

    torch 实现：

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



## 激活函数与GPU加速

tanh函数，sigmoid函数

![20210812113807.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812113807.png)

RELU函数及其他形式：


![20210812113952.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812113952.png)

LeakyReLU：
![20210812114638.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812114638.png)

torch实现： 

    F.leaky_relu()
    nn.LeakyReLU(inplace=True)



![20210812114726.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812114726.png)

SERelu：
![20210812133544.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812133544.png)

gup加速：1.0后，更好实现了

![20210812134116.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812134116.png)


![20210812134713.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812134713.png)


    device = torch.device('cuda:0') #gpu 加速 只有一个cuda显卡直接写为0
    net = MLP().to(device)#gpu 加速
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    criteon = nn.CrossEntropyLoss().to(device)#gpu 加速


## test计算：

不断的训练，导致结果不发生改变了，使用其他数据test时，会出现后面效果很差

![20210812135805.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812135805.png)

argmax 找到最大值的索引，eq进行比较计算，准确率：

![20210812141740.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812141740.png)


![20210812142102.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812142102.png)


When to test
test once per several batch 
test once per epoch
epoch V.S. step?
何时测试

每批次 测试一次
每个代 测试一次

![20210812144425.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812144425.png)

    Epoch：
    训练整个数据集的次数。
    当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次epoch。（也就是说，所有训练样本在神经网络中都进行了一次正向传播 和一次反向传播 ）

    然而，当一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于计算机而言），就需要把它分成多个小块，也就是就是分成多个Batch 来进行训练。

    Batch（批 / 一批样本）：
    将整个训练样本分成若干个Batch。

    Batch_Size（批大小）：
    每批样本的大小。

    Iteration（一次迭代）：
    训练一个Batch就是一次Iteration（这个概念跟程序语言中的迭代器相似）。

    举例
    mnist 数据集有60000张图片作为训练数据，10000张图片作为测试数据。假设现在选择 Batch_Size = 100对模型进行训练。迭代30000次。

    每个 Epoch 要训练的图片数量：60000 (训练集上的所有图像)
    训练集具有的 Batch 个数：60000/100=600
    每个 Epoch 需要完成的 Batch 个数：600
    每个 Epoch 具有的 Iteration 个数：600（==batch，训练一次batch相当于迭代一次）
    每个 Epoch 中发生模型权重更新的次数：600
    训练 10 个Epoch后，模型权重更新的次数：600*10=6000
    不同Epoch的训练，其实用的是同一个训练集的数据。第1个Epoch和第10个Epoch虽然用的都是训练集的60000张图片，但是对模型的权重更新值却是完全不同的。因为不同Epoch的模型处于代价函数空间上的不同位置，模型的训练代越靠后，越接近谷底，其代价越小。
    总共完成30000次迭代，相当于完成了30000/600 = 50 个Epoch

    链接：https://www.jianshu.com/p/45b1c4d30dbe https://www.jianshu.com/p/22c50ded4cf7


## Visdom可视化：

     pip install tensorboardX

![20210812144736.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812144736.png)

TensorboardX

![20210812144914.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812144914.png)

visdom：
![20210812145142.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812145142.png)

    安装

    pip install visdom

    运行

    python -m visdom.server

![20210812150056.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812150056.png)
![20210812150133.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812150133.png)
    有问题可以通过源码，安装

![20210812150153.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812150153.png)

    pip uninstall visdom
    cd 到下载目录

    //pip的执行

    可以git clone到本地后进入文件夹

    pip install -e .　安装

    pip会自动将包复制到site-packages


![20210812150310.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812150310.png)

单条曲线：

![20210812150918.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812150918.png)


多条曲线：

![20210812151739.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812151739.png)

图片和文本的显示：

![20210812152048.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812152048.png)


![20210812152144.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812152144.png)


## 过拟合和欠拟合：


![20210812152833.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812152833.png)

![20210812153106.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812153106.png)

![20210812153209.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812153209.png)

![20210812153356.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812153356.png)
![20210812153516.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812153516.png)
![20210812153640.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812153640.png)

模型能力：model capacity

![20210812153850.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812153850.png)

![20210812153925.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812153925.png)


欠拟合Underfitting：
    的表现是
    
    在训练时和测试时， acc准确率不高，loss也不好

![20210812154253.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812154253.png)

过拟合：Overfitting

     在训练时， acc准确率高，loss也好

     但在测试时， acc准确率不高，loss也不好

    所以泛化能力不好，一般性差

![20210812154637.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812154637.png)

![20210812154754.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812154754.png)


![20210812154923.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812154923.png)

![20210812155643.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812155643.png)


## 交叉验证 K-fold cross-validation

Train-Val-Test划分 固定划分： 训练集--验证集--测试集

![20210812160130.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812160130.png)
![20210812160327.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812160327.png)

K-fold cross-validation：k-重交叉验证

    把 训练集--验证集合并，划分为k份，顺序使用k-1份训练，1份验证，

![20210812160806.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812160806.png)


## 如何消除过拟合：


![20210812163055.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812163055.png)

![20210812163242.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812163242.png)

how to reduce overfitting
    More date ：更多的数据集
    Constraint model complexity ：合适的模型能力
        shallow
        regularization
    Dropout 
    Data argumentation
    Early Stopping

![20210812163848.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812163848.png)
这里讲一下

regularization： 迫使参数的范数接近于0，减少模型复杂度。

    https://blog.csdn.net/niuniu990/article/details/88567008

![20210812164131.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812164131.png)


regularization的类型：
L1:1范数
L2:2范数

![20210812164244.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812164244.png)

![20210812164852.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812164852.png)

    regularization_loss=0
    for param in model.parameters():
        regularization_loss+=torch.sum(torch.abs(param))  #先求绝对值，再求和

    classify_loss=criteon(logits,target)
    loss=classify_loss+0.01*regularization_loss # L1 regularization

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


![20210812165025.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812165025.png)

    device=torch.device('cuda:0')
    net=MLP().to(device)
    optimizer=optim.SCD(net.parameters(), lr=learning_rate, weight_decay=0.01) # L2 weight_decay
    criteon=nn.CrossEntropyLoss().to(device)


## 动量 momentum 与 学习率衰减 learning rate decay:

![20210812165343.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812165343.png)

动量 momentum:

![20210812170544.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812170544.png)


![20210812170702.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812170702.png)

torch 实现：
![20210812170848.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210812170848.png)


    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.78) #直接指定momentum大小

学习率衰减 learning rate decay:




  









# onnx.js 模型在线学习

    看到的案例：
    如何运行PyTorch模型的浏览器ONNX.js 
    https://www.youtube.com/watch?v=Vs730jsRgO8



# opencv4nodejs：

## 安装，--真是老费劲的
1.首先需要全局的上网,不要绕行ip，不然下载文件什么的，都是问题：这些文件不好下载下来，

    剩下的就是安装
    npm install --global --production windows-build-tools
    npm config set msvs_version 2017
    npm install opencv4nodejs
    其中：直接执行，会自动安装opencv3.4.6版本的

2.可以指定opencv版本，将自动安装的opencv build文件夹复制出来，并在环境变量里面指定

    set OPENCV4NODEJS_DISABLE_AUTOBUILD=1 //不自动安装，自己下载opencv，opencv_contrib并build文件，在设置环境变量。。
    在进行npm install opencv4nodejs 安装

3.examples运行，
    https://github.com/justadudewhohacks/opencv4nodejs 上面examples例子问题，
    他好多是路径设置错误，需要修改：
    1.模块写错了：
        if (!cv.xmodules.face) {
        改成 if (!cv.modules.face) {
        去掉x就对了。。nnd弄了这个长时间
    2.文件路径问题：
        const basePath = '../data/face-recognition';
        要改为const basePath = './data/face-recognition';
        去掉其中的点.


    opencv4nodejs安装，运行例子
    git clone https://github.com/justadudewhohacks/opencv4nodejs.git
    git clone https://github.com.cnpmjs.org/justadudewhohacks/opencv4nodejs.git

    cd opencv4nodejs

    npm i


    成功后
    可以运行examples上面的例子
    https://github.com/Robintjhb/opencv4nodejs_test
    

## 安装问题：
process.dlopen：   
    internal/modules/cjs/loader.js:1025 return process.dlopen(module, path.toNamespacedPath(filename));
     
    通过这个解决：
    https://github.com/justadudewhohacks/opencv4nodejs/issues/125

    js文件运行：console.log(process.env.path);

    cmd下：echo %path%

    上面两个opencv bin 位置结果这两个要相同：



应用案例：

# 人脸识别

    1. 输入图片，显示名字
      看了看opencv4nodejs上面的，与pytorch是两个不同的形式，想想还是用pytorch的来吧：
      pytorch训练模型，
      onnx.js加载模型输出结果，
      在进行3d互动操作
    

    2.打开手机/网页摄像头， 输入图片，显示名字，显示名字
        路线就是 
        opencv截取视频图片，
        pytorch训练模型，
        onnx.js加载模型输出结果
        在进行3d互动操作



