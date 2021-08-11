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

dim,keepdim:维度，求max，argmax

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

多层网络求导：

![20210811173052.png](https://raw.githubusercontent.com/Robintjhb/mypicgoformd/main/img/20210811173052.png)


https://www.bilibili.com/video/BV1U54y1E7i6?p=44&spm_id_from=pageDriver
05：50



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



