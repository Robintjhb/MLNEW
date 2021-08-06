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




         // 设置Python的路径
        "python.pythonPath": "C:/ProgramData/Anaconda3/python",
        // 设置Python的代码格式化
        "python.formatting.provider": "yapf",
        // 设置Python的代码检查
        "python.linting.flake8Path": "pycodestyle",
        "python.linting.flake8Enabled": true,
        "python.autoComplete.extraPaths": [
            "C:/ProgramData/Anaconda3/",
            "C:/ProgramData/Anaconda3/Lib/",
            "C:/ProgramData/Anaconda3/Lib/site-packages/",
            "C:/ProgramData/Anaconda3/DLLs/"
        ],
        "python.autoComplete.addBrackets": true




(6) 安装成功验证：

    print(torch.__version__)  #torch 版本

    print(torch.cuda.is_available())  #是否支持gpu







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



