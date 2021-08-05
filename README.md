# MLNEW
重新学习机器学习~加油
保持好，学习的激情与深度，加油

# Pytorch
昨天很高兴，看到了这个，现在重新学习机器学习方面或者说人工智能方面的吧，通过不断的锤炼，做出好用的产品

# pytorch初步：




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

## 问题：
process.dlopen：   
    internal/modules/cjs/loader.js:1025 return process.dlopen(module, path.toNamespacedPath(filename));
     
    通过这个解决：
    https://github.com/justadudewhohacks/opencv4nodejs/issues/125

    js文件运行：console.log(process.env.path);

    cmd下：echo %path%

    上面两个opencv bin 位置结果这两个要相同：
