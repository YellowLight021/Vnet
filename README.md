# AI-Studio-Vnet论文复现
> 一个好的标题会让你的开源项目发挥更大的价值，想不出好的名字也不用担心，起名时就统一使用AIStudio-xxx做开头吧~

## 项目描述
> Vnet模型，最近在计算机视觉和模式识别方面的研究突出了卷积神经网络(CNNs)的能力，以解决具有挑战性的任务，如分类、分割和目标检测。
> 实现了最先进的性能。这种成功归因于cnn学习原始输入数据的层次表示的能力，而不依赖手工制作的特征。当输入通过网络层进行处理时，
> 产生的特性的抽象级别就会增加。浅层掌握局部信息，而深层掌握局部信息。分割是医学图像分析中一个高度相关的任务。自动描绘感兴趣的器官
> 和结构通常是必要的，以执行任务，如视觉增强，计算机辅助诊断，干预和定量指标提取图像。
> 在这项工作中，我们的目标是分割肺实质 。这是一个具有挑战性的任务，因为在不同的扫描中，由于变形和强度分布的变化，肺实质可以呈现不同的外观
> 。此外，由于场的不均匀性， 还是在治疗计划中，解剖边界的估计需要准确

## 项目结构
> 一目了然的项目结构能帮助更多人了解，目录树以及设计思想都很重要~
```
-imgs              #将luna16数据集中的subset文件解压到该目录下。解压完成后目录下应该有raw，mhd两种格式的文件（我实际训练测试只解压了subset0、1）
-seg-lungs-LUNA16  #将luna16数据集中的seg-lungs-LUNA16解压到该目录下。解压完后目录下应该有zraw，mhd两种格式文件
-README.MD
-requirements      #模型训练时候需要pip install 的包。SimpleITK、scikit-image、setproctitle，也可以自行pip install安装
-train.py          #主体运行代码
-utils.py          #主体运行代码用到的一些函数合集
-luna.py           #dataset处理模块。包括一些数据预处理和训练集、测试集划分。
-vnet.py           #vnet模型的paddle代码实现
```
## 使用方式
> 相信你的Fans已经看到这里了，快告诉他们如何快速上手这个项目吧~  
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/3146256)  
B：可以clone下当前github的repo。(确保本地有paddle2.2.0的框架版本环境)
> 1、到官网下载好luna16数据集，实际代码训练的时候可以只解压部分数据集（所有都解压实在太大了，必要性不大，我只解压了两个subset）
> 2、设置train.py的root路径。将root路径改成当前你自己的路径。主要是为了能让代码读取imgs和seg-lungs的数据。读者也可以自己更改合适的路径
> 3、使用终端cd到当前目录下。直接输入python train.py就能运行了（目前只实验了单卡运行环境，并没有做多卡处理后面有条件可以改动下。
> 4、目前没有写inference的处理留了传参接口， 因为只是为了打比赛达到性能指标，我用train和test处理已经够了。
> 5、目前模型以checkpoint来保存的，每训练一个epoch进行一次test和一次checkpoint。将error_rate最小的保存成了checkpoint_model_best.pth.rar。
> 日志文件和checkpoint文件我都会上传百度网盘
