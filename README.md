# AI-Studio-Vnet论文复现

## 项目描述
![images](images/vnet.png)  
> Vnet模型，最近在计算机视觉和模式识别方面的研究突出了卷积神经网络(CNNs)的能力，以解决具有挑战性的任务，如分类、分割和目标检测。
> 实现了最先进的性能。这种成功归因于cnn学习原始输入数据的层次表示的能力，而不依赖手工制作的特征。当输入通过网络层进行处理时，
> 产生的特性的抽象级别就会增加。浅层掌握局部信息，而深层掌握局部信息。分割是医学图像分析中一个高度相关的任务。自动描绘感兴趣的器官
> 和结构通常是必要的，以执行任务，如视觉增强，计算机辅助诊断，干预和定量指标提取图像。
> 在这项工作中，我们的目标是分割肺实质 。这是一个具有挑战性的任务，因为在不同的扫描中，由于变形和强度分布的变化，肺实质可以呈现不同的外观
> 。此外，由于场的不均匀性， 还是在治疗计划中，解剖边界的估计需要准确

**论文：**

- [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://paperswithcode.com/paper/v-net-fully-convolutional-neural-networks-for)

**项目参考：**
- [https://github.com/mattmacy/vnet.pytorch](https://github.com/mattmacy/vnet.pytorch)


##快速开始
###第一步：克隆本项目
git clone https://github.com/YellowLight021/Vnet
cd Vnet

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

###第二步：安装第三方库
pip install -r requirements.txt
#### 代码结构与说明



###第三步：数据集下载
[Luna](https://luna16.grand-challenge.org/Data/) （也可以在baiduaistudio公开数据集进行下载，数据集比较大只需要使用3个subset数据集）

1、将Luna数据集中的subset解压到imgs里面（我解压了前3个subset作为输入data）

2、Luna数据集中的seg-lungs-LUNA16是分割的label文件

###第四步：模型训练

python -m paddle.distributed.launch train.py --nEpochs 300 

模型每训练一个epoch会进行一次validation，训练了95个epoch的时候dice达到98.5%
训练日志在log文件夹下，模型参数在链接：https://pan.baidu.com/s/1aI3CrxmypVLCLPJAUTzPpg 
提取码：qyvl 

###第五步：评估指标对齐

python metric_align.py 发现评估指标和参考代码指标能够对齐





## 在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/clusterprojectdetail/3432461/trainTask) 
1、直接使用aistudio上的数据集(https://aistudio.baidu.com/aistudio/datasetdetail/119677)，https://aistudio.baidu.com/aistudio/datasetdetail/1860

2、在脚本任务环境下直接提交就可以了run.py文件会自动执行数据集解压缩unzip.py，和训练操作train.py操作

3、目前模型以checkpoint来保存的，每训练一个epoch进行一次test和一次checkpoint。将error_rate最小的保存成了checkpoint_model_best.pth.rar。

4、metric_align.py是metric对齐脚本，测试了一下和官方给到的dice计算是能够对齐的
