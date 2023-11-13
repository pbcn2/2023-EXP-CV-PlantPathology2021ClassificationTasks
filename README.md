# 2023-EXP-CV-PlantPathology2021ClassificationTasks

## 配置环境

<font color=red>Attention！</font>

```
操作系统:  Linux 18.04
Pytorch:  1.7.1
CUDA:     10.1.243 版本（使用 nvcc --version 查看）
GPU：     显存8G
```

在根目录下上载`Swin-Transformer-hw.zip`并解压 （`unzip Swin-Transformer-hw.zip`）

```
cd Swin-Transformer
```

### 安装apex

在`Swin-Transformer`目录下上载`apex-22.04-dev.zip`并解压生成`apex-22.04-dev`文件夹

将`apex-22.04-dev`重命名为`apex`

```
$ cd apex
$ pip install -r requirements.txt
$ pip install -v --no-cache-dir ./
```

```
$ cd ../
$ python
>>> import apex
>>> from apex import amp
```

当两个库都能够正常import的时候就证明成功安装了

当运行时出现`AssertionError: amp not installed!`的时候意味着没有安装成功，可以检查一下apex版本是否正确、文件夹放置位置是否正确。

当出现以下语句的时候意味着apex版本不正确，只有使用22.04才可以正常运行

```
Attr ibuteError: module ' torch.distributed' has no attribute '_ reduce_ scatter_base‘
或者是
AttributeError: module 'torch.distributed' has no attribute '_all_gather_base' 
```

### 安装timm

```
pip install timm==0.3.2
```

### 安装其他包

```
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

## <font color=purple>训练</font>

```
python main.py --cfg configs/swin_tiny_patch4_window7_224.yaml --local_rank 0 --batch-size 16
```

### TMUX

因为训练时间过长，所以使用TMUX将训练终端放在后台运行，具体需要用到的指令如下：

创建新会话

```
tmux new -s st
```

执行 `tmux detach` 命令，或使用快捷键 Ctrl + B，再按 D 来退出会话

使用 `tmux ls` 命令可以查看当前所有的会话

需要恢复会话时，使用 `tmux a -t <session-name>`，重新进入之前的会话中，如进入刚才名称为 session1 的会话：

```
tmux a -t st
```

## <font color=purple>推理</font>

```
python test.py
```

## 可视化

[【精选】类别激活热力图grad-cam(pytorch)实战跑图_类激活热力图_半甜田田的博客-CSDN博客](https://blog.csdn.net/rensweet/article/details/123263812)

[TypeError: __init__() got an unexpected keyword argument 'target_layer' · Issue #176 · jacobgil/pytorch-grad-cam · GitHub](https://github.com/jacobgil/pytorch-grad-cam/issues/176)

[[Bug\]vis_cam.py类别激活图可视化时报一个错 TypeError: __call__() got an unexpected keyword argument 'target_category' · Issue #654 · open-mmlab/mmpretrain · GitHub](https://github.com/open-mmlab/mmpretrain/issues/654)

`pip install grad-cam==1.3.6`

https://github.com/jacobgil/pytorch-grad-cam



## Swin Transformer

Swin Transformer是2021年微软研究院发表在ICCV上的一篇文章，问世时在图像分类、目标检测、语义分割多个领域都屠榜。

根据《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》论文摘要所述，Swin Transformer在图像分类数据集ImageNet-1K上取得了87.3%的准确率，在目标检测数据集COCO上取得了58.7%的box AP和51.1%的mask AP，在语义分割数据集ADE20K上去的了53.5%的mIoU。

并且相较于CNN和ViT，Swin-T在多个方面都进行了改进，所以本次作业使用这个模型为基础完成这个分类任务。

### 思想概述

Swin Transformer的思想比较容易理解，如下图所示，ViT(Vision Transformer)的思想是将图片分成16x16大小的patch，每个patch进行注意力机制的计算。而Swin Transformer并不是将所有的图片分成16x16大小的patch，有16x16的，有8x8的，有4x4的。每一个patch作为一个单独的窗口，每一个窗口不再和其它窗口直接计算注意力，而是在自己内部计算注意力，这样就大幅减小了计算量。

![](I:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\vit.png)

为了弥补不同窗口之间的信息传递，Swin Transformer又提出了移动窗口(Shifted Window)的概念(Swin)

### 分块分析

#### 整体架构



![](I:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\total.png)

#### Patch Partion

输入图片尺寸为HxWx3（H = w = 224)，Patch Partion作用就是将图片进行分块。对于每一个Patch，尺寸设定为4x4。然后将所有的Patch在第三维度(颜色通道）上进行叠加，那么经过Patch Partion之后，图片的维度就变成了`[H/4,W/4,4x4x3] = [H/4,W/4,48]`

在模型代码中，Patch embedding也就是Swin Transformer结构图中的Patch Partition与Linear embedding两个模块合并在一起的操作，通过一个卷积层实现，利用4x4，stride为4卷积，将224的图像直接变为`56x56x96`

#### Swin Transformer Block

Swin Transformer Block是Swin Transformer的核心部分。

Swin Transformer Block的输入输出图片维度是不发生变化的，维度的改变依靠的是P-P模块。图中的x2表示，Swin Transformer Block有两个结构。

在右侧小图中，可以看到这两个模块的区别，这两个结构仅有W-MSA和SW-MSA的差别，这两个结构是成对使用的。由LayerNorm层、windowAttention层（Window MultiHead self -attention， W-MSA）、MLP层以及shiftWindowAttention层（SW-MSA）组成。即先经过左边的带有W-MSA的结构再经过右边带有SW-MSA的结构。

