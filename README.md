# 2023-EXP-CV-PlantPathology2021ClassificationTasks



## 环境配置

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



## 模型运行指南

### <font color=purple>训练</font>

```
python main.py --cfg configs/swin_tiny_patch4_window7_224.yaml --local_rank 0 --batch-size 32
```

#### TMUX

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

zip -r output16.zip
```

### <font color=purple>推理</font>

```
python test.py
```

### 可视化

```
pip install grad-cam==1.3.6
```

```
python visual.py
```

可能遇到的报错及解决方案：

[TypeError: __init__() got an unexpected keyword argument 'target_layer' · Issue #176 · jacobgil/pytorch-grad-cam · GitHub](https://github.com/jacobgil/pytorch-grad-cam/issues/176)

[Bug\]vis_cam.py类别激活图可视化时报一个错 TypeError: __call__() got an unexpected keyword argument 'target_category' · Issue #654 · open-mmlab/mmpretrain · GitHub](https://github.com/open-mmlab/mmpretrain/issues/654)

## Swin Transformer 介绍

Swin Transformer是2021年微软研究院发表在ICCV上的一篇文章，问世时在图像分类、目标检测、语义分割多个领域都屠榜。

根据《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》论文摘要所述，Swin Transformer在图像分类数据集ImageNet-1K上取得了87.3%的准确率，在目标检测数据集COCO上取得了58.7%的box AP和51.1%的mask AP，在语义分割数据集ADE20K上去的了53.5%的mIoU。

并且相较于CNN和ViT，Swin-T在多个方面都进行了改进，所以本次作业使用这个模型为基础完成这个分类任务。

### 思想概述

Swin Transformer的思想比较容易理解，如下图所示，ViT(Vision Transformer)的思想是将图片分成16x16大小的patch，每个patch进行注意力机制的计算。而Swin Transformer并不是将所有的图片分成16x16大小的patch，有16x16的，有8x8的，有4x4的。每一个patch作为一个单独的窗口，每一个窗口不再和其它窗口直接计算注意力，而是在自己内部计算注意力，这样就大幅减小了计算量。

![](E:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\vit.png)

为了弥补不同窗口之间的信息传递，Swin Transformer又提出了移动窗口(Shifted Window)的概念(Swin)

### 分块分析

#### 整体架构

![](E:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\total.png)

#### Patch Partion

输入图片尺寸为HxWx3（H = w = 224)，Patch Partion作用就是将图片进行分块。对于每一个Patch，尺寸设定为4x4。然后将所有的Patch在第三维度(颜色通道）上进行叠加，那么经过Patch Partion之后，图片的维度就变成了`[H/4,W/4,4x4x3] = [H/4,W/4,48]`

在模型代码中，Patch embedding也就是Swin Transformer结构图中的Patch Partition与Linear embedding两个模块合并在一起的操作，通过一个卷积层实现，利用4x4，stride为4卷积，将224的图像直接变为`56x56x96`

#### Swin Transformer Block

Swin Transformer Block是Swin Transformer的核心部分。

Swin Transformer Block的输入输出图片维度是不发生变化的，维度的改变依靠的是P-P模块。图中的x2表示，Swin Transformer Block有两个结构。

在右侧小图中，可以看到这两个模块的区别，这两个结构仅有W-MSA和SW-MSA的差别，这两个结构是成对使用的。由LayerNorm层、windowAttention层（Window MultiHead self -attention， W-MSA）、MLP层以及shiftWindowAttention层（SW-MSA）组成。即先经过左边的带有W-MSA的结构再经过右边带有SW-MSA的结构。

#### W-MSA

W-MSA模块就是将特征图划分到一个个窗口(Windows)中，在每个窗口内分别使用多头注意力模块。

代码中的这个模块相较于ViT不加窗口计算全局注意力的MSA，在计算的时间复杂度上面有显著优势，能够大大节省训练时间。原理如图，在此不再赘述。

![](E:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\屏幕截图 2023-11-13 235229.png)

#### <font color=red>SW-MSA</font>

SW-MSA主要是为了让窗口与窗口之间可以发生信息传输。论文中给出了这样一幅图来描述SW-MSA。

![](E:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\45af6be96759d7960c079fda5de60099.png)

表面上看从4个窗口变成了9个窗口，实际上是整个窗口网格从左上角分别向右侧和下方各偏移了M/2个像素（将windows进行半个窗口的循环移位）。但是这样又产生了一个新的问题，那就是每个窗口大小不一样，不利于计算。

于是将左上角的窗口移动到右下角进行合并

以上过程使用torch.roll实现

![](E:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\de89fb1d06167214c4dae5b06fe71488.png)

在相同的窗口中计算自注意力，计算结果如下右图所示，window0的结构保存，但是针对window2的计算，其中3与3、6与6的计算生成了attn mask 中window2中的黄色区域，针对windows2中3与6、6与3之间不应该计算自注意力（attn mask中window2的蓝色区域），将蓝色区域mask赋值为-100，经过softmax之后，起作用可以忽略不计。同理window1与window3的计算一致。

![](E:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\v2-3092a33bdb8a2d6b096f6f7b40ec3b13_720w.jpg)

最后再进行循环移位，恢复原来的位置。

#### Patch Merging

第一个Stage结束之后，后面3个Stage的结构完全一样。和第一个Stage不同的是，后面几个Stage均多了一个Patch Merging的操作。 Patch Merging的操作不难理解，首先是将一个矩阵按间隔提取出四个小矩阵，然后将这四个矩阵在第三通道上进行Concat，在进行LayerNorm之后，通过一个线性层映射成2个通道。这样，通过Patch Merging操作之后的特征图长宽分别减半，通道数翻倍。

![](E:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\bfb1da893466833bcbe0dd8a9d7f7ba8.png)

## 项目代码解析

### 数据集预处理

本数据集来源为

#### plant_dataset

原始数据集为`plant_dataset`文件夹，里面分为了三部分

训练集（`train`）、验证集（`val`）、测试集（`test`）

其中每个文件夹中使用csv文件建立了每张图片的标签索引（在训练集中每个图片的标签不唯一）

#### process.ipynb

这份脚本实现了将原始数据集向目标数据集的转换

首先将各个标签按照OneHot编码进行重新命名，具体规则如下：

```python
label_to_folder = {
    "complex": "100000",
    "frog_eye_leaf_spot": "010000",
    "healthy": "001000",
    "powdery_mildew": "000100",
    "rust": "000010",
    "scab": "000001"
}
```

为了方便模型调用，我将数据集转换成了ImageNet的标准格式。

特别的，对于一个图片有多个标签的情况，这个脚本会将其拷贝多份，分别放入对应的文件夹中，以保证不丢失训练集中的特征。

```
I:.
+---train
|   +---001000
|   +---000001
|   +---010000
|   +---100000
|   +---000100
|   \---000010
+---val
|   +---001000
|   +---000010
|   +---000001
|   +---010000
|   +---000100
|   \---100000
\---test
    +---000100
    +---100000
    +---000001
    +---001000
    +---010000
    \---000010
```

#### OneHot

这个文件夹中保存了数据集的所有图片对应的OneHot编码，以便进行调用、测试。

#### dataset

这个文件夹中就是模型运行的时候实际使用的数据集，是一个ImageNet格式的标准数据集

### 模型调整&运行

#### 获取代码和预训练模型

首先执行以下代码获取原始模型

```
git clone https://github.com/microsoft/Swin-Transformer
```

在`get_start.md`文件中找到预训练模型，下载。这里我们选择下载Swin-Tiny类型的

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\run1.png)

#### 修改config.py文件

找到如下代码行，并根据实际情况对其进行修改

```python
_C.DATA.DATA_PATH = 'dataset'
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME ='swin_tiny_patch4_window7_224.pth'
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 6
```

```python
_C.DATA.DATA_PATH  # 数据集路径的根目录，定义为dataset。

_C.DATA.DATASET  # 数据集的类型，这里只有一种类型imagenet。

_C.MODEL.NAME  # 模型的名字，对应configs下面yaml的名字，会在模型输出的root目录创建对应MODEL.NAME的目录。

_C.MODEL.RESUME  # 预训练模型的目录。

_C.MODEL.NUM_CLASSES  # 模型的类别，默认是1000，按照数据集的类别数量修改。
```

#### 修改build.py

将`nb_classes =1000`改为`nb_classes = config.MODEL.NUM_CLASSES`

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\run2.png)

#### 修改utils.py

由于类别默认是1000，所以加载模型的时候会出现类别对不上的问题，所以需要修改load_checkpoint方法。在加载预训练模型之前增加修改预训练模型的方法：

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\run3.png)

在图片中对应位置添加：

```python
if checkpoint['model']['head.weight'].shape[0] == 1000:
    checkpoint['model']['head.weight'] = torch.nn.Parameter(
        torch.nn.init.xavier_uniform(torch.empty(config.MODEL.NUM_CLASSES, 768)))
    checkpoint['model']['head.bias'] = 
```

#### 修改main.py

将92-94注释，如下图：

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\run4.png)

将312行修改为：torch.distributed.init_process_group('gloo', init_method='file://tmp/somefile', rank=0, world_size=1)

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\run5.png)

#### 运行预训练模型

```
python main.py --cfg configs/swin_tiny_patch4_window7_224.yaml --local_rank 0 --batch-size 16
```

### 推理

这个项目原来没有推理脚本，我自己写了一个

#### 导入包和配置参数

```python
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
from models import build_model
from config import get_config
import argparse

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer Test script', add_help=False)
    parser.add_argument('--cfg', default='configs/swin_tiny_patch4_window7_224.yaml', type=str, metavar="FILE",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', default='output/swin_tiny_patch4_window7_224/default/ckpt_epoch_1.pth',
                        help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--local_rank", default='0', type=int, help='local rank for DistributedDataParallel')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config
```

定义`class`、创建`transform`、将图像`resize`为`224×224`大小、定义类别（OneHot），顺序和数据集对应。

```python
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
classes = ("000001", "000010", "000100", "001000", "010000", "100000")
```

#### 创建模型

```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, config = parse_option()
model = build_model(config)
checkpoint = torch.load('output/swin_tiny_patch4_window7_224/default/ckpt_epoch_1.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
model.to(DEVICE)
```

判断gpu是否可用，如果不可以使用cpu。获取`config`参数。创建模型。从检查点加载训练的模型权重。将权重放入`model`中。

#### 开始推理

```python
# 用于记录准确度的字典
accuracy_dict = {cls: {"correct": 0, "total": 0} for cls in classes}

# 遍历每个类别
for cls in classes:
    cls_path = os.path.join('dataset/test/', cls)
    for file in os.listdir(cls_path):
        file_path = os.path.join(cls_path, file)
        if os.path.isfile(file_path):
            img = Image.open(file_path)
            img = transform_test(img)
            img = img.unsqueeze(0).to(DEVICE)
            out = model(img)
            _, pred = torch.max(out, 1)
            predicted_class = classes[pred.item()]

            # 打印每张图片的预测结果
            print(f"Image: {file}, Actual:{cls}, Predicted: {predicted_class}")

            accuracy_dict[cls]["total"] += 1
            if predicted_class == cls:
                accuracy_dict[cls]["correct"] += 1

# 计算并打印准确度
for cls in classes:
    acc = accuracy_dict[cls]["correct"] / accuracy_dict[cls]["total"] * 100
    print(f"Accuracy for {cls}: {acc:.2f}%")

# 计算总体准确度
total_correct = sum([accuracy_dict[cls]["correct"] for cls in classes])
total = sum([accuracy_dict[cls]["total"] for cls in classes])
overall_accuracy = total_correct / total * 100
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
```

### 可视化

这部分调用了GradCAM库，将注意力具象化为热力图叠加在原始图像上，具体思路如下

#### 构建模型

```python
def parse_option():
    # ...
    return args, config


# 配置网络
args, config = parse_option()
model = build_model(config)
model.to("cpu")
checkpoint = torch.load("/hy-tmp/output/swin_tiny_patch4_window7_224/default/ckpt_epoch_99.pth", map_location=torch.device('cpu'))
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
model.eval()
print(model)
```

#### 获取目标层

```python
# 获取特征图的层
target_layer = [model.layers[-1].blocks[-1].norm1]
```

为了使用Grad-CAM，需要指定模型中的目标层，通常是最后一层卷积层或者与之类似的层。

#### Reshape Transformer

这是Grad-CAM用于处理特殊的结构（如Swin Transformer）的一个关键函数。G-CAM最初应用于CNN的可视化的，在这里使用需要调整一下。

由于Swin Transformer的输出并不是传统CNN中的标准特征图形状，这个函数将输出重新整形为Grad-CAM能够处理的形式。它将张量从其原始形状转换为`(大小, 高度, 宽度, 通道数)`的格式，并重新排列维度以匹配CNN特征图的`(大小, 通道数, 高度, 宽度)`格式。

```python
# transformer特殊需要
def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
```

#### 创建Grad-CAM

使用模型、目标层和特殊的`reshape_transform`函数来创建Grad-CAM对象。

```python
cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform)
```

#### 图像处理与可视化

对于给定的图像路径和类别ID，脚本使用Grad-CAM生成激活映射，这些映射显示了模型在做出决策时关注的图像区域。然后，这些映射被转换为热图并叠加到原始图像上，以可视化模型的注意力焦点。

```python
# 可视化
for image_path, class_id in zip(image_paths, class_ids):
    input_tensor = img_process(image_path)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=class_id)
    
    
    # 确保 grayscale_cam 是二维的，并且其值在 0 到 1 之间
    grayscale_cam = grayscale_cam[0, :]  # 取第一个元素，如果它是一个批处理
    grayscale_cam = np.maximum(grayscale_cam, 0)  # 确保没有负值
    grayscale_cam = grayscale_cam / grayscale_cam.max()  # 归一化

    # 转换为 uint8 并应用颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)

    # 叠加热图
    # 加载原始图像
    rgb_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    
    heatmap = np.float32(heatmap) / 255
    cam_image = heatmap + np.float32(rgb_img / 255)
    cam_image = cam_image / np.max(cam_image)

    # 显示和保存结果
    # plt.imshow(cam_image)
    # plt.show()

    visualization = show_cam_on_image(rgb_img / 255.0, grayscale_cam)

    cv2.imwrite("cam_image.jpg", visualization)
```

## 结果展示

首先对超参数BatchSize进行调参，在`epoch = 100`的情况下测试了从8-128的各值，结果如下：

#### `B_S = 8`

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\bs8-e99.png)

#### `B_S = 16`

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\bs16-e99.png)

#### `B_S = 32`

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\bs32-e99.png)

#### `B_S = 64`

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\bs64-e99.png)

#### `B_S = 128`

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\bs128-e99.png)

增大BatchSize会使得显存的使用量明显增加，在`B_S = 64`以上的情况，显存的使用量已经在12GB以上了；较小的`B_S`可能能够帮助程序跳出局部最优解达到sota；较大的B_S的训练速度较快

综合考虑以上情况和准确度，选择`B_S = 32`进行后续的分析。



## 可视化-类激活热力图

以下图片均在`BatchSize = 32; epoch = 300`的情况下生成。

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\bs32-e300.png)

类激活热力图：用于检查图像哪一部分对模型的最终输出有更大的贡献。具体某个类别对应到图片的那个区域响应最大，也就是对该类别的识别贡献最大。

这部分使用pytorch-grad-cam库帮助实现：https://github.com/jacobgil/pytorch-grad-cam

可视化部分关键的就是要从模型中抽取层进行可视化，关键代码如下：

```python
target_layer = [model.layers[-1].blocks[-1].norm1]
target_layer = [model.layers[-1].blocks[-2].norm1]
```

![](E:\001CV\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\total.png)

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\imgs_for_README\new_total.png)

在 Swin Transformer 中，一个 `BasicLayer` 包含多个 `SwinTransformerBlock` 实例。每个 `SwinTransformerBlock`<font color=purple>（上图最右侧的两个block）</font> 通常包含以下几个主要部分：

1. **LayerNorm (`norm1`)**：第一个规范化层。
2. **Window Attention (`attn`)**：这是 Swin Transformer 的核心，窗口内的自注意力机制。
3. **LayerNorm (`norm2`)**：第二个规范化层。
4. **MLP**：多层感知器，通常包含两个全连接层。

在 Swin Transformer 的设计中，有一个重要的特性是**窗口切换**（window shifting）。这种设计在连续的 `SwinTransformerBlock` 之间交替进行。具体来说：

- 第一个 `SwinTransformerBlock`（例如 `model.layers[-1].blocks[-2]`）进行标准的窗口注意力操作，其中每个窗口内的像素/特征只与同一窗口内的其他像素/特征进行自注意力计算。
- 紧随其后的第二个 `SwinTransformerBlock`（例如 `model.layers[-1].blocks[-1]`）则进行窗口切换。在这个阶段，窗口会在空间上稍微移动，这样之前不在同一窗口中的像素/特征点就可以进行交互。

这种窗口切换策略的目的是为了允许不同窗口间的信息交流，增加模型的表示能力，同时避免了标准自注意力机制的高计算成本。

因此，当选择 `model.layers[-1].blocks[-1].norm1` 与 `model.layers[-1].blocks[-2].norm1` 作为目标层时，实际上是在选择两个相邻的 `SwinTransformerBlock` 中的第一个规范化层。这两个块在窗口注意力机制的执行方式上有所不同，一个是标准窗口注意力，另一个是进行了窗口切换的窗口注意力。

### `layers[-1].blocks[-1].norm2`（最后一层）注意力可视化

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output1.png" style="zoom: 67%;" />

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output2.png" style="zoom:67%;" />

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output3.png" style="zoom:67%;" />

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output4.png" style="zoom:67%;" />



### `blocks[-1].norm2`与`blocks[-2].norm2`对比

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output5.png" style="zoom:80%;" />

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output6.png" style="zoom:80%;" />

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output7.png" style="zoom:80%;" />



### `blocks[-1].norm2`与`blocks[-2].norm1`对比

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output8.png" style="zoom:80%;" />

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output9.png" style="zoom:80%;" />

<img src="E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output10.png" style="zoom:80%;" />



### Attention随epoch增加而精进

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output11.png)

![](E:\001cv\2023-EXP-CV-PlantPathology2021ClassificationTasks\result_img\visual\output12.png)





