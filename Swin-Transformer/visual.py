import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import build_model
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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


# 配置网络
args, config = parse_option()
model = build_model(config)
model.to("cpu")
checkpoint = torch.load("/hy-tmp/output/swin_tiny_patch4_window7_224/default/ckpt_epoch_299.pth", map_location=torch.device('cpu'))
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
model.eval()
# print(model)

# 获取特征图的层
target_layer = [model.layers[-1].blocks[-1].norm2]
# [model.blocks[-1].norm1]
# transformer特殊需要
def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# 图像预处理
def img_process(rgb_img_dir):
    img = cv2.imread(rgb_img_dir, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.transpose(2, 0, 1)  # Change data layout from HWC to CHW
    img_tensor = torch.tensor(img[np.newaxis, ...], dtype=torch.float32)
    return img_tensor

# 创建 Grad-CAM 对象
cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform)

# 图像和类别 - 替换为您的图像和类别
# image_paths = ["/Swin-Transformer/dataset/test/010000/839e978ed69e7808.jpg"]  # 替换为您的图像路径
# image_paths = ["/Swin-Transformer/dataset/test/000001/80c5d3895739ff50.jpg"]  # 替换为您的图像路径
# image_paths = ["/Swin-Transformer/dataset/test/000010/84d7fe5c7c0c43c2.jpg"]  # 替换为您的图像路径

image_paths = ["/Swin-Transformer/dataset/test/000100/80bcfd9f60f06307.jpg"]  # 替换为您的图像路径
# image_paths = ["/Swin-Transformer/dataset/train/000001/ec60fc9fc1c11bc1.jpg"]  # 替换为您的图像路径

class_ids = [2]  # 替换为对应的类别索引

# dir_path = "visual/010000-839e978ed69e7808.jpg"
# dir_path = "visual/000001-80c5d3895739ff50.jpg"
dir_path = "visual/000100-80bcfd9f60f06307.jpg"
# dir_path = "visual/double_tag"

os.makedirs(dir_path, exist_ok=True)

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
    
    cv2.imwrite(os.path.join(dir_path, "orignal_img.jpg"), rgb_img)
    
    heatmap = np.float32(heatmap) / 255
    cam_image = heatmap + np.float32(rgb_img / 255)
    cam_image = cam_image / np.max(cam_image)

    # 显示和保存结果
    plt.imshow(cam_image)
    plt.show()
    
    visualization = show_cam_on_image(rgb_img / 255.0, grayscale_cam)

    cv2.imwrite(os.path.join(dir_path, "cam_image.jpg"), visualization)
