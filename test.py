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

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# classes = ("100000", "010000", "001000", "000100", "000010", "000001")

classes = ("000001", "000010", "000100", "001000", "010000", "100000")


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, config = parse_option()
model = build_model(config)
checkpoint = torch.load('/hy-tmp/output/swin_tiny_patch4_window7_224/default/ckpt_epoch_99.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
model.to(DEVICE)



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

