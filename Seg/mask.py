import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from models.model_zoo import get_segmentation_model  # 使用与训练代码相同的模型加载方式


def load_model(checkpoint_path):
    """完全对齐eval.py的模型加载逻辑"""
    # 创建与eval.py完全一致的参数配置
    class Args:
        model = 'deeplabv3'
        backbone = 'resnet18'
        aux = True
        pretrained_base = 'None'  # 必须为字符串
        local_rank = 0
        norm_layer = torch.nn.BatchNorm2d

    args = Args()
    
    # 严格复制eval.py的模型构建方式
    model = get_segmentation_model(
        model=args.model,
        backbone=args.backbone,
        aux=args.aux,
        pretrained=checkpoint_path,  # 直接加载用户权重
        pretrained_base=args.pretrained_base,
        norm_layer=args.norm_layer,
        num_class=150  # ADE20K类别数
    )
    
    # 处理多GPU权重前缀
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = {
        k.replace('module.model.', '')
         .replace('module.', '')
         .replace('model.', ''): v
        for k, v in state_dict.items()
    }
    
    # 关键！先验证backbone结构再加载权重
    print("验证模型结构...")  # 调试信息
    print("Backbone类型:", type(model.backbone))  # 应为nn.Module
    
    # 加载权重并验证
    model.load_state_dict(new_state_dict, strict=True)
    print("Backbone验证通过:", hasattr(model.backbone, 'conv1'))  # 应为True
    
    return model.eval().cuda()

def visualize_segmentation(img_path, alpha=0.6):
    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    orig_img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(orig_img).unsqueeze(0).cuda()
    
    # 模型推理
    model = load_model('work_dir/ade20k/3g/kd_deeplabv3_resnet18_ade20k_best_model.pth')
    
    # 模型推理
    with torch.no_grad():
        outputs = model(img_tensor)
        pred_mask = torch.argmax(outputs[0].squeeze(), dim=0).cpu().numpy()
    palette = np.random.randint(0, 255, (150, 3), dtype=np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image', fontsize=14)
    
    axes[1].imshow(palette[pred_mask])
    axes[1].set_title('Semantic Mask', fontsize=14)
    
    axes[2].imshow(orig_img)
    axes[2].imshow(palette[pred_mask], alpha=alpha)
    axes[2].set_title(f'Blended Result (α={alpha})', fontsize=14)
    
    [ax.axis('off') for ax in axes]
    plt.savefig('result.jpg', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--alpha', type=float, default=0.6)
    args = parser.parse_args()
    
    visualize_segmentation(args.input, args.alpha)