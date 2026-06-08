#!/usr/bin/env python3
"""
BiRefNet 抠图工具 — 拖入图片 → 双击脚本 → 自动输出透明背景PNG。

使用 BiRefNet 深度学习模型进行高精度前后景分离。
支持 GPU 自动加速（CUDA），CPU 回退。
适合角色立绘、物件抠图等需要高质量透明背景输出的场景。

用法：
  1. 把 PNG/JPG 图片放入 input/ 文件夹（支持子文件夹）
  2. 运行：python birefnet_matting.py
  3. 结果输出到 output/ 文件夹

依赖安装：
  pip install torch torchvision numpy opencv-python timm kornia einops huggingface-hub pillow tqdm
"""

import os
import sys
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# 国内用户: 使用 HF 镜像加速下载 (设为空字符串直连)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ============ 配置 ============
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# 模型输入尺寸 (BiRefNet 默认 1024，越大越精细但越吃显存)
MODEL_SIZE = (1024, 1024)

# 是否启用前景精修 (修复边缘颜色渗漏，稍慢但质量更好)
REFINE_FOREGROUND = True
REFINE_RADIUS = 90  # 精修模糊半径

# 边缘羽化半径 (px，消除硬边缘)
FEATHER_RADIUS = 1

# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet 归一化参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ============ 模型加载 ============

def load_model():
    """加载 BiRefNet 预训练模型 (从 HuggingFace Hub 自动下载)。"""
    print(f"[设备] {DEVICE}")
    if DEVICE == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)} | 显存: {torch.cuda.get_device_properties(0).total_mem // 1024**3} GB")

    # 添加 BiRefNet 到 path (如果本脚本不在 BiRefNet 目录)
    _ensure_birefnet_import()

    from models.birefnet import BiRefNet

    print("[模型] 正在加载 BiRefNet...")
    try:
        model = BiRefNet.from_pretrained("ZhengPeng7/BiRefNet")
        print("[模型] 从 HuggingFace Hub 加载成功")
    except Exception as e:
        print(f"[警告] HF Hub 加载失败 ({e})")
        print("[模型] 尝试从本地权重加载...")
        model = _load_from_local_ckpt()

    model = model.to(DEVICE)
    model.eval()
    print("[模型] 加载完成，准备就绪")
    return model


def _ensure_birefnet_import():
    """确保 BiRefNet 源码在 sys.path 中。"""
    birefnet_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "BiRefNet"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "BiRefNet"),
    ]
    for p in birefnet_paths:
        p = os.path.normpath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return
    print("[警告] 找不到 BiRefNet 源码目录，请确认已克隆到正确位置")


def _load_from_local_ckpt():
    """回退方案：从本地 .pth 文件加载。"""
    from models.birefnet import BiRefNet
    from utils import check_state_dict

    model = BiRefNet(bb_pretrained=False)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "BiRefNet", "ckpts")
    ckpt_dirs = sorted([d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]) if os.path.isdir(ckpt_dir) else []

    if ckpt_dirs:
        ckpt_folder = os.path.join(ckpt_dir, ckpt_dirs[-1])
        pth_files = sorted([f for f in os.listdir(ckpt_folder) if f.endswith('.pth')])
        if pth_files:
            ckpt_path = os.path.join(ckpt_folder, pth_files[-1])
            print(f"[模型] 加载本地权重: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            return model

    raise FileNotFoundError(
        "无法加载 BiRefNet 模型。请：\n"
        "  1. 联网后自动从 HuggingFace 下载，或\n"
        "  2. 在 BiRefNet/ckpts/ 下放置预训练 .pth 文件\n"
        "  下载地址: https://huggingface.co/ZhengPeng7/BiRefNet"
    )


# ============ 前景精修 ============

def refine_foreground_cpu(image, alpha, r=90):
    """CPU 版前景精修 (FB-Blur-Fusion)，修复边缘颜色渗漏。"""
    image_np = np.array(image, dtype=np.float32) / 255.0
    alpha_np = np.array(alpha, dtype=np.float32) / 255.0

    blurred_alpha = _cv_blur(alpha_np, r)[:, :, np.newaxis]
    FG = image_np

    blurred_FGA = _cv_blur(FG * alpha_np[:, :, np.newaxis], r)
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = _cv_blur(FG * (1 - alpha_np[:, :, np.newaxis]), r)
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)

    FG_refined = blurred_FG + alpha_np[:, :, np.newaxis] * (
        image_np - alpha_np[:, :, np.newaxis] * blurred_FG - (1 - alpha_np[:, :, np.newaxis]) * blurred_B
    )
    FG_refined = np.clip(FG_refined, 0, 1)

    # 第二遍 (小半径)
    alpha_np2 = alpha_np
    blurred_alpha2 = _cv_blur(alpha_np2, 6)[:, :, np.newaxis]
    blurred_FGA2 = _cv_blur(FG_refined * alpha_np2[:, :, np.newaxis], 6)
    blurred_FG2 = blurred_FGA2 / (blurred_alpha2 + 1e-5)
    blurred_B1A2 = _cv_blur(blurred_B * (1 - alpha_np2[:, :, np.newaxis]), 6)
    blurred_B2 = blurred_B1A2 / ((1 - blurred_alpha2) + 1e-5)
    FG_final = blurred_FG2 + alpha_np2[:, :, np.newaxis] * (
        image_np - alpha_np2[:, :, np.newaxis] * blurred_FG2 - (1 - alpha_np2[:, :, np.newaxis]) * blurred_B2
    )
    FG_final = np.clip(FG_final, 0, 1)

    return Image.fromarray((FG_final * 255).astype(np.uint8))


def _cv_blur(img, r):
    """简单的 box blur (避免依赖 OpenCV)。"""
    import scipy.ndimage as ndi
    if img.ndim == 2:
        return ndi.uniform_filter(img, size=r)
    else:
        result = np.zeros_like(img)
        for c in range(img.shape[-1]):
            result[:, :, c] = ndi.uniform_filter(img[:, :, c], size=r)
        return result


# ============ 核心推理 ============

@torch.no_grad()
def process_image(model, img_path, output_path):
    """处理单张图片：推理 mask → 合成 RGBA → 保存。"""
    # 加载原图
    orig = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig.size

    # 预处理
    transform = transforms.Compose([
        transforms.Resize(MODEL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    input_tensor = transform(orig).unsqueeze(0).to(DEVICE)

    # 推理
    if DEVICE == "cuda":
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            preds = model(input_tensor)
    else:
        preds = model(input_tensor)

    # 取最后一层输出 → sigmoid → mask
    mask = preds[-1].sigmoid().squeeze().cpu().numpy()

    # 缩放到原图尺寸
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize((orig_w, orig_h), Image.LANCZOS)

    # 前景精修
    if REFINE_FOREGROUND:
        orig = refine_foreground_cpu(orig, mask_img, r=REFINE_RADIUS)

    # 合成 RGBA
    rgba = orig.convert("RGBA")
    alpha = mask_img

    # 边缘羽化
    if FEATHER_RADIUS > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(FEATHER_RADIUS))

    rgba.putalpha(alpha)
    rgba.save(output_path, "PNG")


# ============ 批量处理 ============

def get_image_files(folder):
    """获取文件夹内所有图片（支持子目录）。"""
    if not os.path.isdir(folder):
        return []
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp", ".bmp"}
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            if any(f.endswith(ext) for ext in exts):
                files.append(os.path.join(root, f))
    return sorted(files)


def process_folder(model, input_dir, output_dir):
    """批量处理文件夹内所有图片。"""
    files = get_image_files(input_dir)
    if not files:
        print(f"错误: {input_dir}/ 下未找到图片文件 (png/jpg/jpeg/webp/bmp)")
        return

    print(f"检测到 {len(files)} 张图片")
    print(f"模型尺寸: {MODEL_SIZE[0]}x{MODEL_SIZE[1]} | 前景精修: {'开' if REFINE_FOREGROUND else '关'}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    for i, src_path in enumerate(tqdm(files, desc="抠图进度", unit="张")):
        # 保持子目录结构
        rel_path = os.path.relpath(src_path, input_dir)
        dst_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            process_image(model, src_path, dst_path)
        except Exception as e:
            print(f"\n[跳过] {rel_path}: {e}")
            continue

    print(f"\n完成！输出目录: {os.path.abspath(output_dir)}")


# ============ 入口 ============

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  BiRefNet 抠图工具")
    print("  模型: BiRefNet (Swin-T backbone)")
    print(f"  设备: {DEVICE.upper()}")
    print("=" * 60)
    print()

    model = load_model()
    print()

    process_folder(model, INPUT_DIR, OUTPUT_DIR)

    print()
    print("处理完成，可以关闭窗口。")


if __name__ == "__main__":
    main()
