#!/usr/bin/env python3
"""
Real-ESRGAN 超分辨率放大工具 — 拖入图片 → 双击脚本 → 自动输出高清图片。

使用 Real-ESRGAN x4plus 模型进行 AI 超分辨率放大。
支持 GPU 自动加速（CUDA），CPU 回退。
适合游戏素材、立绘、像素图放大。

用法：
  1. 把 PNG/JPG 图片放入 input/ 文件夹（支持子文件夹）
  2. 运行：python realesrgan_upscale.py
  3. 结果输出到 output/ 文件夹

模型自动下载 (~67MB，仅首次)
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


# ============ 配置 ============
INPUT_DIR = "input"
OUTPUT_DIR = "output"
WEIGHTS_DIR = "weights"

# 模型选择 (自动下载)
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_NAME = "RealESRGAN_x4plus.pth"

# 放大倍数 (2/3/4，默认 4)
SCALE = 4

# 分块大小 (显存不足时调小，如 256)
TILE_SIZE = 512
TILE_PAD = 32

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============ 模型加载 ============

def download_model():
    """下载模型权重（如不存在）。"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_path = os.path.join(WEIGHTS_DIR, MODEL_NAME)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 10_000_000:
        print(f"[模型] {MODEL_NAME} 已存在 ({os.path.getsize(model_path)//1024//1024}MB)")
        return model_path

    print(f"[模型] 下载 {MODEL_NAME} (~67MB)...")
    import urllib.request
    def _report(count, block_size, total_size):
        if count % 100 == 0:
            pct = count * block_size * 100 / total_size
            print(f"\r  下载进度: {pct:.0f}%", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, model_path, _report)
    print("\r  下载完成!              ")
    return model_path


def load_model():
    """加载 Real-ESRGAN 模型。"""
    print(f"[设备] {DEVICE}")
    if DEVICE == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)} | 显存: {torch.cuda.get_device_properties(0).total_mem // 1024**3} GB")

    model_path = download_model()

    print("[模型] 加载中...")
    from spandrel import ModelLoader
    model = ModelLoader().load_from_file(model_path)
    model = model.to(DEVICE)
    model.eval()
    print(f"[模型] 加载完成 (scale={model.scale})")
    return model


# ============ 核心推理 ============

def pad_tensor(tensor, tile_size, pad):
    """Padding to divisible size."""
    _, _, h, w = tensor.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, (h, w)


def tile_process(model, tensor, tile_size=512, tile_pad=32):
    """分块推理，避免显存溢出。"""
    batch, channel, height, width = tensor.shape
    output_height = height * model.scale
    output_width = width * model.scale
    output = tensor.new_zeros((batch, channel, output_height, output_width))

    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size

    for y in range(tiles_y):
        for x in range(tiles_x):
            # 计算 tile 范围（含 padding）
            x_start = max(0, x * tile_size - tile_pad)
            y_start = max(0, y * tile_size - tile_pad)
            x_end = min(width, (x + 1) * tile_size + tile_pad)
            y_end = min(height, (y + 1) * tile_size + tile_pad)

            tile = tensor[:, :, y_start:y_end, x_start:x_end]

            with torch.no_grad():
                tile_out = model(tile)

            # 计算输出中的有效范围
            out_x_start = x * tile_size * model.scale
            out_y_start = y * tile_size * model.scale
            out_x_end = min(output_width, (x + 1) * tile_size * model.scale)
            out_y_end = min(output_height, (y + 1) * tile_size * model.scale)

            # 裁剪 padding 后的有效区域
            valid_x_start = (x * tile_size - x_start) * model.scale
            valid_y_start = (y * tile_size - y_start) * model.scale
            valid_x_end = valid_x_start + (out_x_end - out_x_start)
            valid_y_end = valid_y_start + (out_y_end - out_y_start)

            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = \
                tile_out[:, :, valid_y_start:valid_y_end, valid_x_start:valid_x_end]

    return output


@torch.no_grad()
def upscale_image(model, img_path, output_path):
    """处理单张图片：超分放大 → 保存。"""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # 转为 tensor
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

    # 分块推理
    start = time.time()
    if TILE_SIZE > 0 and (orig_w > TILE_SIZE or orig_h > TILE_SIZE):
        output = tile_process(model, img_tensor, TILE_SIZE, TILE_PAD)
    else:
        output = model(img_tensor)
    elapsed = time.time() - start

    # 转回 PIL
    output = output.squeeze(0).clamp(0, 1)
    result = transforms.ToPILImage()(output.cpu())
    result.save(output_path, "PNG")

    new_w, new_h = result.size
    print(f"  {orig_w}x{orig_h} -> {new_w}x{new_h} | {elapsed:.1f}s")


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
        print(f"错误: {input_dir}/ 下未找到图片文件")
        return

    print(f"检测到 {len(files)} 张图片")
    print(f"放大倍数: {SCALE}x | 分块: {TILE_SIZE}px")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    for i, src_path in enumerate(tqdm(files, desc="放大进度", unit="张")):
        rel_path = os.path.relpath(src_path, input_dir)
        dst_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            upscale_image(model, src_path, dst_path)
        except Exception as e:
            print(f"\n[跳过] {rel_path}: {e}")
            continue

    print(f"\n完成！输出目录: {os.path.abspath(output_dir)}")


# ============ 入口 ============

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Real-ESRGAN 超分辨率放大工具")
    print(f"  模型: {MODEL_NAME} ({SCALE}x)")
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
