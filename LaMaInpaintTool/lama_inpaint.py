#!/usr/bin/env python3
"""
LaMa AI 图像修复工具 — 拖入图片+遮罩 → 双击脚本 → 自动擦除/补全。

使用 LaMa (Large Mask Inpainting) 深度学习模型进行高质量图像修复。
擦水印、去杂物、修穿帮、补缺失区域。

用法：
  1. 把原图（如 photo.png）和遮罩（photo_mask.png）放入 input/ 文件夹
     遮罩规则：白色区域 = 需要擦除/修复的位置
  2. 运行：python lama_inpaint.py
  3. 结果输出到 output/ 文件夹

遮罩制作：
  - 用任意画图工具在原图上涂白要删除的区域
  - 或单独创建同名 _mask.png 黑白图
  - 如果没有遮罩文件，脚本会跳过该图片

依赖：onnxruntime（已安装）
模型自动下载 (~170MB，仅首次)
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from tqdm import tqdm


# ============ 配置 ============
INPUT_DIR = "input"
OUTPUT_DIR = "output"
WEIGHTS_DIR = "weights"

MODEL_NAME = "lama_fp32.onnx"
MODEL_REPO = "Carve/LaMa-ONNX"

# 处理分辨率 (LaMa ONNX 模型固定输入尺寸)
MODEL_INPUT_SIZE = 512

# 设备 (CPU / CUDA)
PROVIDERS = ["CPUExecutionProvider"]


def download_model():
    """下载 LaMa ONNX 模型（如不存在）。"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_path = os.path.join(WEIGHTS_DIR, MODEL_NAME)
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) // 1024 // 1024
        print(f"[模型] {MODEL_NAME} 已存在 ({size_mb}MB)")
        return model_path

    print(f"[模型] 从 HF 镜像下载 {MODEL_NAME} (~170MB)...")
    import os as _os
    _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(MODEL_REPO, MODEL_NAME, local_dir=WEIGHTS_DIR)
    print(f"[模型] 下载完成")
    return path


# ============ ONNX 推理 ============

def load_session(model_path):
    """加载 ONNX Runtime 会话。"""
    import onnxruntime as ort
    print(f"[ONNX] 可用执行器: {ort.get_available_providers()}")
    session = ort.InferenceSession(model_path, providers=PROVIDERS)
    input_name = session.get_inputs()[0].name
    mask_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    return session, input_name, mask_name, output_name


def prepare_inputs(image, mask, target_size=MODEL_INPUT_SIZE):
    """
    准备 ONNX 输入。
    - image: PIL RGB
    - mask: PIL L (白色=擦除区域)
    所有输入统一缩放到 target_size x target_size。
    返回: image_np (1,3,S,S), mask_np (1,1,S,S)
    """
    orig_size = image.size  # (w, h)

    # 确保 image 和 mask 尺寸一致
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.LANCZOS)

    # 统一缩放到目标尺寸
    image = image.resize((target_size, target_size), Image.BICUBIC)
    mask = mask.resize((target_size, target_size), Image.LANCZOS)

    img_np = np.array(image, dtype=np.float32) / 255.0
    mask_np = np.array(mask, dtype=np.float32) / 255.0

    # 转 NCHW
    img_tensor = img_np.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, S, S)
    mask_tensor = mask_np[np.newaxis, np.newaxis, ...]        # (1, 1, S, S)

    return img_tensor.astype(np.float32), mask_tensor.astype(np.float32), orig_size


def inpaint_image(session, input_name, mask_name, output_name, img_path, mask_path, output_path):
    """处理单张图片：AI 修复 → 保存。"""
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    start = time.time()
    img_tensor, mask_tensor, orig_size = prepare_inputs(image, mask)

    result = session.run([output_name], {input_name: img_tensor, mask_name: mask_tensor})[0]
    elapsed = time.time() - start

    # 转回 PIL
    result = result[0].transpose(1, 2, 0)  # CHW -> HWC
    result = np.clip(result, 0, 1)
    result_img = Image.fromarray((result * 255).astype(np.uint8))

    # 还原到原始尺寸 (使用高质量插值)
    result_img = result_img.resize(orig_size, Image.LANCZOS)

    result_img.save(output_path, "PNG")
    orig_w, orig_h = orig_size
    print(f"  {orig_w}x{orig_h} | {elapsed:.1f}s")


# ============ 批量处理 ============

def find_image_pairs(input_dir):
    """查找原图+遮罩配对。"""
    if not os.path.isdir(input_dir):
        return []
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp", ".bmp"}
    pairs = []
    all_files = set()

    for root, dirs, filenames in os.walk(input_dir):
        for f in filenames:
            all_files.add(os.path.join(root, f))

    for fpath in sorted(all_files):
        name, ext = os.path.splitext(fpath)
        # 跳过遮罩文件
        if name.endswith("_mask"):
            continue
        # 查找对应遮罩
        mask_path = name + "_mask" + ext
        if os.path.exists(mask_path):
            pairs.append((fpath, mask_path))
        # 也查找 png 遮罩对应 jpg 等
        else:
            found = False
            for mask_ext in [".png", ".PNG"]:
                mp = name + "_mask" + mask_ext
                if os.path.exists(mp):
                    pairs.append((fpath, mp))
                    found = True
                    break

    return pairs


def process_folder(session, input_name, mask_name, output_name, input_dir, output_dir):
    """批量处理文件夹内所有图片。"""
    pairs = find_image_pairs(input_dir)
    if not pairs:
        print(f"错误: 未找到图片+遮罩配对")
        print(f"  请将图片和遮罩放入 {input_dir}/")
        print(f"  例如: photo.png + photo_mask.png （遮罩白色=擦除区域）")
        return

    print(f"检测到 {len(pairs)} 组图片+遮罩")
    print(f"模型输入尺寸: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} (自动缩放)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    for i, (img_path, mask_path) in enumerate(tqdm(pairs, desc="修复进度", unit="组")):
        rel_img = os.path.relpath(img_path, input_dir)
        dst_path = os.path.join(output_dir, os.path.splitext(rel_img)[0] + ".png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            inpaint_image(session, input_name, mask_name, output_name, img_path, mask_path, dst_path)
        except Exception as e:
            print(f"\n[跳过] {os.path.basename(img_path)}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n完成！输出目录: {os.path.abspath(output_dir)}")


# ============ 入口 ============

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  LaMa AI 图像修复工具")
    print("  擦水印 / 去杂物 / 修穿帮 / 补缺失")
    print("=" * 60)
    print()

    model_path = download_model()
    session, input_name, mask_name, output_name = load_session(model_path)
    print()

    process_folder(session, input_name, mask_name, output_name, INPUT_DIR, OUTPUT_DIR)

    print()
    print("处理完成，可以关闭窗口。")


if __name__ == "__main__":
    main()
