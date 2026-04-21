#!/usr/bin/env python3
"""
序列帧图集生成工具 - 将PNG序列帧按网格排列组合成一张大图。

用法：
  1. 把PNG序列帧放入 input/ 文件夹（或子文件夹）
  2. 运行：python auto_generate_spritesheet.py
  3. 结果输出到 output/ 文件夹
"""

import os
import re
import math
from PIL import Image


INPUT_DIR = "input"
OUTPUT_DIR = "output"
COLS = 4  # 每行帧数（偏好值，脚本会自动调整以适配纹理限制）
MAX_TEXTURE_SIZE = 8192  # GPU 纹理尺寸上限（PC 通常 8192，高端卡 16384）


def _natural_sort_key(s: str) -> list:
    """自然排序键：将字符串按文本和数字分段，数字按数值排序。"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", s)]


def get_png_files(folder: str) -> list[str]:
    """返回文件夹中所有PNG文件（按自然排序）。"""
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    return sorted(files, key=_natural_sort_key)


def generate_spritesheet(input_dir: str, output_dir: str, name: str = "spritesheet") -> None:
    """将单个文件夹内的PNG序列帧组合成图集。"""
    files = get_png_files(input_dir)
    if not files:
        print(f"错误: {input_dir}/ 下未找到PNG文件")
        return

    print(f"检测到 {len(files)} 张PNG")

    # 加载所有帧
    frames = []
    for fname in files:
        img = Image.open(os.path.join(input_dir, fname)).convert("RGBA")
        frames.append(img)

    # 计算单元格尺寸（取最大宽高）
    cell_w = max(img.width for img in frames)
    cell_h = max(img.height for img in frames)

    # 自动计算列数，确保整张图不超过 GPU 纹理限制
    total = len(frames)
    max_cols = max(1, MAX_TEXTURE_SIZE // cell_w)
    max_rows = max(1, MAX_TEXTURE_SIZE // cell_h)

    if cell_w > MAX_TEXTURE_SIZE or cell_h > MAX_TEXTURE_SIZE:
        print(f"错误: 单帧尺寸 {cell_w}x{cell_h} 超过纹理上限 {MAX_TEXTURE_SIZE}，无法生成图集")
        return

    cols = min(COLS, max_cols)
    while cols <= max_cols:
        rows = math.ceil(total / cols)
        if rows <= max_rows:
            break
        cols += 1
    else:
        print(f"错误: 即使最大列数 {max_cols} 也无法容纳 {total} 帧（单帧 {cell_w}x{cell_h}）")
        return

    if cols != COLS:
        print(f"  自动调整列数: {COLS} -> {cols}（适配纹理限制 {MAX_TEXTURE_SIZE}）")

    # 创建大图（透明背景）
    sheet_w = cols * cell_w
    sheet_h = rows * cell_h
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

    # 逐帧粘贴（居中放置）
    for i, img in enumerate(frames):
        row = i // cols
        col = i % cols
        x = col * cell_w + (cell_w - img.width) // 2
        y = row * cell_h + (cell_h - img.height) // 2
        sheet.paste(img, (x, y), img)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.png")
    sheet.save(output_path)
    print(f"图集已生成: {output_path} ({sheet_w}x{sheet_h})")
    print(f"  单元格: {cell_w}x{cell_h}, 网格: {cols}x{rows}, 总帧数: {total}")


def main() -> None:
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 情况1：根目录直接有PNG文件
    files = get_png_files(input_dir)
    if files:
        generate_spritesheet(input_dir, output_dir)
        return

    # 情况2：子文件夹模式
    subdirs = []
    if os.path.isdir(input_dir):
        for name in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, name)
            if os.path.isdir(path):
                subdirs.append(name)

    if not subdirs:
        print(f"错误: {input_dir}/ 下未找到PNG文件或子文件夹")
        return

    print(f"检测到 {len(subdirs)} 个子文件夹，将分别处理")
    print("=" * 60)

    for subdir in subdirs:
        sub_input = os.path.join(input_dir, subdir)
        sub_output = output_dir
        sub_files = get_png_files(sub_input)
        if not sub_files:
            print(f"\n跳过 {subdir}/ (无PNG文件)")
            continue
        print(f"\n>>> 处理子文件夹: {subdir}/ ({len(sub_files)} 张)")
        generate_spritesheet(sub_input, sub_output, name=subdir)

    print(f"\n全部完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
