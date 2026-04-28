#!/usr/bin/env python3
"""
工具2：序列帧缩放合并工具 - 将所有PNG缩放到宽度1280（高度等比），
按每行2图的形式纵向拼接为一张大图。

用法：
  1. 把PNG序列帧放入 input/ 文件夹（或子文件夹）
  2. 运行：python resize_to_1920_column.py
  3. 结果输出到 output/ 文件夹
"""

import math
import os
import re
from PIL import Image


INPUT_DIR = "input"
OUTPUT_DIR = "output"
TARGET_WIDTH = 1280
MAX_TEXTURE_SIZE = 8192  # Godot 纹理上限


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


def resize_and_merge(input_dir: str, output_dir: str, name: str = "merged_column") -> None:
    """将单个文件夹内的PNG缩放到目标宽度后拼接为图集。"""
    files = get_png_files(input_dir)
    if not files:
        print(f"错误: {input_dir}/ 下未找到PNG文件")
        return

    total = len(files)
    print(f"检测到 {total} 张PNG，正在缩放至宽度 {TARGET_WIDTH}...")

    # 加载并缩放所有帧
    frames = []
    for fname in files:
        img = Image.open(os.path.join(input_dir, fname)).convert("RGBA")
        if img.width != TARGET_WIDTH:
            ratio = TARGET_WIDTH / img.width
            new_h = round(img.height * ratio)
            img = img.resize((TARGET_WIDTH, new_h), Image.LANCZOS)
        frames.append(img)
        print(f"  {fname}: {img.width}x{img.height}")

    # 取最大帧高作为行高
    max_h = max(img.height for img in frames)

    # 自动计算列数，确保宽度和高度都在纹理上限内
    cols = 1
    while cols <= total:
        rows = math.ceil(total / cols)
        sheet_w = TARGET_WIDTH * cols
        sheet_h = rows * max_h
        if sheet_w <= MAX_TEXTURE_SIZE and sheet_h <= MAX_TEXTURE_SIZE:
            break
        cols += 1
    else:
        print(f"错误: 无法在 {MAX_TEXTURE_SIZE} 纹理限制内容纳 {total} 帧")
        return

    if cols == 1:
        print(f"  自动列数: 1（适配纹理限制 {MAX_TEXTURE_SIZE}）")
    else:
        print(f"  自动列数: {cols}（适配纹理限制 {MAX_TEXTURE_SIZE}）")

    # 计算每行实际高度（居中放置）
    row_heights = []
    for r in range(rows):
        row_max_h = 0
        for c in range(cols):
            idx = r * cols + c
            if idx < total:
                row_max_h = max(row_max_h, frames[idx].height)
        row_heights.append(row_max_h)

    sheet_w = TARGET_WIDTH * cols
    sheet_h = sum(row_heights)
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

    # 逐帧拼接
    y_offset = 0
    for r in range(rows):
        row_h = row_heights[r]
        for c in range(cols):
            idx = r * cols + c
            if idx >= total:
                break
            img = frames[idx]
            x = c * TARGET_WIDTH
            y = y_offset + (row_h - img.height) // 2  # 居中放置
            sheet.paste(img, (x, y), img)
        y_offset += row_h

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.png")
    sheet.save(output_path)
    print(f"合并完成: {output_path} ({sheet_w}x{sheet_h})")
    print(f"  总帧数: {total}, 列数: {cols}, 行数: {rows}")


def main() -> None:
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 情况1：根目录直接有PNG文件
    files = get_png_files(input_dir)
    if files:
        resize_and_merge(input_dir, output_dir)
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
        sub_files = get_png_files(sub_input)
        if not sub_files:
            print(f"\n跳过 {subdir}/ (无PNG文件)")
            continue
        print(f"\n>>> 处理子文件夹: {subdir}/ ({len(sub_files)} 张)")
        resize_and_merge(sub_input, output_dir, name=subdir)

    print(f"\n全部完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
