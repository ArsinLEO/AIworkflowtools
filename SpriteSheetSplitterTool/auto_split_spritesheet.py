#!/usr/bin/env python3
"""
精灵图集拆分工具 - 将 Sprite Sheet 按指定行列数拆回单个帧。

用法：
  1. 把 Sprite Sheet PNG 放入 input/ 文件夹
  2. 运行：python auto_split_spritesheet.py
  3. 输入列数和行数
  4. 结果输出到 output/ 文件夹
"""

import os
import re
from PIL import Image


INPUT_DIR = "input"
OUTPUT_DIR = "output"


def _natural_sort_key(s: str) -> list:
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", s)]


def get_png_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    return sorted(files, key=_natural_sort_key)


def split_spritesheet(filepath: str, output_dir: str,
                      cols: int, rows: int,
                      skip_empty: bool = True) -> int:
    """按 cols×rows 均匀网格拆分，所有输出帧大小一致。"""
    img = Image.open(filepath).convert("RGBA")
    w, h = img.size
    cell_w = w // cols
    cell_h = h // rows
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for ri in range(rows):
        for ci in range(cols):
            x0 = ci * cell_w
            y0 = ri * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            frame = img.crop((x0, y0, x1, y1))
            if skip_empty and frame.getbbox() is None:
                continue
            frame.save(os.path.join(output_dir, f"{count}.png"))
            count += 1

    return count


def main() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 收集所有待处理的文件：根级PNG → output/文件名/，子文件夹 → output/子文件夹/
    tasks: list[tuple[str, str]] = []  # (path, output_subdir)

    root_files = get_png_files(INPUT_DIR)
    for fname in root_files:
        sub_name = os.path.splitext(fname)[0]
        tasks.append((os.path.join(INPUT_DIR, fname), os.path.join(OUTPUT_DIR, sub_name)))

    if os.path.isdir(INPUT_DIR):
        subdirs = [d for d in os.listdir(INPUT_DIR)
                   if os.path.isdir(os.path.join(INPUT_DIR, d))
                   and d not in ("output", "__pycache__")]
        for sub in sorted(subdirs):
            sub_input = os.path.join(INPUT_DIR, sub)
            for fname in get_png_files(sub_input):
                tasks.append((os.path.join(sub_input, fname), os.path.join(OUTPUT_DIR, sub)))

    if not tasks:
        print(f"错误: {INPUT_DIR}/ 下未找到PNG文件")
        print("请将精灵图集PNG放入 input/ 文件夹后重试")
        input("\n按回车键关闭窗口...")
        return

    print("=" * 60)
    print("精灵图集拆分工具")
    print("=" * 60)
    print(f"\n找到 {len(tasks)} 张图集:")
    for filepath, _ in tasks:
        img = Image.open(filepath)
        print(f"  {os.path.basename(filepath)}  ({img.size[0]} x {img.size[1]})")
        img.close()

    # 询问行列数
    print()
    while True:
        try:
            cols = int(input("请输入列数 (cols): ").strip())
            if cols > 0:
                break
            print("列数必须 > 0")
        except ValueError:
            print("请输入整数")

    while True:
        try:
            rows = int(input("请输入行数 (rows): ").strip())
            if rows > 0:
                break
            print("行数必须 > 0")
        except ValueError:
            print("请输入整数")

    print(f"\n将以 {cols}列 × {rows}行 拆分...")
    print()

    total = 0
    for filepath, out_dir in tasks:
        fname = os.path.basename(filepath)
        img = Image.open(filepath)
        w, h = img.size
        cell_w = w // cols
        cell_h = h // rows
        img.close()

        print(f">>> {fname} ({w}x{h})")
        print(f"    单元格: {cell_w}x{cell_h}")

        count = split_spritesheet(filepath, out_dir, cols, rows)
        total += count
        print(f"    拆分: {count} 帧 → {out_dir}/")

    print(f"\n{'=' * 60}")
    print(f"全部完成！共拆分 {total} 帧，帧尺寸统一为 {cell_w}x{cell_h}")
    print(f"结果在 output/ 文件夹中")
    print("=" * 60)
    input("\n按回车键关闭窗口...")


if __name__ == "__main__":
    main()
