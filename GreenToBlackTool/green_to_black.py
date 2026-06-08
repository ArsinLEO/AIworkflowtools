#!/usr/bin/env python3
"""
绿色像素替换工具 — 将 PNG 图片中的绿色像素替换为纯黑色。

绿色判定（默认）:
  - 绿色通道值 > 红色通道值 + 阈值
  - 绿色通道值 > 蓝色通道值 + 阈值
  - 默认阈值: 30（绿色通道比红/蓝多 30 以上即视为绿色像素）

用法:
  python green_to_black.py              → 交互模式，可调整阈值
  python green_to_black.py --auto       → 全自动，使用默认阈值
"""

import os
import re
import sys
from pathlib import Path
from PIL import Image


INPUT_DIR = "input"
OUTPUT_DIR = "output"
DEFAULT_THRESHOLD = 30  # 绿色通道需比红/蓝多出多少


def find_all_png_files(folder: str) -> list[Path]:
    files = list(Path(folder).rglob("*.png"))
    files.sort(key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", p.name)])
    return files


def is_green_pixel(r: int, g: int, b: int, threshold: int) -> bool:
    """判定一个像素是否为绿色。"""
    return g > r + threshold and g > b + threshold


def replace_green_with_black(img: Image.Image, threshold: int) -> tuple:
    """
    将图片中的绿色像素替换为纯黑色 (0, 0, 0, alpha 不变)。
    返回 (处理后的图片, 替换像素数, 总像素数)。
    """
    img = img.convert("RGBA")
    w, h = img.size
    pixels = img.load()
    replaced = 0
    total = w * h

    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if a > 0 and is_green_pixel(r, g, b, threshold):
                pixels[x, y] = (0, 0, 0, a)
                replaced += 1

    return img, replaced, total


def main():
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    input_dir = script_dir / INPUT_DIR
    output_dir = script_dir / OUTPUT_DIR

    if not input_dir.is_dir():
        print(f"错误: 找不到 {INPUT_DIR}/ 文件夹")
        return

    files = find_all_png_files(str(input_dir))
    if not files:
        print(f"错误: {INPUT_DIR}/ 中没有 .png 文件")
        return

    ref = Image.open(files[0])
    img_w, img_h = ref.size

    print("=" * 60)
    print("  绿色像素 → 纯黑色 替换工具")
    print("=" * 60)
    print(f"\n找到 {len(files)} 个 PNG 文件")
    print(f"参考尺寸: {img_w} x {img_h}\n")

    auto_mode = "--auto" in sys.argv

    if auto_mode:
        threshold = DEFAULT_THRESHOLD
        print(f"自动模式，绿色判定阈值: {threshold}")
        print("（绿色通道比红/蓝多 {} 以上视为绿色像素）\n".format(threshold))
    else:
        print("绿色判定: G > R + 阈值 且 G > B + 阈值")
        print(f"默认阈值: {DEFAULT_THRESHOLD}")
        print("  → 阈值越小越敏感（更多像素被识别为绿色）")
        print("  → 阈值越大越严格（只替换明显的绿色像素）\n")

        threshold_str = input(f"阈值 (默认 {DEFAULT_THRESHOLD}): ").strip()
        try:
            threshold = int(threshold_str) if threshold_str else DEFAULT_THRESHOLD
        except ValueError:
            print(f"输入无效，使用默认值: {DEFAULT_THRESHOLD}")
            threshold = DEFAULT_THRESHOLD

    # 先用第一帧测试
    test_img, test_replaced, test_total = replace_green_with_black(ref.copy(), threshold)
    test_pct = test_replaced / test_total * 100 if test_total > 0 else 0
    print(f"\n测试第一帧: 替换了 {test_replaced} / {test_total} 像素 ({test_pct:.1f}%)")

    if test_replaced == 0:
        print("警告: 第一帧未检测到绿色像素，请降低阈值后重试。")
        return

    if not auto_mode:
        print()
        confirm = input("确认开始处理全部帧？(y/n): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("已取消。")
            return

    # 处理全部帧
    print(f"\n正在处理 {len(files)} 帧，阈值={threshold}...")
    done = 0
    total_replaced = 0

    for f in files:
        img = Image.open(f)
        result, replaced, pixel_count = replace_green_with_black(img, threshold)
        total_replaced += replaced

        out_path = output_dir / f.relative_to(input_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)

        done += 1
        if done % 10 == 0 or done == len(files):
            print(f"  处理 [{done}/{len(files)}]")

    avg_replaced = total_replaced / done if done > 0 else 0
    avg_pct = avg_replaced / (img_w * img_h) * 100
    print(f"\n全部完成！共处理 {done} 帧")
    print(f"平均每帧替换 {avg_replaced:.0f} 个绿色像素 ({avg_pct:.1f}%)")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
