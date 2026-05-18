#!/usr/bin/env python3
"""
交互式去水印工具 - 先尝试自动检测，失败后让用户手动指定水印区域。

特性：
  - 复用 auto_remove_watermark 的检测逻辑（不动原代码）
  - 自动检测失败后进入交互模式，询问水印位置和大小
  - 支持角落模式（快速指定）和自定义矩形模式（精确指定）
  - 与 SpriteSheetSplitterTool 交互风格一致

用法：
  1. 把需要去水印的PNG放入 input/ 文件夹
  2. 运行：python auto_remove_watermark_interactive.py
  3. 如果自动检测不到水印，按提示手动指定区域
  4. 结果输出到 output/ 文件夹
"""

import os
import sys
from pathlib import Path
from PIL import Image

# 复用原脚本的工具函数和蒙版处理管线
from auto_remove_watermark import (
    get_png_files,
    load_frames,
    build_watermark_mask,
    refine_mask,
    remove_watermark_from_frame,
    INPUT_DIR,
    OUTPUT_DIR,
    PROCESS_SCALE,
)

# ============================================================
# 交互模式配置
# ============================================================

# 角落默认尺寸（相对画面百分比）
DEFAULT_CORNER_W_PCT = 20
DEFAULT_CORNER_H_PCT = 15


def _ask_corner_mode(img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """角落模式：选择角落 + 百分比，返回 (min_x, min_y, max_x, max_y) 或 None。"""
    print()
    print("  Select corner:")
    print("    1. Top-Left")
    print("    2. Top-Right")
    print("    3. Bottom-Left")
    print("    4. Bottom-Right")
    print("    0. Cancel")

    while True:
        try:
            choice = input("  > ").strip()
            if choice == "0":
                return None
            corner = int(choice)
            if 1 <= corner <= 4:
                break
            print("  Please enter 1-4 or 0 to cancel")
        except ValueError:
            print("  Please enter a number")

    print()
    w_pct_str = input(f"  Watermark width (% of image, default {DEFAULT_CORNER_W_PCT}): ").strip()
    h_pct_str = input(f"  Watermark height (% of image, default {DEFAULT_CORNER_H_PCT}): ").strip()

    try:
        w_pct = float(w_pct_str) if w_pct_str else DEFAULT_CORNER_W_PCT
        h_pct = float(h_pct_str) if h_pct_str else DEFAULT_CORNER_H_PCT
    except ValueError:
        print("  Invalid percentage, using defaults")
        w_pct = DEFAULT_CORNER_W_PCT
        h_pct = DEFAULT_CORNER_H_PCT

    w_px = max(1, int(img_w * w_pct / 100))
    h_px = max(1, int(img_h * h_pct / 100))

    if corner == 1:  # top-left
        x0, y0 = 0, 0
    elif corner == 2:  # top-right
        x0, y0 = img_w - w_px, 0
    elif corner == 3:  # bottom-left
        x0, y0 = 0, img_h - h_px
    else:  # bottom-right
        x0, y0 = img_w - w_px, img_h - h_px

    x1, y1 = x0 + w_px - 1, y0 + h_px - 1
    return x0, y0, x1, y1


def _ask_custom_mode(img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """自定义矩形模式：直接输入像素坐标，返回 (min_x, min_y, max_x, max_y) 或 None。"""
    print()
    print(f"  Image size: {img_w} x {img_h}")
    print("  Enter pixel coordinates (0 to cancel):")

    try:
        x_str = input(f"  X (0-{img_w - 1}): ").strip()
        if x_str == "0" or x_str == "":
            # check if really cancelling
            pass
        x = int(x_str)
        if x < 0 or x >= img_w:
            print("  X out of range, cancelling")
            return None

        y_str = input(f"  Y (0-{img_h - 1}): ").strip()
        y = int(y_str)
        if y < 0 or y >= img_h:
            print("  Y out of range, cancelling")
            return None

        w_str = input(f"  Width (1-{img_w - x}): ").strip()
        w = int(w_str)
        if w <= 0 or x + w > img_w:
            print("  Width out of range, cancelling")
            return None

        h_str = input(f"  Height (1-{img_h - y}): ").strip()
        h = int(h_str)
        if h <= 0 or y + h > img_h:
            print("  Height out of range, cancelling")
            return None

        return x, y, x + w - 1, y + h - 1

    except ValueError:
        print("  Invalid input, cancelling")
        return None


def ask_watermark_region(img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """交互式询问水印区域。返回 (min_x, min_y, max_x, max_y) 或 None 表示跳过。"""
    print()
    print("  Specify watermark region method:")
    print("    1. Corner mode (top-left, top-right, etc.)")
    print("    2. Custom rectangle (exact pixel coords)")
    print("    0. Skip (copy originals to output)")

    while True:
        method = input("  > ").strip()
        if method == "0":
            return None
        if method == "1":
            return _ask_corner_mode(img_w, img_h)
        if method == "2":
            return _ask_custom_mode(img_w, img_h)
        print("  Please enter 0, 1, or 2")


def build_manual_mask(w: int, h: int,
                      region: tuple[int, int, int, int]) -> Image.Image:
    """根据用户指定的区域创建蒙版（白=水印，黑=保留）。"""
    min_x, min_y, max_x, max_y = region
    mask_data = [0] * (w * h)
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            mask_data[y * w + x] = 255
    mask = Image.new('L', (w, h))
    mask.putdata(mask_data)
    return mask


def process_with_fallback(input_dir: str, output_dir: str) -> None:
    """先尝试自动检测，失败后走交互式手动指定。"""
    files = get_png_files(input_dir)
    if not files:
        print(f"Error: No PNG files found in {input_dir}/")
        return

    print(f"Detected {len(files)} PNG files")
    print("=" * 60)

    # 加载第一帧获取原始尺寸
    first_img = Image.open(os.path.join(input_dir, files[0])).convert("RGBA")
    orig_w, orig_h = first_img.size
    first_img.close()

    mask = None

    # ========== Step 1: 尝试自动检测 ==========
    print("\n[Auto-detect] Searching for watermark...")

    small_frames = load_frames(input_dir, scale=PROCESS_SCALE)
    small_w, small_h = small_frames[0].size

    if len(small_frames) >= 2:
        mask, detected = build_watermark_mask(small_frames)
        if mask is not None:
            wm_pixels = sum(1 for p in mask.getdata() if p > 128)
            print(f"[Auto-detect] SUCCESS: detected at {detected}, {wm_pixels} pixels")
        else:
            print(f"[Auto-detect] FAILED: no watermark found or exceeds safety threshold")
            if detected:
                print(f"[Auto-detect] Candidates rejected: {detected}")

    # ========== Step 2: 交互式回退 ==========
    if mask is None:
        print()
        print("=" * 60)
        print("Interactive Watermark Specification")
        print(f"Image: {orig_w} x {orig_h}, {len(files)} frames")
        print("=" * 60)

        # 在缩略图上询问，然后缩放回原尺寸
        region_small = ask_watermark_region(small_w, small_h)
        if region_small is None:
            print("\nSkipping watermark removal, copying originals...")
            os.makedirs(output_dir, exist_ok=True)
            for fname in files:
                src = Image.open(os.path.join(input_dir, fname)).convert("RGBA")
                src.save(os.path.join(output_dir, fname))
            print(f"Done! Originals copied to: {output_dir}")
            return

        # 构建蒙版（在缩略图尺寸上）
        mask = build_manual_mask(small_w, small_h, region_small)
        mask = refine_mask(mask)

        px_count = sum(1 for p in mask.getdata() if p > 128)
        print(f"\n[Manual] Region mask created: {px_count} pixels to remove")

    else:
        # 自动检测成功，也做 refine
        mask = refine_mask(mask)

    # 将蒙版从缩略图尺寸缩放回原尺寸
    orig_size = (orig_w, orig_h)
    if mask.size != orig_size:
        mask = mask.resize(orig_size, Image.NEAREST)

    # ========== Step 3: 批量应用蒙版 ==========
    print(f"\nApplying watermark removal to {len(files)} frames...")
    os.makedirs(output_dir, exist_ok=True)

    for i, fname in enumerate(files):
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        frame = Image.open(src_path).convert("RGBA")
        result = remove_watermark_from_frame(frame, mask)
        result.save(dst_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)} done")

    print(f"\nDone! Output: {output_dir}")


def run_interactive(input_dir: str, output_dir: str) -> None:
    """主流程：支持直接放PNG或子文件夹模式。"""
    # 情况1：根目录直接有PNG文件
    files = get_png_files(input_dir)
    if files:
        process_with_fallback(input_dir, output_dir)
        return

    # 情况2：子文件夹模式
    subdirs = []
    if os.path.isdir(input_dir):
        for name in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, name)
            if os.path.isdir(path):
                subdirs.append(name)

    if not subdirs:
        print(f"Error: No PNG files or subdirectories found in {input_dir}/")
        return

    print(f"Detected {len(subdirs)} subdirectories")
    print("=" * 60)

    for subdir in subdirs:
        sub_input = os.path.join(input_dir, subdir)
        sub_output = os.path.join(output_dir, subdir)

        sub_files = get_png_files(sub_input)
        if not sub_files:
            print(f"\nSkipping {subdir}/ (no PNG files)")
            continue

        print(f"\n>>> Processing: {subdir}/ ({len(sub_files)} frames)")
        process_with_fallback(sub_input, sub_output)

    print(f"\nAll done! Output: {output_dir}")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Interactive Watermark Removal Tool")
    print("Auto-detect first, manual fallback if needed")
    print("=" * 60)

    run_interactive(input_dir, output_dir)


if __name__ == "__main__":
    main()
