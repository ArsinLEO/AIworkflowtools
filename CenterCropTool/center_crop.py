"""Center-crop images — batch crop all PNGs in input/ (including subdirectories).

Reads from input/ recursively, writes to output/ preserving folder structure.
"""

import os
import sys
from pathlib import Path
from PIL import Image


def center_crop(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    w, h = img.size
    left = (w - target_width) // 2
    top = (h - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    return img.crop((left, top, right, bottom))


def main():
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"

    if not input_dir.is_dir():
        print(f"错误: 找不到 input/ 文件夹")
        return

    # Find images recursively
    files = sorted(input_dir.rglob("*.png"))
    if not files:
        print("input/ 及其子文件夹中没有找到 .png 文件。")
        return

    # Show summary
    print("=" * 50)
    print(f"  找到 {len(files)} 个 PNG 文件")
    # Show folder structure
    folders = set(f.relative_to(input_dir).parent for f in files)
    for folder in sorted(folders):
        folder_name = str(folder) if str(folder) != "." else "(根目录)"
        count = sum(1 for f in files if f.relative_to(input_dir).parent == folder)
        print(f"    {folder_name}/  ({count} 张)")
    # Show first file size as reference
    first = Image.open(files[0])
    print(f"  参考尺寸: {files[0].relative_to(input_dir)}  →  {first.size[0]}x{first.size[1]}")
    print("=" * 50)

    # Get target size
    try:
        size_str = input("输入目标尺寸（正方形只需一个数，如 640）: ").strip()
        if "x" in size_str.lower():
            parts = size_str.lower().split("x")
            target_w = int(parts[0].strip())
            target_h = int(parts[1].strip())
        else:
            target_w = target_h = int(size_str)
    except ValueError:
        print("错误: 请输入有效数字，如 640 或 640x480")
        return

    print(f"\n目标尺寸: {target_w}x{target_h}")
    confirm = input("确认开始裁切？(y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("已取消。")
        return

    # Process
    done = 0
    skipped = 0

    for f in files:
        img = Image.open(f)
        orig_w, orig_h = img.size

        if orig_w < target_w or orig_h < target_h:
            rel = f.relative_to(input_dir)
            print(f"  跳过 {rel}: {orig_w}x{orig_h} < {target_w}x{target_h}")
            skipped += 1
            continue

        cropped = center_crop(img, target_w, target_h)
        out_path = output_dir / f.relative_to(input_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out_path)
        crop_px = (orig_w - target_w) // 2
        rel = f.relative_to(input_dir)
        print(f"  完成 {rel}: {orig_w}x{orig_h} → {target_w}x{target_h} (四边各裁 {crop_px}px)")
        done += 1

    print(f"\n处理完毕: {done} 张完成, {skipped} 张跳过")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
