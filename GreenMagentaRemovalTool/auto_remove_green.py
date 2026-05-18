#!/usr/bin/env python3
"""
绿幕去除工具 - 移除序列帧中的绿色像素（转为透明）。

检测算法（两阶段 + 距离变换约束）：
  阶段1 — 高置信度：greenness >= HIGH 直接移除
  阶段2 — 边缘溢出：从"深背景"（距角色>THRESHOLD px的透明区）队列泛洪。
           仅扩展 greenness >= LOW 的不透明像素。

greenness = G - max(R, B)，纯绿(0,255,0)→255，自然物体通常<20

用法：
  1. 把需要处理的PNG序列帧放入 input/ 文件夹
  2. 运行：python auto_remove_green.py
  3. 结果输出到 output/ 文件夹
"""

import os
from collections import deque
from PIL import Image


# ============ 配置 ============
INPUT_DIR = "input"
OUTPUT_DIR = "output"

HIGH = 40       # 阶段1：greenness >= 此值直接移除
LOW = 25        # 阶段2：greenness >= 此值且邻近深背景时移除（越高越保护角色内部）
DISTANCE_THRESHOLD = 15   # 距角色>此值的透明区为"深背景"
SPILL_ITERATIONS = 12     # 泛洪轮数（越大清理越深但可能误伤角色）


def get_png_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def greenness(r: int, g: int, b: int) -> int:
    return g - max(r, b)


def _compute_distance_transform(w: int, h: int, pixels) -> list[int]:
    """BFS：每个像素到最近不透明像素的切比雪夫距离。"""
    dist = [999999] * (w * h)
    queue = deque()

    for y in range(h):
        for x in range(w):
            if pixels[x, y][3] > 5:
                idx = y * w + x
                dist[idx] = 0
                queue.append(idx)

    if not queue:
        return [0] * (w * h)

    while queue:
        idx = queue.popleft()
        d = dist[idx]
        x, y = idx % w, idx // w
        nd = d + 1
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                nidx = ny * w + nx
                if dist[nidx] == 999999:
                    dist[nidx] = nd
                    queue.append(nidx)

    return dist


def build_mask(w: int, h: int, pixels, dist: list[int]) -> Image.Image:
    """两阶段蒙版构建，队列BFS泛洪。"""
    mask = [0] * (w * h)

    # 阶段1：高置信度
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if pixels[x, y][3] < 5:
                continue
            if greenness(pixels[x, y][0], pixels[x, y][1], pixels[x, y][2]) >= HIGH:
                mask[idx] = 1

    # 阶段2种子：深背景透明区 + 阶段1标记
    queue = deque()
    in_queue = [False] * (w * h)

    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if mask[idx] == 1:
                queue.append(idx)
                in_queue[idx] = True
            elif pixels[x, y][3] < 5 and dist[idx] > DISTANCE_THRESHOLD:
                queue.append(idx)
                in_queue[idx] = True

    # 层级BFS泛洪（仅通过不透明像素）
    for iteration in range(SPILL_ITERATIONS):
        if not queue:
            break
        for _ in range(len(queue)):
            idx = queue.popleft()
            x, y = idx % w, idx // w
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    nidx = ny * w + nx
                    if in_queue[nidx] or mask[nidx] == 1:
                        continue
                    if pixels[nx, ny][3] < 5:
                        continue

                    r, g, b = pixels[nx, ny][0], pixels[nx, ny][1], pixels[nx, ny][2]
                    if greenness(r, g, b) >= LOW:
                        mask[nidx] = 1
                        queue.append(nidx)
                        in_queue[nidx] = True

    mask_img = Image.new('L', (w, h))
    mask_img.putdata([255 if v == 1 else 0 for v in mask])
    return mask_img


def remove_pixels_from_frame(frame: Image.Image, mask: Image.Image) -> Image.Image:
    frame_rgba = frame.convert("RGBA")
    if mask.size != frame_rgba.size:
        mask = mask.resize(frame_rgba.size, Image.NEAREST)

    r, g, b, a = frame_rgba.split()
    w, h = frame_rgba.size
    a_data = list(a.get_flattened_data()) if hasattr(a, 'get_flattened_data') else list(a.getdata())
    mask_data = list(mask.get_flattened_data()) if hasattr(mask, 'get_flattened_data') else list(mask.getdata())

    new_a_data = [0 if mask_data[i] > 128 else a_data[i] for i in range(len(a_data))]

    new_a = Image.new('L', (w, h))
    new_a.putdata(new_a_data)
    return Image.merge('RGBA', (r, g, b, new_a))


def process_folder(input_dir: str, output_dir: str) -> None:
    files = get_png_files(input_dir)
    if not files:
        print(f"错误: {input_dir}/ 下未找到PNG文件")
        return

    print(f"检测到 {len(files)} 张PNG")
    print(f"greenness>={HIGH}(直删) >={LOW}(溢出)  深背景>{DISTANCE_THRESHOLD}px  泛洪{SPILL_ITERATIONS}轮")
    print("=" * 60)

    total_removed = 0
    os.makedirs(output_dir, exist_ok=True)

    for i, fname in enumerate(files):
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        img = Image.open(src_path).convert("RGBA")
        w, h = img.size
        pixels = img.load()

        dist = _compute_distance_transform(w, h, pixels)
        mask = build_mask(w, h, pixels, dist)
        mask_data = list(mask.get_flattened_data()) if hasattr(mask, 'get_flattened_data') else list(mask.getdata())
        removed = sum(1 for v in mask_data if v > 128)

        if removed == 0:
            img.save(dst_path)
        else:
            result = remove_pixels_from_frame(img, mask)
            result.save(dst_path)

        total_removed += removed

        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)} 完成 (累计移除: {total_removed})")

    print(f"\n完成！总计移除 {total_removed} 像素, 输出: {output_dir}")


def run_removal(input_dir: str, output_dir: str) -> None:
    files = get_png_files(input_dir)
    if files:
        process_folder(input_dir, output_dir)
        return

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
        sub_output = os.path.join(output_dir, subdir)
        sub_files = get_png_files(sub_input)
        if not sub_files:
            print(f"\n跳过 {subdir}/ (无PNG文件)")
            continue
        print(f"\n>>> 处理子文件夹: {subdir}/ ({len(sub_files)} 张)")
        process_folder(sub_input, sub_output)

    print(f"\n全部完成！输出目录: {output_dir}")


def main() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("【绿幕去除工具】")
    print()
    run_removal(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
