#!/usr/bin/env python3
"""
序列帧对齐工具 - 一键对齐多个动画序列到参考图。

用法:
  1. 在 reference/ 文件夹放入一张标准参考图（仅1张PNG）
  2. 在 input/ 文件夹放入多个子文件夹，每个子文件夹是一组序列帧
  3. 运行: python auto_align_frames.py
  4. 结果输出到 output/ 文件夹

输出说明:
  - 所有序列帧画布与参考图一致
  - 实体A（最大连通组件）对齐到参考图的位置和大小
  - 画布扩展为正方形（居中，不拉伸）
  - 最终缩放到 640x640
  - 文件名重命名为 0.png, 1.png, 2.png ...
"""

import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from PIL import Image


# ============ 配置 ============
REF_DIR = "reference"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
PARAMS_FILE = "align_params.json"

# ============ 核心函数 ============

def find_single_reference_image(folder: str) -> str:
    """在文件夹中找到唯一的PNG文件作为参考图。"""
    pngs = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not pngs:
        raise RuntimeError(f"{folder}/ 下未找到PNG文件，请放入一张标准参考图")
    if len(pngs) > 1:
        print(f"警告: {folder}/ 下有多张PNG，使用第一张: {pngs[0]}")
    return os.path.join(folder, pngs[0])


def find_all_png_files(folder: str) -> list[str]:
    """返回文件夹中所有PNG文件（按自然排序：1 < 2 < 10 < 11）。"""
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)])
    return files


def detect_main_component_bounds(img_path: str, alpha_threshold: int = 10) -> tuple | None:
    """
    返回最大连通非透明区域的边界框 (min_x, min_y, max_x, max_y)。
    使用最大连通组件排除分离的光晕/噪点。
    """
    img = Image.open(img_path).convert("RGBA")
    w, h = img.size
    pixels = img.load()

    mask = [[pixels[x, y][3] > alpha_threshold for y in range(h)] for x in range(w)]
    visited = [[False] * h for _ in range(w)]
    largest = None

    for sx in range(w):
        for sy in range(h):
            if not mask[sx][sy] or visited[sx][sy]:
                continue
            queue = deque([(sx, sy)])
            visited[sx][sy] = True
            comp = [(sx, sy)]
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[nx][ny] and not visited[nx][ny]:
                        visited[nx][ny] = True
                        comp.append((nx, ny))
                        queue.append((nx, ny))
            if largest is None or len(comp) > len(largest):
                largest = comp

    if not largest:
        return None

    xs = [p[0] for p in largest]
    ys = [p[1] for p in largest]
    return (min(xs), min(ys), max(xs), max(ys))


def compute_alignment(ref_bbox: tuple, tgt_bbox: tuple) -> dict:
    """计算缩放比例和平移量，使目标实体A对齐到参考实体A。"""
    r_x1, r_y1, r_x2, r_y2 = ref_bbox
    t_x1, t_y1, t_x2, t_y2 = tgt_bbox

    ref_w = r_x2 - r_x1
    ref_h = r_y2 - r_y1
    tgt_w = t_x2 - t_x1
    tgt_h = t_y2 - t_y1

    scale = ref_h / tgt_h

    offset_x = r_x1 - t_x1 * scale
    offset_y = r_y1 - t_y1 * scale

    return {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "ref_bbox": ref_bbox,
        "tgt_bbox": tgt_bbox,
        "ref_size": (ref_w, ref_h),
        "tgt_size": (tgt_w, tgt_h),
    }


def align_frame(src_path: str, dst_path: str, canvas_size: tuple, scale: float, offset_x: float, offset_y: float) -> None:
    """对单张图进行缩放+平移，输出到目标尺寸画布。"""
    img = Image.open(src_path).convert("RGBA")
    canvas_w, canvas_h = canvas_size

    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    scaled = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    paste_x = int(round(offset_x))
    paste_y = int(round(offset_y))
    canvas.paste(scaled, (paste_x, paste_y), scaled)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    canvas.save(dst_path)


def run_phase1() -> dict:
    """第一阶段：检测参考图，计算每个输入文件夹的对齐参数。"""
    # 1. 读取参考图
    ref_path = find_single_reference_image(REF_DIR)
    ref_img = Image.open(ref_path)
    canvas_w, canvas_h = ref_img.size

    ref_bbox = detect_main_component_bounds(ref_path)
    if not ref_bbox:
        raise RuntimeError("参考图未检测到实体A！")

    print(f"参考图: {os.path.basename(ref_path)}")
    print(f"画布尺寸: {canvas_w}x{canvas_h}")
    print(f"参考实体A: {ref_bbox}  尺寸: ({ref_bbox[2]-ref_bbox[0]}x{ref_bbox[3]-ref_bbox[1]})")
    print("=" * 60)

    # 2. 遍历输入文件夹
    params = {
        "ref_image": os.path.basename(ref_path),
        "canvas_size": [canvas_w, canvas_h],
        "ref_bbox": ref_bbox,
        "folders": {},
    }

    input_subfolders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]
    if not input_subfolders:
        raise RuntimeError(f"{INPUT_DIR}/ 下未找到子文件夹")

    for sub_name in sorted(input_subfolders):
        sub_path = os.path.join(INPUT_DIR, sub_name)
        files = find_all_png_files(sub_path)
        if not files:
            print(f"跳过空文件夹: {sub_name}")
            continue

        first_frame = os.path.join(sub_path, files[0])
        tgt_bbox = detect_main_component_bounds(first_frame)
        if not tgt_bbox:
            print(f"警告: {sub_name} 第一帧未检测到实体A，跳过")
            continue

        align = compute_alignment(ref_bbox, tgt_bbox)
        params["folders"][sub_name] = {
            "first_frame": files[0],
            "scale": align["scale"],
            "offset_x": align["offset_x"],
            "offset_y": align["offset_y"],
            "tgt_bbox": tgt_bbox,
        }

        # 生成预览
        preview_path = os.path.join(OUTPUT_DIR, "_preview", sub_name, files[0])
        align_frame(first_frame, preview_path, (canvas_w, canvas_h), align["scale"], align["offset_x"], align["offset_y"])

        print(f"\n[{sub_name}] ({len(files)} 帧)")
        print(f"  第一帧: {files[0]}")
        print(f"  实体A: {tgt_bbox}  尺寸: ({tgt_bbox[2]-tgt_bbox[0]}x{tgt_bbox[3]-tgt_bbox[1]})")
        print(f"  scale : {align['scale']:.4f}")
        print(f"  offset_x: {align['offset_x']:.1f}")
        print(f"  offset_y: {align['offset_y']:.1f}")
        print(f"  预览: {preview_path}")

    # 保存参数
    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"参数已保存到: {PARAMS_FILE}")
    print(f"预览图已保存到: {OUTPUT_DIR}/_preview/")
    print("\n请检查预览图。确认无误后运行: python auto_align_frames.py --apply")

    return params


def run_phase2() -> None:
    """第二阶段：应用对齐参数到所有帧，重命名为 0.png, 1.png, ..."""
    if not os.path.exists(PARAMS_FILE):
        print(f"错误: 参数文件不存在: {PARAMS_FILE}")
        print("请先运行第一阶段（不带 --apply 参数）")
        sys.exit(1)

    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        params = json.load(f)

    canvas_w, canvas_h = params["canvas_size"]

    for sub_name, cfg in params["folders"].items():
        sub_path = os.path.join(INPUT_DIR, sub_name)
        if not os.path.isdir(sub_path):
            print(f"跳过: {sub_name}")
            continue

        files = find_all_png_files(sub_path)
        scale = cfg["scale"]
        offset_x = cfg["offset_x"]
        offset_y = cfg["offset_y"]

        final_dir = os.path.join(OUTPUT_DIR, sub_name)
        os.makedirs(final_dir, exist_ok=True)

        print(f"\n处理 [{sub_name}] ({len(files)} 帧)...")

        for i, fname in enumerate(files):
            src = os.path.join(sub_path, fname)
            dst = os.path.join(final_dir, f"{i}.png")
            align_frame(src, dst, (canvas_w, canvas_h), scale, offset_x, offset_y)
            if (i + 1) % 20 == 0 or i + 1 == len(files):
                print(f"  处理 {i + 1}/{len(files)}")

    print(f"\n全部完成！输出画布: {canvas_w}x{canvas_h}")
    print(f"最终输出目录: {OUTPUT_DIR}/")


def main() -> None:
    # 确保必要目录存在
    os.makedirs(REF_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "--apply":
        run_phase2()
    else:
        run_phase1()


if __name__ == "__main__":
    main()
