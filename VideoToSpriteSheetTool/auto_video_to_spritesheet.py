#!/usr/bin/env python3
"""
视频转精灵图集工具 - 将 MP4 视频按指定帧率提取帧，组合成精灵图集。

用法：
  1. 把 MP4 视频放入 input/ 文件夹
  2. 运行：python auto_video_to_spritesheet.py
  3. 精灵图集输出到 output/ 文件夹，原始帧保留在 frames/ 文件夹

特性：
  - 按目标帧率（默认12FPS）均匀提取帧
  - 保持原始帧尺寸，不做任何缩放或压缩
  - 自动计算网格布局，确保不超出 Godot 纹理上限（8192px）
  - 帧数超出一张图集时自动分割为多张图集
  - 支持子文件夹批量处理
"""

import os
import re
import math
from pathlib import Path
import cv2
from PIL import Image


# ============================================================
# 配置参数
# ============================================================

INPUT_DIR = "input"
OUTPUT_DIR = "output"
FRAMES_DIR = "frames"           # 提取的原始帧（按视频名存放子文件夹）
COLS = 4                        # 每行帧数偏好（脚本会自动调整以适配纹理限制）
MAX_TEXTURE_SIZE = 8192         # Godot 纹理尺寸上限（PC 通常 8192，高端卡 16384）

# ============================================================


def _natural_sort_key(s: str) -> list:
    """自然排序键：数字按数值排序，文本按字典序。"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", s)]


def get_video_files(folder: str) -> list[str]:
    """返回文件夹中所有视频文件（按自然排序）。"""
    if not os.path.isdir(folder):
        return []
    exts = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    files = [f for f in os.listdir(folder)
             if Path(f).suffix.lower() in exts]
    return sorted(files, key=_natural_sort_key)


def extract_frames(video_path: str, output_dir: str, target_fps: int):
    """从视频中按目标帧率提取帧，保存为PNG序列。

    使用最近邻帧采样确保精确的目标帧率，避免 banker's rounding 带来的偏差。
    返回: (提取帧数, 源视频帧率, 视频时长秒, 实际帧率)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    source_fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration: float = total_frames / source_fps if source_fps > 0 else 0.0

    # 预计算目标源帧索引（最近邻采样，避免浮点累积漂移）
    expected: int = max(1, int(duration * target_fps))
    target_indices: set[int] = set()
    for i in range(expected):
        src_idx: int = min(
            int(i * source_fps / target_fps + 0.5),  # nearest neighbour
            total_frames - 1
        )
        target_indices.add(src_idx)

    os.makedirs(output_dir, exist_ok=True)

    # 清空旧帧，避免与上次残留混合
    for old_file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, old_file))

    frame_idx: int = 0
    saved: int = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(frame_rgba)
            img.save(os.path.join(output_dir, f"{saved:06d}.png"))
            saved += 1
        frame_idx += 1

    cap.release()
    actual_fps: float = saved / duration if duration > 0 else 0.0
    return saved, source_fps, duration, actual_fps


def compose_spritesheet(frames_dir: str, output_dir: str, name: str,
                        max_texture_size: int, preferred_cols: int) -> dict:
    """将帧文件夹中的PNG序列组合成精灵图集。

    若总帧数超出单张纹理限制，自动分割为多张图集。
    返回: {"sheets", "total_frames", "cell_w", "cell_h", "max_sheet_w", "max_sheet_h"}
    """
    files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".png")],
        key=_natural_sort_key
    )

    if not files:
        return {"sheets": 0, "total_frames": 0, "cell_w": 0, "cell_h": 0,
                "max_sheet_w": 0, "max_sheet_h": 0}

    # 获取帧尺寸（视频提取的所有帧尺寸一致）
    first_img = Image.open(os.path.join(frames_dir, files[0]))
    cell_w: int = first_img.width
    cell_h: int = first_img.height
    first_img.close()

    total: int = len(files)

    # 检查单帧是否超过纹理限制
    if cell_w > max_texture_size or cell_h > max_texture_size:
        raise RuntimeError(
            f"帧尺寸 {cell_w}x{cell_h} 超过纹理上限 {max_texture_size}，"
            f"无法生成图集。请先缩小视频分辨率。"
        )

    # 计算每张图集最多容纳的帧数
    max_cols: int = max(1, max_texture_size // cell_w)
    max_rows: int = max(1, max_texture_size // cell_h)
    max_per_sheet: int = max_cols * max_rows

    # 需要几张图集
    num_sheets: int = math.ceil(total / max_per_sheet)

    os.makedirs(output_dir, exist_ok=True)

    total_saved: int = 0
    max_sheet_w: int = 0
    max_sheet_h: int = 0
    for sheet_idx in range(num_sheets):
        start: int = sheet_idx * max_per_sheet
        end: int = min(start + max_per_sheet, total)
        sheet_files = files[start:end]
        sheet_count: int = len(sheet_files)

        # 为当前分片计算最优网格
        cols: int = min(preferred_cols, max_cols)
        while cols <= max_cols:
            rows: int = math.ceil(sheet_count / cols)
            if rows <= max_rows:
                break
            cols += 1

        sheet_w: int = cols * cell_w
        sheet_h: int = rows * cell_h
        max_sheet_w = max(max_sheet_w, sheet_w)
        max_sheet_h = max(max_sheet_h, sheet_h)

        sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

        for i, fname in enumerate(sheet_files):
            row: int = i // cols
            col: int = i % cols
            x: int = col * cell_w
            y: int = row * cell_h
            img = Image.open(os.path.join(frames_dir, fname))
            sheet.paste(img, (x, y))
            img.close()

        # 命名：多张时加 _part1, _part2 后缀
        out_name: str = f"{name}_part{sheet_idx + 1}" if num_sheets > 1 else name
        out_path: str = os.path.join(output_dir, f"{out_name}.png")
        sheet.save(out_path)

        print(f"  [{sheet_idx + 1}/{num_sheets}] {out_name}.png "
              f"({sheet_w}x{sheet_h}), {sheet_count} frames, grid {cols}x{rows}")

        total_saved += sheet_count

    return {
        "sheets": num_sheets,
        "total_frames": total_saved,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "max_sheet_w": max_sheet_w,
        "max_sheet_h": max_sheet_h,
    }


def process_video(video_path: str, input_dir: str, output_dir: str,
                  target_fps: int) -> None:
    """处理单个视频：提取帧 → 合成图集。"""
    video_name: str = Path(video_path).stem
    video_file: str = os.path.basename(video_path)

    # 帧输出到 frames/<视频名>/
    frames_subdir: str = os.path.join(FRAMES_DIR, video_name)

    print(f"\n{'=' * 60}")
    print(f"处理: {video_file}")
    print(f"{'=' * 60}")

    # Step 1: 提取帧
    print(f"  提取帧 (目标 {target_fps} FPS)...")
    saved, source_fps, duration, actual_fps = extract_frames(
        video_path, frames_subdir, target_fps
    )
    print(f"    源视频: {source_fps:.1f} FPS, {duration:.1f}秒")
    print(f"    提取: {saved} 帧 (实际 {actual_fps:.1f} FPS)")

    # Step 2: 合成图集
    print(f"  合成精灵图集...")
    result = compose_spritesheet(
        frames_subdir, output_dir, video_name,
        MAX_TEXTURE_SIZE, COLS
    )

    if result["sheets"] == 0:
        print(f"  错误: 未生成图集")
        return

    print(f"  总帧数: {result['total_frames']}")
    print(f"  单元格: {result['cell_w']}x{result['cell_h']}")
    print(f"  图集数: {result['sheets']}")

    # 检查是否接近纹理限制
    max_w: int = result["max_sheet_w"]
    max_h: int = result["max_sheet_h"]
    if max_w > MAX_TEXTURE_SIZE * 0.85 or max_h > MAX_TEXTURE_SIZE * 0.85:
        pct_w: float = max_w / MAX_TEXTURE_SIZE * 100
        pct_h: float = max_h / MAX_TEXTURE_SIZE * 100
        print(f"  [!] Max sheet dimension {max_w}x{max_h} "
              f"({pct_w:.0f}%/{pct_h:.0f}% of {MAX_TEXTURE_SIZE} limit)")


def main() -> None:
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    input_dir: str = os.path.join(script_dir, INPUT_DIR)
    output_dir: str = os.path.join(script_dir, OUTPUT_DIR)

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("视频转精灵图集工具")
    print(f"纹理上限: {MAX_TEXTURE_SIZE}px")
    print("=" * 60)

    # 交互式询问帧率
    while True:
        try:
            fps_input: str = input("每秒提取几帧？(推荐 8-24，默认 12): ").strip()
            if fps_input == "":
                target_fps = 12
                break
            target_fps = int(fps_input)
            if target_fps <= 0:
                print("请输入大于 0 的整数")
                continue
            if target_fps > 120:
                confirm: str = input(f"帧率 {target_fps} 较高，可能产生大量帧，确认？(y/n): ").strip().lower()
                if confirm in ("y", "yes"):
                    break
                continue
            break
        except ValueError:
            print("请输入有效的整数")
    print()

    # 情况1：input/ 根目录直接有视频文件
    video_files = get_video_files(input_dir)
    if video_files:
        print(f"检测到 {len(video_files)} 个视频文件")
        for vf in video_files:
            process_video(
                os.path.join(input_dir, vf),
                input_dir, output_dir,
                target_fps
            )
        print(f"\n{'=' * 60}")
        print(f"全部完成！")
        print(f"  精灵图集: {output_dir}/")
        print(f"  原始帧:   {FRAMES_DIR}/")
        print(f"{'=' * 60}")
        return

    # 情况2：子文件夹模式
    if os.path.isdir(input_dir):
        subdirs = [d for d in os.listdir(input_dir)
                   if os.path.isdir(os.path.join(input_dir, d))
                   and d not in ("output", "__pycache__", FRAMES_DIR)]
        if subdirs:
            print(f"检测到 {len(subdirs)} 个子文件夹，将分别处理")
            for subdir in sorted(subdirs):
                sub_input = os.path.join(input_dir, subdir)
                sub_videos = get_video_files(sub_input)
                if not sub_videos:
                    print(f"\n跳过 {subdir}/ (无视频文件)")
                    continue
                print(f"\n>>> 子文件夹: {subdir}/ ({len(sub_videos)} 个视频)")
                for vf in sub_videos:
                    process_video(
                        os.path.join(sub_input, vf),
                        sub_input, output_dir,
                        target_fps
                    )
            print(f"\n{'=' * 60}")
            print(f"全部完成！")
            print(f"  精灵图集: {output_dir}/")
            print(f"  原始帧:   {FRAMES_DIR}/")
            print(f"{'=' * 60}")
            return

    print(f"\n错误: {INPUT_DIR}/ 下未找到视频文件")
    print(f"支持的格式: .mp4, .avi, .mov, .webm, .mkv")
    print(f"请将视频放入 {INPUT_DIR}/ 文件夹后重试")


if __name__ == "__main__":
    main()
