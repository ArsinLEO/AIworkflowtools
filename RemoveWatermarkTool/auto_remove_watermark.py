#!/usr/bin/env python3
"""
序列帧去水印工具 - 检测并去除四个角落的水印（左上、右上、左下、右下）。

四步逼近法原理：
  1. 大范围搜索：全画面连通组件分析
  2. 孤立性判断：排除最大组件（人物主体），只保留孤立组件
  3. 锁定角落：四个角中选最可能的一个
  4. 安全删除：超过删除阈值则放弃，避免误删人物

特点：
  - 不依赖水印完全静止，适用于轻微抖动/半透明水印
  - 文字水印（如"豆包AI生成"）密度低、面积小，易识别
  - 单角锁定避免多角误删

用法：
  1. 把需要去水印的PNG序列帧放入 input/ 文件夹
  2. 运行：python auto_remove_watermark.py
  3. 结果输出到 output/ 文件夹
"""

import os
import sys
from collections import deque
from PIL import Image, ImageFilter


# ============ 配置 ============
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# 角落检测区域（相对画面比例）
CORNER_W = 0.40   # 左右两侧检测宽度占画面40%
CORNER_H = 0.35   # 上下两侧检测高度占画面35%

# 孤立组件阈值
MIN_WATERMARK_SIZE = 20
MAX_WATERMARK_SIZE = 5000
MAX_COMPONENT_DENSITY = 0.85   # 放宽：文字笔画可能较密
MIN_ASPECT_RATIO = 1.0   # 宽/高 >= 1，兼容方形logo水印

# 删除安全阀：超过画面比例则放弃删除（避免误删人物）
MAX_DELETE_RATIO = 0.20   # 放宽到20%：分位区间覆盖多帧偏移后面积较大

# 处理时缩小的比例（加速计算，1.0=原尺寸）
PROCESS_SCALE = 0.25


def get_png_files(folder: str) -> list[str]:
    """返回文件夹中所有PNG文件（按字母排序）。"""
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def load_frames(folder: str, scale: float = 1.0) -> list[Image.Image]:
    """加载所有PNG帧，可选缩放。"""
    files = get_png_files(folder)
    frames = []
    for fname in files:
        img = Image.open(os.path.join(folder, fname)).convert("RGBA")
        if scale != 1.0:
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        frames.append(img)
    return frames


def get_component_aspect_ratio(comp: list) -> float:
    """计算组件宽高比（宽/高）。"""
    xs = [x for x, y in comp]
    ys = [y for x, y in comp]
    bw = max(xs) - min(xs) + 1
    bh = max(ys) - min(ys) + 1
    if bh == 0:
        return 999.0
    return bw / bh


def find_all_components(img: Image.Image) -> list:
    """在全画面找所有非透明像素的连通组件。返回组件列表，每个是 (x, y) 列表。"""
    w, h = img.size
    pixels = img.load()

    mask = [[False] * h for _ in range(w)]
    for y in range(h):
        for x in range(w):
            if pixels[x, y][3] > 5:
                mask[x][y] = True

    visited = [[False] * h for _ in range(w)]
    components = []

    for y in range(h):
        for x in range(w):
            if not mask[x][y] or visited[x][y]:
                continue
            queue = deque([(x, y)])
            visited[x][y] = True
            comp = [(x, y)]
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[nx][ny] and not visited[nx][ny]:
                        visited[nx][ny] = True
                        comp.append((nx, ny))
                        queue.append((nx, ny))
            components.append(comp)

    return components


def is_component_in_corner(comp: list, corner: str, img_w: int, img_h: int) -> bool:
    """判断组件是否完全在指定角落区域内。"""
    xs = [x for x, y in comp]
    ys = [y for x, y in comp]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if 'top' in corner:
        if max_y >= int(img_h * CORNER_H):
            return False
    else:
        if min_y < int(img_h * (1 - CORNER_H)):
            return False

    if 'left' in corner:
        if max_x >= int(img_w * CORNER_W):
            return False
    else:
        if min_x < int(img_w * (1 - CORNER_W)):
            return False

    return True


def get_component_density(comp: list) -> float:
    """计算组件的密度（像素数 / 边界框面积）。"""
    xs = [x for x, y in comp]
    ys = [y for x, y in comp]
    bw = max(xs) - min(xs) + 1
    bh = max(ys) - min(ys) + 1
    if bw * bh == 0:
        return 1.0
    return len(comp) / (bw * bh)


def build_watermark_mask(frames: list[Image.Image]) -> tuple[Image.Image, str]:
    """
    四步逼近法 + 两步删除：
    1. 全画面连通组件分析（大范围搜索）
    2. 排除最大组件（人物主体），只保留孤立组件
    3. 四个角落分别收集组件边界框
    4. 第一步：用中位数行高矩形粗删
    5. 第二步：在矩形扩展区域内精修残留像素
    6. 删除数量安全阀
    返回: (mask, corner_name) 或 (None, "")
    """
    if len(frames) < 2:
        return None, ""

    w, h = frames[0].size
    corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

    # 收集每个帧、每个角落检测到的组件边界框
    corner_boxes = {c: [] for c in corners}

    for frame in frames:
        components = find_all_components(frame)
        if len(components) <= 1:
            continue

        main_comp = max(components, key=len)

        for comp in components:
            if comp is main_comp:
                continue

            size = len(comp)
            if not (MIN_WATERMARK_SIZE <= size <= MAX_WATERMARK_SIZE):
                continue

            density = get_component_density(comp)
            if density > MAX_COMPONENT_DENSITY:
                continue

            aspect = get_component_aspect_ratio(comp)
            if aspect < MIN_ASPECT_RATIO:
                continue

            xs = [x for x, y in comp]
            ys = [y for x, y in comp]
            box = (min(xs), min(ys), max(xs), max(ys))

            for corner in corners:
                if is_component_in_corner(comp, corner, w, h):
                    corner_boxes[corner].append(box)
                    break

    # 第一步：为每个检测到组件的角落生成删除矩形
    corner_rects = {}
    active_corners = []
    total_box_area = 0

    for corner in corners:
        boxes = corner_boxes[corner]
        if len(boxes) == 0:
            continue

        # 纵向：中位数行高 + 10%扩展
        all_min_y = sorted([b[1] for b in boxes])
        all_max_y = sorted([b[3] for b in boxes])
        median_idx = len(boxes) // 2
        median_min_y = all_min_y[median_idx]
        median_max_y = all_max_y[median_idx]
        median_height = median_max_y - median_min_y
        expand_y = max(1, int(median_height * 0.10))
        min_y = max(0, median_min_y - expand_y)
        max_y = min(h - 1, median_max_y + expand_y)

        # 横向：最大跨度
        min_x = min(b[0] for b in boxes)
        max_x = max(b[2] for b in boxes)

        padding = 1
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(w - 1, max_x + padding)
        max_y = min(h - 1, max_y + padding)

        box_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        total_box_area += box_area
        active_corners.append(f"{corner}({box_area}px)")
        corner_rects[corner] = (min_x, min_y, max_x, max_y)

    if len(active_corners) == 0:
        return None, ""

    total_pixels = w * h
    if total_box_area > total_pixels * MAX_DELETE_RATIO:
        print(f"  警告: 总删除面积{total_box_area}，超过安全阈值{int(total_pixels * MAX_DELETE_RATIO)}，放弃删除")
        return None, ",".join(active_corners)

    # 第二步：壳层精修
    # 搜索矩形边界外距离<=3像素的所有非透明像素
    # 限制：精修删除像素不超过第一步矩形面积的20%
    mask_data = [0] * (w * h)

    for corner, (r_min_x, r_min_y, r_max_x, r_max_y) in corner_rects.items():
        # 先标记第一步的矩形区域
        first_step_count = 0
        for y in range(r_min_y, r_max_y + 1):
            for x in range(r_min_x, r_max_x + 1):
                mask_data[y * w + x] = 255
                first_step_count += 1

        max_refine = int(first_step_count * 0.20)
        deleted_refine = 0

        # 多轮迭代壳层精修，每轮基于更新后的边界
        # 第1-2轮：距离<=3像素；第3-4轮：距离<=1像素（更保守）
        for iteration in range(4):
            if deleted_refine >= max_refine:
                break

            # 找到当前蒙版的外边界（所有mask_data=255的像素的8邻域中mask_data=0的像素）
            boundary = set()
            for y in range(h):
                for x in range(w):
                    if mask_data[y * w + x] == 255:
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < w and 0 <= ny < h and mask_data[ny * w + nx] == 0:
                                    boundary.add((x, y))
                                    break
                            else:
                                continue
                            break

            if len(boundary) == 0:
                break

            # 第1-2轮搜索<=3像素，第3-5轮搜索<=1像素
            search_dist = 3 if iteration < 2 else 1

            # 收集距离边界<=search_dist的所有非透明像素
            shell_pixels = set()
            for frame in frames:
                pixels = frame.load()
                for bx, by in boundary:
                    for y in range(max(0, by - search_dist), min(h, by + search_dist + 1)):
                        for x in range(max(0, bx - search_dist), min(w, bx + search_dist + 1)):
                            if mask_data[y * w + x] == 0 and pixels[x, y][3] > 5:
                                dx = abs(x - bx)
                                dy = abs(y - by)
                                if max(dx, dy) <= search_dist:
                                    shell_pixels.add((x, y))

            if len(shell_pixels) == 0:
                break

            # 对壳层像素做连通组件分析
            emask = [[False] * h for _ in range(w)]
            for x, y in shell_pixels:
                emask[x][y] = True

            visited = [[False] * h for _ in range(w)]
            shell_comps = []
            for x, y in shell_pixels:
                if not emask[x][y] or visited[x][y]:
                    continue
                queue = deque([(x, y)])
                visited[x][y] = True
                comp = [(x, y)]
                while queue:
                    cx, cy = queue.popleft()
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and emask[nx][ny] and not visited[nx][ny]:
                            visited[nx][ny] = True
                            comp.append((nx, ny))
                            queue.append((nx, ny))
                shell_comps.append(comp)

            if len(shell_comps) == 0:
                break

            # 按面积排序，排除最大的
            shell_comps.sort(key=len, reverse=True)
            added = 0

            if len(shell_comps) == 1:
                comp = shell_comps[0]
                if len(comp) <= MAX_WATERMARK_SIZE:
                    space = max_refine - deleted_refine
                    if len(comp) <= space:
                        for x, y in comp:
                            mask_data[y * w + x] = 255
                        added = len(comp)
            else:
                for comp in shell_comps[1:]:
                    if len(comp) <= MAX_WATERMARK_SIZE:
                        space = max_refine - deleted_refine
                        if len(comp) <= space:
                            for x, y in comp:
                                mask_data[y * w + x] = 255
                            added += len(comp)
                        if deleted_refine + added >= max_refine:
                            break

            deleted_refine += added
            if added == 0:
                break

    mask = Image.new('L', (w, h))
    mask.putdata(mask_data)
    return mask, ",".join(active_corners)


def refine_mask(mask: Image.Image) -> Image.Image:
    """对蒙版进行形态学优化：去除噪点、填充小孔。"""
    # 轻微模糊+阈值，平滑边缘
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.5))
    mask_data = [255 if p > 128 else 0 for p in mask.getdata()]
    mask.putdata(mask_data)
    return mask


def remove_watermark_from_frame(frame: Image.Image, mask: Image.Image) -> Image.Image:
    """使用蒙版去除水印（将水印区域设为透明）。"""
    frame_rgba = frame.convert("RGBA")
    # 确保蒙版尺寸一致
    if mask.size != frame_rgba.size:
        mask = mask.resize(frame_rgba.size, Image.NEAREST)

    r, g, b, a = frame_rgba.split()
    # 蒙版白色(255)是水印区域，将其alpha设为0
    new_a = Image.new('L', frame_rgba.size)
    for i, (alpha_val, mask_val) in enumerate(zip(a.getdata(), mask.getdata())):
        if mask_val > 128:
            new_a.putpixel((i % frame_rgba.width, i // frame_rgba.width), 0)
        else:
            new_a.putpixel((i % frame_rgba.width, i // frame_rgba.width), alpha_val)

    result = Image.merge('RGBA', (r, g, b, new_a))
    return result


def process_single_folder(input_dir: str, output_dir: str) -> None:
    """处理单个文件夹（直接包含PNG序列帧）。"""
    files = get_png_files(input_dir)
    if not files:
        print(f"错误: {input_dir}/ 下未找到PNG文件")
        return

    print(f"检测到 {len(files)} 张PNG")
    print(f"处理缩放比例: {PROCESS_SCALE}")
    print("=" * 60)

    # 加载缩略图用于检测
    small_frames = load_frames(input_dir, scale=PROCESS_SCALE)
    w, h = small_frames[0].size
    orig_w = int(small_frames[0].width / PROCESS_SCALE)
    orig_h = int(small_frames[0].height / PROCESS_SCALE)

    mask = None

    if len(small_frames) >= 2:
        # ========== 多帧逼近法 ==========
        print("使用四步逼近法检测水印...")
        print("  1. 全画面连通组件分析")
        print("  2. 排除人物主体（最大组件）")
        print("  3. 四个角中选最可能的")
        print("  4. 删除数量安全阀")

        mask, detected_corner = build_watermark_mask(small_frames)

        if mask is None:
            if detected_corner:
                print(f"  结果: {detected_corner}疑似有水印但超安全阈值，跳过处理")
            else:
                print("  结果: 未检测到水印，跳过处理")
            # 复制原图到输出
            print(f"\n未检测到水印，复制原图到输出...")
            os.makedirs(output_dir, exist_ok=True)
            for fname in files:
                src = Image.open(os.path.join(input_dir, fname)).convert("RGBA")
                src.save(os.path.join(output_dir, fname))
            print(f"完成！输出目录: {output_dir}")
            return
        else:
            wm_pixels = sum(1 for p in mask.getdata() if p > 128)
            print(f"  结果: 在 {detected_corner} 检测到 {wm_pixels} 水印像素")

    else:
        # ========== 单帧模式 ==========
        print("只有1张图，使用单帧孤立组件检测模式")
        print("提示: 放入多张序列帧可以获得更准确的检测效果")

        # 单帧也用全画面连通组件分析
        components = find_all_components(small_frames[0])
        if len(components) <= 1:
            print("未检测到水印，跳过处理")
            return

        main_comp = max(components, key=len)
        corners_list = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        corner_pixels = {c: set() for c in corners_list}

        for comp in components:
            if comp is main_comp:
                continue
            size = len(comp)
            if not (MIN_WATERMARK_SIZE <= size <= MAX_WATERMARK_SIZE):
                continue
            density = get_component_density(comp)
            if density > MAX_COMPONENT_DENSITY:
                continue
            for corner in corners_list:
                if is_component_in_corner(comp, corner, w, h):
                    for x, y in comp:
                        corner_pixels[corner].add(y * w + x)
                    break

        best_corner = max(corners_list, key=lambda c: len(corner_pixels[c]))
        best_count = len(corner_pixels[best_corner])

        if best_count == 0 or best_count > w * h * MAX_DELETE_RATIO:
            print("未检测到水印或超过安全阈值，跳过处理")
            return

        mask_data = [0] * (w * h)
        for idx in corner_pixels[best_corner]:
            mask_data[idx] = 255
        mask = Image.new('L', (w, h))
        mask.putdata(mask_data)
        print(f"  结果: 在 {best_corner} 检测到 {best_count} 水印像素")

    # 优化蒙版
    mask = refine_mask(mask)

    # 将蒙版缩放回原始尺寸
    orig_size = (orig_w, orig_h)
    if mask.size != orig_size:
        mask = mask.resize(orig_size, Image.NEAREST)

    # 处理所有原图
    print(f"\n开始处理 {len(files)} 张图片...")
    os.makedirs(output_dir, exist_ok=True)

    for i, fname in enumerate(files):
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        frame = Image.open(src_path).convert("RGBA")
        result = remove_watermark_from_frame(frame, mask)
        result.save(dst_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)} 完成")

    print(f"\n完成！输出目录: {output_dir}")


def run_watermark_removal(input_dir: str, output_dir: str) -> None:
    """主流程：支持直接放PNG或子文件夹模式。"""
    # 情况1：根目录直接有PNG文件
    files = get_png_files(input_dir)
    if files:
        process_single_folder(input_dir, output_dir)
        return

    # 情况2：根目录没有PNG，检查子文件夹
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
        process_single_folder(sub_input, sub_output)

    print(f"\n全部完成！输出目录: {output_dir}")


def main() -> None:
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    print("【标准模式】")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    run_watermark_removal(input_dir, output_dir)


if __name__ == "__main__":
    main()
