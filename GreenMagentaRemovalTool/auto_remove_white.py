#!/usr/bin/env python3
"""
白幕去除工具 v2 - 距离变换保护主体 + BFS泛洪 + 边缘羽化。

核心思路：白色天然存在于角色内部（高光、白衬衫、眼白），不能像绿幕那样
从所有透明区泛洪。v2 引入距离变换（distance transform）——先计算每个像素到
最近角色像素的距离，只有"深背景"透明区（距角色足够的）才作为 BFS 种子。
角色内部的白色区域被距离保护，不会被误删。

管线：
  阶段0 — 距离变换：BFS 计算每个像素到最近不透明像素的切比雪夫距离
  阶段1 — 高置信度直删：whiteness >= HIGH
  阶段2 — BFS 泛洪：仅从"深背景"透明区（距离 > DISTANCE_THRESHOLD）+
           阶段1 已标记区出发，沿 whiteness >= LOW 扩展
  阶段3 — Alpha 羽化

whiteness = min(R, G, B)，纯白(255,255,255)→255

用法：
  1. PNG 序列帧放入 input/ 文件夹
  2. 运行：python auto_remove_white.py
  3. 结果输出到 output/
"""

import os
from collections import deque
import numpy as np
from PIL import Image, ImageFilter


# ============ 配置 ============
INPUT_DIR = "input"
OUTPUT_DIR = "output"

HIGH = 200                 # 阶段1：whiteness >= 此值直接判定为白幕
LOW = 60                   # 阶段2：BFS 沿 whiteness >= LOW 传播（越低越激进）
DISTANCE_THRESHOLD = 15    # 距角色 > 此值(px)的透明区才算"深背景"，可做泛洪种子
SPILL_ITERATIONS = 20      # BFS 泛洪最大轮数
FEATHER_RADIUS = 1         # Alpha 边缘羽化半径（px）


def get_png_files(folder: str) -> list[str]:
	if not os.path.isdir(folder):
		return []
	return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def compute_distance_transform(alpha: np.ndarray) -> np.ndarray:
	"""BFS：每个像素到最近不透明像素的切比雪夫距离。

	返回 shape (h, w) int32 数组。不透明像素自身距离 = 0。
	"""
	h, w = alpha.shape
	dist = np.full((h, w), 999999, dtype=np.int32)
	queue = deque()

	# 种子：所有不透明像素
	opaque_ys, opaque_xs = np.where(alpha >= 5)
	for i in range(len(opaque_ys)):
		y, x = opaque_ys[i], opaque_xs[i]
		dist[y, x] = 0
		queue.append((y, x))

	if not queue:
		return dist

	neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
	while queue:
		cy, cx = queue.popleft()
		nd = dist[cy, cx] + 1
		for dy, dx in neighbors:
			ny, nx = cy + dy, cx + dx
			if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] == 999999:
				dist[ny, nx] = nd
				queue.append((ny, nx))

	return dist


def process_frame(img: Image.Image) -> Image.Image:
	"""处理单帧：距离变换 → 蒙版构建 → 透明化 → 羽化。"""
	arr = np.array(img.convert("RGBA"), dtype=np.uint8)
	h, w = arr.shape[0], arr.shape[1]
	alpha = arr[:, :, 3]
	r = arr[:, :, 0].astype(np.int16)
	g = arr[:, :, 1].astype(np.int16)
	b = arr[:, :, 2].astype(np.int16)

	# 白色度：min(R, G, B)
	wn = np.minimum(np.minimum(r, g), b)

	# === 阶段0：距离变换（保护角色主体） ===
	dist = compute_distance_transform(alpha)

	# === 阶段1：高置信度直删 ===
	mask = np.zeros((h, w), dtype=np.bool_)
	mask[(alpha >= 5) & (wn >= HIGH)] = True

	# === 阶段2：BFS 泛洪（仅从深背景出发） ===
	# 种子 = 深背景透明区（距角色远） + 阶段1 已标记区
	seeds = mask.copy()
	deep_bg = (alpha < 5) & (dist > DISTANCE_THRESHOLD)
	seeds |= deep_bg

	neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
	queue = deque()
	in_queue = np.zeros((h, w), dtype=np.bool_)
	seed_ys, seed_xs = np.where(seeds)
	for i in range(len(seed_ys)):
		sy, sx = seed_ys[i], seed_xs[i]
		queue.append((sy, sx))
		in_queue[sy, sx] = True

	for _ in range(SPILL_ITERATIONS):
		if not queue:
			break
		for _ in range(len(queue)):
			cy, cx = queue.popleft()
			for dy, dx in neighbors:
				ny, nx = cy + dy, cx + dx
				if 0 <= ny < h and 0 <= nx < w:
					if in_queue[ny, nx] or mask[ny, nx]:
						continue
					if alpha[ny, nx] < 5:
						continue
					if wn[ny, nx] >= LOW:
						mask[ny, nx] = True
						queue.append((ny, nx))
						in_queue[ny, nx] = True

	# 写回RGB
	arr[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
	arr[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
	arr[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)

	# 遮罩像素 alpha 归零
	arr[mask, 3] = 0

	# === Alpha 边缘羽化 ===
	if FEATHER_RADIUS > 0:
		result_img = Image.fromarray(arr, 'RGBA')
		alpha_ch = result_img.getchannel('A')
		alpha_blurred = alpha_ch.filter(ImageFilter.GaussianBlur(FEATHER_RADIUS))
		result_img.putalpha(alpha_blurred)
		return result_img

	return Image.fromarray(arr, 'RGBA')


def process_folder(input_dir: str, output_dir: str) -> None:
	files = get_png_files(input_dir)
	if not files:
		print(f"错误: {input_dir}/ 下未找到PNG文件")
		return

	print(f"检测到 {len(files)} 张PNG")
	print(f"whiteness>={HIGH}(直删) >={LOW}(泛洪)  深背景>{DISTANCE_THRESHOLD}px  "
		f"泛洪{SPILL_ITERATIONS}轮  羽化={FEATHER_RADIUS}px")
	print("=" * 60)

	os.makedirs(output_dir, exist_ok=True)

	for i, fname in enumerate(files):
		src_path = os.path.join(input_dir, fname)
		dst_path = os.path.join(output_dir, fname)

		img = Image.open(src_path)
		result = process_frame(img)
		result.save(dst_path)

		if (i + 1) % 20 == 0 or (i + 1) == len(files):
			print(f"  {i + 1}/{len(files)} 完成")

	print(f"\n完成！输出: {output_dir}")


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
	print("【白幕去除工具 v2 - 距离变换保护主体】")
	print()
	run_removal(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
	main()
