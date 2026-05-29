#!/usr/bin/env python3
"""
阴影去除工具 - 移除图像底部的中性黑色阴影（底部种子泛洪算法）。

适用场景：绿幕去除后，角色底部残留的投影/阴影（近纯黑、中性色温）。
不适用于：角色本体暗面（暖色调暗色区域不受影响）。

算法：
  1. 对每列，找到最底部的不透明像素
  2. 如果是"阴影候选"（暗色 + 中性色温 + 低饱和），作为种子
  3. 从种子BFS泛洪，沿同类像素传播
  4. 泛洪到达的像素 → 设为透明

影 vs 角色暗面：
  阴影：(0,1,1)  R≈G≈B≈0  |R-B|≈0  max-min≈1
  骰子暗面：(39,29,19)  R明显>B  |R-B|≈20  max-min≈20

用法：
  1. 把去过绿幕的PNG序列帧放入 input/ 文件夹
  2. 运行：python auto_remove_shadow.py
  3. 结果输出到 output/ 文件夹
"""

import os
from collections import deque
import numpy as np
from PIL import Image, ImageFilter


# ============ 配置 ============
INPUT_DIR = "input"
OUTPUT_DIR = "output"

SHADOW_MAX_BRIGHTNESS = 30  # 阴影候选：max(R,G,B) < 此值（阴影近纯黑≈1，骰子暗面≈39）
SHADOW_MAX_WARMTH = 8       # 阴影候选：|R-B| < 此值（阴影中性黑，骰子暖色R-B≈20）
SHADOW_MAX_SATURATION = 10  # 阴影候选：max-min < 此值（阴影灰度，骰子有色相）

FEATHER_RADIUS = 2          # 阴影去除后 alpha 边缘羽化（px）


def get_png_files(folder: str) -> list[str]:
	if not os.path.isdir(folder):
		return []
	return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def process_frame(img: Image.Image) -> Image.Image:
	"""处理单帧：检测并移除底部阴影。"""
	arr = np.array(img.convert("RGBA"), dtype=np.uint8)
	h, w = arr.shape[:2]
	alpha = arr[:, :, 3]

	opaque = alpha >= 5
	if not opaque.any():
		return img

	r = arr[:, :, 0].astype(int)
	g = arr[:, :, 1].astype(int)
	b = arr[:, :, 2].astype(int)

	brightness = np.maximum(np.maximum(r, g), b)
	color_range = brightness - np.minimum(np.minimum(r, g), b)

	# 阴影候选：暗色 + 中性色温 + 低饱和
	shadow_candidate = (
		opaque
		& (brightness < SHADOW_MAX_BRIGHTNESS)
		& (np.abs(r - b) < SHADOW_MAX_WARMTH)
		& (color_range < SHADOW_MAX_SATURATION)
	)
	if not shadow_candidate.any():
		return img

	# 种子：每列最底部的不透明像素，如果是阴影候选则作为种子
	seeds = np.zeros((h, w), dtype=np.bool_)
	for x in range(w):
		col_opaque = np.where(opaque[:, x])[0]
		if len(col_opaque) > 0:
			bottom_y = col_opaque[-1]
			if shadow_candidate[bottom_y, x]:
				seeds[bottom_y, x] = True

	if not seeds.any():
		return img

	# BFS 泛洪：从种子出发，沿阴影候选像素传播
	visited = np.zeros((h, w), dtype=np.bool_)
	queue = deque()
	seed_ys, seed_xs = np.where(seeds)
	for i in range(len(seed_ys)):
		queue.append((seed_ys[i], seed_xs[i]))
		visited[seed_ys[i], seed_xs[i]] = True

	neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
	while queue:
		cy, cx = queue.popleft()
		for dy, dx in neighbors:
			ny, nx = cy + dy, cx + dx
			if 0 <= ny < h and 0 <= nx < w:
				if visited[ny, nx]:
					continue
				if shadow_candidate[ny, nx]:
					visited[ny, nx] = True
					queue.append((ny, nx))

	# 移除阴影像素
	arr[visited, 3] = 0

	# Alpha 边缘羽化
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
	print(f"阴影候选: brightness<{SHADOW_MAX_BRIGHTNESS}  |R-B|<{SHADOW_MAX_WARMTH}  sat<{SHADOW_MAX_SATURATION}")
	print(f"羽化={FEATHER_RADIUS}px")
	print("=" * 60)

	total_removed = 0
	os.makedirs(output_dir, exist_ok=True)

	for i, fname in enumerate(files):
		src_path = os.path.join(input_dir, fname)
		dst_path = os.path.join(output_dir, fname)

		img = Image.open(src_path)
		arr_before = np.array(img.convert("RGBA"))
		before_count = (arr_before[:, :, 3] >= 5).sum()

		result = process_frame(img)

		arr_after = np.array(result)
		after_count = (arr_after[:, :, 3] >= 5).sum()
		removed = before_count - after_count
		total_removed += removed

		result.save(dst_path)

		if (i + 1) % 20 == 0 or (i + 1) == len(files):
			print(f"  {i + 1}/{len(files)} 完成 (累计移除: {total_removed} px)")

	print(f"\n完成！总计移除 {total_removed} 阴影像素, 输出: {output_dir}")


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
	print("【阴影去除工具 - 底部种子泛洪】")
	print()
	run_removal(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
	main()
