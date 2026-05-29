#!/usr/bin/env python3
"""
绿幕/洋红幕去除工具 v3 - numpy向量化 + BFS泛洪 + despill色边抑制。

检测算法：
  阶段1 — 高置信度：chroma >= HIGH 直接移除
  阶段2 — BFS泛洪：从透明区出发，沿 chroma >= LOW 的路径扩展
  阶段3 — despill：先对所有不透明像素抑制色偏通道，再应用蒙版+羽化

greenness  = G - max(R, B)，纯绿(0,255,0)→255
magentaness = min(R, B) - G，纯洋红(255,0,255)→255

用法：
  1. 把需要处理的PNG序列帧放入 input/ 文件夹
  2. 运行：python auto_remove_green_magenta.py
  3. 结果输出到 output/ 文件夹
"""

import os
from collections import deque
import numpy as np
from PIL import Image, ImageFilter


# ============ 配置 ============
INPUT_DIR = "input"
OUTPUT_DIR = "output"

GREEN_HIGH = 30
MAGENTA_HIGH = 30

GREEN_LOW = 5
MAGENTA_LOW = 5

SPILL_ITERATIONS = 30
DESPILL_STRENGTH = 1.0
FEATHER_RADIUS = 1

REMOVE_GREEN = True
REMOVE_MAGENTA = True


def get_png_files(folder: str) -> list[str]:
	if not os.path.isdir(folder):
		return []
	return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def process_frame(img: Image.Image) -> Image.Image:
	"""处理单帧：蒙版构建 + despill + 透明化 + 羽化。"""
	arr = np.array(img.convert("RGBA"), dtype=np.uint8)
	h, w = arr.shape[0], arr.shape[1]
	alpha = arr[:, :, 3]
	r = arr[:, :, 0].astype(np.int16)
	g = arr[:, :, 1].astype(np.int16)
	b = arr[:, :, 2].astype(np.int16)

	mask = np.zeros((h, w), dtype=np.bool_)
	opaque_mask = alpha >= 5

	# === 阶段1：高置信度直接标记 ===
	if REMOVE_GREEN:
		gn = g - np.maximum(r, b)
		mask |= opaque_mask & (gn >= GREEN_HIGH)

	if REMOVE_MAGENTA:
		mn = np.minimum(r, b) - g
		mask |= opaque_mask & (mn >= MAGENTA_HIGH)

	# === 阶段2：BFS泛洪 ===
	# 计算传播用的色度值（绿色度或洋红度，取较大的那个）
	propagation = np.zeros((h, w), dtype=np.int16)
	if REMOVE_GREEN:
		propagation = np.maximum(propagation, gn)
	if REMOVE_MAGENTA:
		propagation = np.maximum(propagation, mn)

	seeds = (alpha < 5) | mask
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
					if propagation[ny, nx] >= GREEN_LOW if REMOVE_GREEN else propagation[ny, nx] >= MAGENTA_LOW:
						mask[ny, nx] = True
						queue.append((ny, nx))
						in_queue[ny, nx] = True

	# === 阶段3：despill 抑制色偏（对所有不透明像素，包括即将遮罩的） ===
	# 关键：先despill再遮罩，这样羽化拉回来的边缘像素也有正确的去色RGB
	if DESPILL_STRENGTH > 0 and opaque_mask.any():
		if REMOVE_GREEN:
			# G = min(G, max(R, B)) 抑制绿色
			g_excess = g > np.maximum(r, b)
			needs_despill = opaque_mask & g_excess
			if needs_despill.any():
				max_rb = np.maximum(r[needs_despill], b[needs_despill])
				g[needs_despill] = (g[needs_despill] + (max_rb - g[needs_despill]) * DESPILL_STRENGTH).astype(np.int16)

		if REMOVE_MAGENTA:
			# R = B = min(R, B, G + (R-G)*(1-strength)) 抑制洋红
			m_excess = np.minimum(r, b) > g
			needs_despill_m = opaque_mask & m_excess
			if needs_despill_m.any():
				g_m = g[needs_despill_m]
				r_m = r[needs_despill_m]
				b_m = b[needs_despill_m]
				r[needs_despill_m] = (r_m + (g_m - r_m) * DESPILL_STRENGTH).astype(np.int16)
				b[needs_despill_m] = (b_m + (g_m - b_m) * DESPILL_STRENGTH).astype(np.int16)

	# 写回RGB
	arr[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
	arr[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
	arr[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)

	# 遮罩像素alpha归零
	arr[mask, 3] = 0

	# Alpha边缘羽化
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
	mode = []
	if REMOVE_GREEN:
		mode.append(f"绿>={GREEN_HIGH}/{GREEN_LOW}")
	if REMOVE_MAGENTA:
		mode.append(f"洋红>={MAGENTA_HIGH}/{MAGENTA_LOW}")
	print(f"{' + '.join(mode)}  泛洪{SPILL_ITERATIONS}轮  despill={DESPILL_STRENGTH}  羽化={FEATHER_RADIUS}px")
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
	print("【绿幕/洋红幕去除工具 v3 - numpy向量化 + despill】")
	print()
	run_removal(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
	main()
