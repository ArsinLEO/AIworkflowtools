#!/usr/bin/env python3
"""
绿幕去除工具 v3 - 移除序列帧中的绿色像素（转为透明）+ 绿边溢出抑制(despill)。

检测算法（numpy向量化 + BFS泛洪 + despill抑制）：
  阶段1 — 高置信度：greenness >= HIGH 直接移除
  阶段2 — BFS泛洪：从透明区/已移除区出发，沿 greenness >= LOW 的路径扩展
  阶段3 — despill(去绿溢出)：对保留的不透明像素，抑制绿色通道 G=min(G, max(R,B))

greenness = G - max(R, B)，纯绿(0,255,0)→255

用法：
  1. 把需要处理的PNG序列帧放入 input/ 文件夹
  2. 运行：python auto_remove_green.py
  3. 结果输出到 output/ 文件夹
"""

import os
from collections import deque
import numpy as np
from PIL import Image, ImageFilter


# ============ 配置 ============
INPUT_DIR = "input"
OUTPUT_DIR = "output"

HIGH = 30            # 阶段1：greenness >= 此值直接移除
LOW = 5              # 阶段2：BFS泛洪沿 greenness >= LOW 的路径传播
SPILL_ITERATIONS = 30  # BFS泛洪最大轮数
DESPILL_STRENGTH = 1.0  # despill强度 (1.0=完全抑制绿色)
FEATHER_RADIUS = 1      # alpha边缘羽化半径（px）


def get_png_files(folder: str) -> list[str]:
	if not os.path.isdir(folder):
		return []
	return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def process_frame(img: Image.Image) -> Image.Image:
	"""处理单帧：蒙版构建 + 透明化 + despill + 羽化。"""
	# 转为numpy数组 (h, w, 4) RGBA uint8
	arr = np.array(img.convert("RGBA"), dtype=np.uint8)
	h, w = arr.shape[0], arr.shape[1]
	alpha = arr[:, :, 3]
	r, g, b = arr[:, :, 0].astype(np.int16), arr[:, :, 1].astype(np.int16), arr[:, :, 2].astype(np.int16)

	# 绿色度: G - max(R, B)
	gn = g - np.maximum(r, b)  # shape (h, w), int16

	# === 阶段1：高置信度直接标记 ===
	mask = np.zeros((h, w), dtype=np.bool_)
	mask[(alpha >= 5) & (gn >= HIGH)] = True

	# === 阶段2：BFS泛洪 ===
	# 种子：所有透明像素（alpha<5）+ 阶段1已标记像素
	seeds = (alpha < 5) | mask  # shape (h, w)

	# 预计算邻居偏移
	neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

	# 用deque做BFS：从种子出发，沿 gn>=LOW 的不透明像素扩展
	queue = deque()
	# 初始化队列：种子中与 gn>=LOW 不透明像素相邻的透明像素
	in_queue = np.zeros((h, w), dtype=np.bool_)

	# 收集种子中需要入队的像素
	seed_ys, seed_xs = np.where(seeds)
	for i in range(len(seed_ys)):
		sy, sx = seed_ys[i], seed_xs[i]
		queue.append((sy, sx))
		in_queue[sy, sx] = True

	# 层级BFS
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
					if gn[ny, nx] >= LOW:
						mask[ny, nx] = True
						queue.append((ny, nx))
						in_queue[ny, nx] = True

	# === 阶段3：despill 抑制绿色溢出（对所有不透明像素，包括即将遮罩的） ===
	# 关键：先despill再遮罩，这样羽化拉回来的边缘像素也有正确的去绿RGB
	if DESPILL_STRENGTH > 0:
		opaque = alpha >= 5
		if opaque.any():
			g_op = g[opaque]
			r_op = r[opaque]
			b_op = b[opaque]
			max_rb = np.maximum(r_op, b_op)
			target_g = np.minimum(g_op, max_rb)
			g[opaque] = (g_op + (target_g - g_op) * DESPILL_STRENGTH).astype(np.int16)

	# 写回RGB（clamp到0-255）
	arr[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
	arr[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
	arr[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)

	# 遮罩像素alpha归零
	arr[mask, 3] = 0


	# === Alpha边缘羽化 ===
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
	print(f"greenness>={HIGH}(直删) >={LOW}(溢出)  泛洪{SPILL_ITERATIONS}轮  despill={DESPILL_STRENGTH}  羽化={FEATHER_RADIUS}px")
	print("=" * 60)

	total_px = 0
	total_removed = 0
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
	print("【绿幕去除工具 v3 - numpy向量化 + despill】")
	print()
	run_removal(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
	main()
