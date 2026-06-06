"""
BlackWhiteMattingTool — 黑白底差法抠图工具

原理：
  I_white = α×F + (1-α)×255
  I_black = α×F + (1-α)×0
  → α = 1 - (I_white - I_black) / 255

采用差法剪影 + 形态学填洞 + Trimap 策略：
  Step 1: 差法计算初始 α（利用双图关系精确区分角色/背景）
  Step 2: 低阈值二值化 → 宽松角色轮廓
  Step 3: 形态学闭运算填洞 → 完整无洞剪影
  Step 4: 连通分量去噪 → 丢弃背景碎片
  Step 5: 腐蚀/膨胀 → 核心主体/边缘带
  Step 6: 边缘带用差法 α 精算，核心区锁定 α=1

用法：将黑底图放入 input/black/，白底图放入 input/white/，运行脚本即可。
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage


# ============================================================
# 可调参数
# ============================================================

# 剪影二值化阈值：差法 α 高于此值 → 纳入初始剪影
# 设低一点（0.3）可捕获更多角色区域，后续靠闭运算填洞
SILHOUETTE_ALPHA_THRESHOLD = 0.3

# 形态学闭运算迭代次数（填洞用）
CLOSE_ITERATIONS = 12

# 连通分量最小面积（小于此值的碎片丢弃）
MIN_COMPONENT_SIZE = 500

# Trimap 腐蚀/膨胀半径
ERODE_RADIUS = 6
DILATE_RADIUS = 6

# 边缘区平滑次数
EDGE_SMOOTH_PASSES = 2


# ============================================================
# 核心算法
# ============================================================

def compute_alpha(img_white: np.ndarray, img_black: np.ndarray) -> np.ndarray:
    """差法计算 alpha（每像素独立，基于两张图的关系）"""
    diff = img_white.astype(np.float32) - img_black.astype(np.float32)
    alpha_per_ch = 1.0 - diff / 255.0
    return np.clip(np.max(alpha_per_ch, axis=2), 0.0, 1.0)


def build_silhouette_mask(alpha_diff: np.ndarray) -> np.ndarray:
    """
    从差法 alpha 构建完整角色剪影。
    差法基于双图关系，天然能区分：
      - 角色暗部（两图都暗 → 差小 → α高） ✓ 保留
      - 脚底阴影（黑底暗、白底白 → 差大 → α低） ✓ 排除
    低阈值 + 形态学闭运算确保角色内部无洞。
    """
    # 低阈值二值化 — 宽松纳入
    binary = alpha_diff > SILHOUETTE_ALPHA_THRESHOLD

    # 形态学闭运算 — 填洞
    struct = ndimage.generate_binary_structure(2, 2)
    closed = ndimage.binary_closing(binary, structure=struct,
                                    iterations=CLOSE_ITERATIONS)

    # 连通分量 — 只保留主体
    labeled, n_labels = ndimage.label(closed)
    label_sizes = np.bincount(labeled.ravel())
    if len(label_sizes) > 1:
        keep_labels = np.where(label_sizes[1:] > MIN_COMPONENT_SIZE)[0] + 1
        mask_clean = np.isin(labeled, keep_labels)
    else:
        mask_clean = closed

    # 填充剩余孔洞
    return ndimage.binary_fill_holes(mask_clean)


def build_trimap(silhouette: np.ndarray) -> tuple:
    """从完整剪影构建 Trimap (core_fg, core_bg, edge_mask)"""
    struct = ndimage.generate_binary_structure(2, 2)

    core_fg = ndimage.binary_erosion(
        silhouette, structure=struct, iterations=ERODE_RADIUS)
    dilated = ndimage.binary_dilation(
        silhouette, structure=struct, iterations=DILATE_RADIUS)
    core_bg = ~dilated
    edge_mask = ~core_fg & ~core_bg

    return core_fg, core_bg, edge_mask


def process_pair(white_path: Path, black_path: Path, output_dir: Path,
                  rel_path: Path) -> dict:
    """处理一对黑白底图，输出 alpha 预览 + RGBA 抠图结果"""
    img_w = Image.open(white_path).convert("RGB")
    img_b = Image.open(black_path).convert("RGB")

    if img_w.size != img_b.size:
        return {"error": f"尺寸不一致: 白底{img_w.size} vs 黑底{img_b.size}"}

    arr_w = np.array(img_w, dtype=np.float32)
    arr_b = np.array(img_b, dtype=np.float32)

    # Step 1-3: 差法 → 宽松二值化 → 闭运算填洞 → 完整剪影
    alpha_diff = compute_alpha(arr_w, arr_b)
    silhouette = build_silhouette_mask(alpha_diff)

    # Step 4: Trimap
    core_fg, core_bg, edge_mask = build_trimap(silhouette)

    # Step 5: 最终 alpha — 核心区锁定，边缘带用差法精算
    H, W = alpha_diff.shape
    alpha = np.zeros((H, W), dtype=np.float32)
    alpha[core_fg] = 1.0
    alpha[edge_mask] = alpha_diff[edge_mask]

    # 边缘带局部平滑
    for _ in range(EDGE_SMOOTH_PASSES):
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        smoothed = ndimage.convolve(
            alpha.astype(np.float64), kernel.astype(np.float64)
        ).astype(np.float32)
        alpha = np.where(edge_mask, smoothed, alpha)

    alpha = np.clip(alpha, 0.0, 1.0)

    # Alpha 预览 — 灰度图
    alpha_path = output_dir / "alpha" / rel_path.with_suffix(".png")
    alpha_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((alpha * 255).astype(np.uint8), "L").save(alpha_path)

    # RGBA 抠图结果 — RGB 取自黑底图
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(arr_b, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)

    rgba_path = output_dir / "rgba" / rel_path.with_suffix(".png")
    rgba_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(rgba_path)

    return {
        "core_fg_pct": core_fg.mean() * 100,
        "core_bg_pct": core_bg.mean() * 100,
        "edge_pct": edge_mask.mean() * 100,
    }


# ============================================================
# 文件发现与分发
# ============================================================

def find_pairs(input_dir: Path) -> list[tuple[Path, Path, Path]]:
    """扫描 input/black/ 和 input/white/，按文件名匹配配对"""
    black_dir = input_dir / "black"
    white_dir = input_dir / "white"

    if not black_dir.exists() or not white_dir.exists():
        return []

    black_files = {f.relative_to(black_dir): f
                   for f in sorted(black_dir.rglob("*.png"))}
    white_files = {f.relative_to(white_dir): f
                   for f in sorted(white_dir.rglob("*.png"))}

    common = sorted(set(black_files.keys()) & set(white_files.keys()))

    return [(white_files[k], black_files[k], k) for k in common]


# ============================================================
# 主流程
# ============================================================

def main():
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"

    (input_dir / "black").mkdir(parents=True, exist_ok=True)
    (input_dir / "white").mkdir(parents=True, exist_ok=True)
    (output_dir / "alpha").mkdir(parents=True, exist_ok=True)
    (output_dir / "rgba").mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(input_dir)

    print("=" * 56)
    print("  黑白底差法抠图工具")
    print("=" * 56)

    if not pairs:
        print("\n[错误] 未找到匹配的图片对。")
        print(f"  请将黑底图放入: {input_dir / 'black'}")
        print(f"  请将白底图放入: {input_dir / 'white'}")
        print("  文件名一致即可自动配对（支持子目录）。")
        return

    folders: dict[str, int] = {}
    for _, _, rel_path in pairs:
        folder = rel_path.parent.as_posix() if rel_path.parent.as_posix() != "." else "(根目录)"
        folders[folder] = folders.get(folder, 0) + 1

    print(f"\n  找到 {len(pairs)} 对匹配图片:")
    for folder, count in folders.items():
        print(f"    {folder}: {count} 张")

    first_white, first_black, _ = pairs[0]
    ref = Image.open(first_white)
    print(f"\n  参考尺寸: {ref.size[0]} x {ref.size[1]}")
    print(f"  剪影阈值: α>{SILHOUETTE_ALPHA_THRESHOLD}")
    print(f"  闭运算: {CLOSE_ITERATIONS}次 | 腐蚀: {ERODE_RADIUS}px | 膨胀: {DILATE_RADIUS}px")
    print(f"  最小组件: {MIN_COMPONENT_SIZE}px")

    confirm = input("\n确认开始处理？(y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("已取消。")
        return

    print(f"\n开始处理 {len(pairs)} 对图片...\n")
    done, skipped = 0, 0

    for white_path, black_path, rel_path in pairs:
        result = process_pair(white_path, black_path, output_dir, rel_path)

        if "error" in result:
            print(f"  [跳过] {rel_path} — {result['error']}")
            skipped += 1
        else:
            print(f"  [{done+1}/{len(pairs)}] {rel_path}  "
                  f"核心{result['core_fg_pct']:.0f}% "
                  f"边缘{result['edge_pct']:.1f}%")
            done += 1

    print(f"\n{'=' * 56}")
    print(f"  完成: {done} 张 | 跳过: {skipped} 张")
    print(f"  Alpha 预览: {output_dir / 'alpha'}")
    print(f"  RGBA 抠图: {output_dir / 'rgba'}")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    main()
