"""ColorMatchTool — match input images to a reference color grade.

Two modes (auto-selected):
  LUT mode:  reference and input have same-dimension images → extract per-channel
             curves from pixel pairs. Best when source/ref are the same content.
  LAB mode:  different content → match color statistics (mean + std) in LAB space.

Reads from reference/ + input/, writes to output/.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image


# ── LAB color transfer ──────────────────────────────────────────────

def rgb_to_lab(img_np):
    """RGB (0-255) → LAB. img_np: HxWx3 float32."""
    img = img_np / 255.0
    mask = img > 0.04045
    img[mask] = ((img[mask] + 0.055) / 1.055) ** 2.4
    img[~mask] /= 12.92

    r, g, b_ = img[:,:,0], img[:,:,1], img[:,:,2]
    x = r * 0.4124564 + g * 0.3575761 + b_ * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b_ * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b_ * 0.9503041

    xn, yn, zn = 0.95047, 1.0, 1.08883
    delta = 6.0 / 29.0
    def f(t):
        d3 = delta ** 3
        out = np.where(t > d3, t ** (1.0/3.0), t / (3.0 * delta**2) + 4.0/29.0)
        return out

    L = 116.0 * f(y / yn) - 16.0
    a_ = 500.0 * (f(x / xn) - f(y / yn))
    b_lab = 200.0 * (f(y / yn) - f(z / zn))
    return np.dstack([L, a_, b_lab])


def lab_to_rgb(lab):
    """LAB → RGB (0-255). lab: HxWx3 float32."""
    L, a_, b_lab = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    delta = 6.0 / 29.0
    fy = (L + 16.0) / 116.0
    fx = a_ / 500.0 + fy
    fz = fy - b_lab / 200.0

    d3 = delta ** 3
    def finv(t):
        return np.where(t > delta, t ** 3, 3.0 * delta**2 * (t - 4.0/29.0))

    xn, yn, zn = 0.95047, 1.0, 1.08883
    x = finv(fx) * xn
    y = finv(fy) * yn
    z = finv(fz) * zn

    r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b_ = x *  0.0556434 + y * -0.2040259 + z *  1.0572252
    rgb = np.dstack([r, g, b_])

    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1.0/2.4)) - 0.055
    rgb[~mask] *= 12.92
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def lab_transfer(src, ref):
    """Match src (HxWx3 uint8) to ref (HxWx3 uint8) color stats in LAB."""
    src_lab = rgb_to_lab(src.astype(np.float32))
    ref_lab = rgb_to_lab(ref.astype(np.float32))
    for c in range(3):
        sm, ss = src_lab[:,:,c].mean(), src_lab[:,:,c].std()
        rm, rs = ref_lab[:,:,c].mean(), ref_lab[:,:,c].std()
        src_lab[:,:,c] = (src_lab[:,:,c] - sm) * (rs / max(ss, 1e-6)) + rm
    return lab_to_rgb(src_lab)


# ── LUT from pixel pairs ────────────────────────────────────────────

def build_lut_from_pair(src_np, ref_np, alpha_mask=None):
    """Build per-channel 256-entry LUT from two same-size images."""
    if alpha_mask is None:
        alpha_mask = np.ones(src_np.shape[:2], dtype=bool)
    lut = {}
    for c_idx, name in enumerate(["R", "G", "B"]):
        l = np.zeros(256, dtype=np.float32)
        for v in range(256):
            matched = ref_np[:,:,c_idx][alpha_mask & (src_np[:,:,c_idx] == v)]
            l[v] = float(np.median(matched)) if len(matched) > 0 else float(v)
        # Smooth
        from scipy.ndimage import uniform_filter1d
        l = uniform_filter1d(l, size=5)
        lut[name] = np.clip(l, 0, 255).astype(np.uint8)
    return lut


def apply_lut(img_np, lut):
    """Apply per-channel LUT to an HxWx3 uint8 image."""
    out = img_np.copy()
    out[:,:,0] = lut["R"][img_np[:,:,0]]
    out[:,:,1] = lut["G"][img_np[:,:,1]]
    out[:,:,2] = lut["B"][img_np[:,:,2]]
    return out


# ── Main ────────────────────────────────────────────────────────────

def main():
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    ref_dir = script_dir / "reference"
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"

    for d, name in [(ref_dir, "reference"), (input_dir, "input")]:
        if not d.is_dir():
            print(f"错误: 找不到 {name}/ 文件夹")
            return

    # Find reference images
    ref_files = sorted(ref_dir.rglob("*.png"))
    if not ref_files:
        print("错误: reference/ 中没有 .png 文件")
        return

    # Find input images (recursive)
    input_files = sorted(input_dir.rglob("*.png"))
    if not input_files:
        print("错误: input/ 中没有 .png 文件")
        return

    # ── Display summary ──
    print("=" * 55)
    print(f"  reference/: {len(ref_files)} 张")
    for rf in ref_files:
        ri = Image.open(rf)
        print(f"    {rf.name}  ({ri.size[0]}x{ri.size[1]})")
    print(f"  input/:     {len(input_files)} 张")
    folders = set(f.relative_to(input_dir).parent for f in input_files)
    for folder in sorted(folders):
        fn = str(folder) if str(folder) != "." else "(根目录)"
        cnt = sum(1 for f in input_files if f.relative_to(input_dir).parent == folder)
        print(f"    {fn}/  ({cnt} 张)")
    print("=" * 55)

    # ── Determine mode ──
    # LUT mode: reference has one image, find same-dim input
    lut = None
    ref_img_path = ref_files[0]  # Use first reference image
    ref_img = Image.open(ref_img_path).convert("RGB")
    ref_np = np.array(ref_img, dtype=np.uint8)

    pair_src = None
    for inp in input_files:
        img = Image.open(inp)
        if img.size == ref_img.size:
            pair_src = inp
            break

    if pair_src:
        print(f"\n>> LUT 模式")
        print(f"   参考: {ref_img_path.name}")
        print(f"   配对源: {pair_src.relative_to(input_dir)}  (同尺寸 {ref_img.size[0]}x{ref_img.size[1]})")
        pair_img = Image.open(pair_src).convert("RGB")
        pair_np = np.array(pair_img, dtype=np.uint8)
        lut = build_lut_from_pair(pair_np, ref_np)
        # Validate
        result = apply_lut(pair_np, lut)
        err = np.mean(np.abs(result.astype(float) - ref_np.astype(float)))
        print(f"   LUT 拟合误差: {err:.1f} (out of 255)")
    else:
        print(f"\n>> LAB 色彩迁移模式 (无同尺寸配对图)")
        print(f"   参考: {ref_img_path.name} ({ref_img.size[0]}x{ref_img.size[1]})")

    # ── Confirm ──
    confirm = input("\n确认开始处理？(y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("已取消。")
        return

    # ── Process ──
    done = 0
    for f in input_files:
        img = Image.open(f).convert("RGB")
        src_np = np.array(img, dtype=np.uint8)

        if lut is not None:
            # Apply LUT directly (works regardless of size)
            result_np = apply_lut(src_np, lut)
        else:
            result_np = lab_transfer(src_np, ref_np)

        out_path = output_dir / f.relative_to(input_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_np).save(out_path)

        rel = f.relative_to(input_dir)
        print(f"  完成 {rel}")

        # Save alpha if present
        img_rgba = Image.open(f)
        if img_rgba.mode == "RGBA":
            result_rgba = img_rgba.copy()
            result_rgba.putalpha(img_rgba.split()[3])
            result_rgba_np = np.array(result_rgba, dtype=np.uint8)
            result_rgba_np[:,:,:3] = result_np
            Image.fromarray(result_rgba_np).save(out_path)

        done += 1

    print(f"\n处理完毕: {done} 张")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
