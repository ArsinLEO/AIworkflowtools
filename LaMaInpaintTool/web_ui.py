#!/usr/bin/env python3
"""
LaMa AI 图像修复 — 网页画笔界面。

双击运行，浏览器自动打开。在图上涂抹标记要擦除的区域，点"修复"即可。
支持拖拽图片、缩放画笔、撤销操作。
"""

import os
import sys
import numpy as np
from PIL import Image
import gradio as gr

# 确保脚本所在目录为工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 设置 HF 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_INPUT_SIZE = 512
MODEL_REPO = "Carve/LaMa-ONNX"
MODEL_NAME = "lama_fp32.onnx"
WEIGHTS_DIR = "weights"

# ============ 模型加载 ============

_session = None
_input_name = None
_mask_name = None
_output_name = None


def get_session():
    """懒加载 ONNX 会话。"""
    global _session, _input_name, _mask_name, _output_name
    if _session is not None:
        return _session, _input_name, _mask_name, _output_name

    import onnxruntime as ort

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_path = os.path.join(WEIGHTS_DIR, MODEL_NAME)

    if not os.path.exists(model_path):
        print(f"[模型] 下载 {MODEL_NAME} (~170MB)...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(MODEL_REPO, MODEL_NAME, local_dir=WEIGHTS_DIR)
        print("[模型] 下载完成")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _input_name = session.get_inputs()[0].name
    _mask_name = session.get_inputs()[1].name
    _output_name = session.get_outputs()[0].name
    _session = session
    return session, _input_name, _mask_name, _output_name


def inpaint_brush(original, mask_layer):
    """
    Gradio 回调：接收原图和画笔图层，执行 LaMa 修复。

    参数:
        original: PIL Image (RGB) — 用户上传/拖入的原图
        mask_layer: PIL Image (RGBA) — 用户在图上绘制的画笔图层
                    （画笔笔迹不透明 = 要擦除的区域）
    返回:
        PIL Image (RGB) — 修复后的图片
    """
    if original is None:
        return None

    original = original.convert("RGB")
    orig_size = original.size

    # 从画笔图层提取遮罩 (alpha 通道的不透明度 = 擦除强度)
    if mask_layer is None:
        # 没有画任何东西，返回原图
        return original

    mask_layer = mask_layer.convert("RGBA")
    mask_arr = np.array(mask_layer)

    # 用 alpha 通道作为遮罩 (非透明 = 画笔涂过 = 要擦除)
    mask = Image.fromarray(mask_arr[:, :, 3], mode="L")

    # 如果遮罩全黑（没涂任何东西），返回原图
    if np.array(mask).max() < 5:
        return original

    # 确保尺寸一致
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.LANCZOS)

    # 缩放到模型输入尺寸
    img_512 = original.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BICUBIC)
    mask_512 = mask.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)

    # 准备 ONNX 输入
    img_np = np.array(img_512, dtype=np.float32) / 255.0
    mask_np = np.array(mask_512, dtype=np.float32) / 255.0

    img_tensor = img_np.transpose(2, 0, 1)[np.newaxis, ...]
    mask_tensor = mask_np[np.newaxis, np.newaxis, ...]

    # 推理
    session, in_name, mask_name, out_name = get_session()
    result = session.run([out_name], {in_name: img_tensor, mask_name: mask_tensor})[0]

    # 转回 PIL
    result = result[0].transpose(1, 2, 0)
    result = np.clip(result, 0, 1)
    result_img = Image.fromarray((result * 255).astype(np.uint8))

    # 还原尺寸
    result_img = result_img.resize(orig_size, Image.LANCZOS)

    return result_img


# ============ Gradio UI ============

def create_ui():
    """构建画笔界面。"""
    with gr.Blocks(title="LaMa AI 图像修复") as demo:
        gr.Markdown("""
        # LaMa AI 图像修复
        **拖入图片 → 画笔涂抹要擦除的区域 → 点"修复"**
        适合：去水印、擦杂物、修穿帮、清文字
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.ImageEditor(
                    label="拖入图片，用画笔涂抹要擦除的区域",
                    type="pil",
                    brush=gr.Brush(
                        default_size=20,
                        colors=["rgba(255,255,255,1)"],
                        color_mode="fixed",
                    ),
                    interactive=True,
                    height=500,
                )
                with gr.Row():
                    inpaint_btn = gr.Button("修复", variant="primary", size="lg")
                    clear_btn = gr.Button("清除全部", size="sm")

            with gr.Column(scale=1):
                output_img = gr.Image(
                    label="修复结果",
                    type="pil",
                    interactive=False,
                    height=500,
                )

        gr.Markdown("""
        ### 使用技巧
        - **画笔大小**：右下角滑块调整
        - **撤销**：Ctrl+Z
        - **多物体**：一笔一笔涂，涂好后点一次"修复"
        - **效果不佳**：涂得比物体稍大一圈效果更好
        - **大图**：会自动压缩处理，不影响最终分辨率
        """, elem_classes=["brush-hint"])

        inpaint_btn.click(
            fn=inpaint_brush,
            inputs=[input_img, input_img],
            outputs=[output_img],
        )

        clear_btn.click(
            fn=lambda: (None, None),
            outputs=[input_img, output_img],
        )

    return demo


def main():
    # 预加载模型
    print("正在加载 LaMa 模型...")
    get_session()
    print("模型就绪！")

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7880,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
        .brush-hint {
            font-size: 14px;
            color: #666;
            margin-top: 8px;
        }
        """,
    )


if __name__ == "__main__":
    main()
