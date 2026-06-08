#!/usr/bin/env python3
"""双击运行：调用 BiRefNet 对 input/ 中所有图片进行抠图，输出到 output/。"""

import os
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    result = subprocess.run(
        [sys.executable, "birefnet_matting.py"],
        capture_output=False,
        text=True,
    )
except Exception as e:
    print(f"错误: {e}")

input("\n按回车键关闭窗口...")
