#!/usr/bin/env python3
"""双击运行：将序列帧缩放至宽度1920并纵向合并。"""

import os
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    result = subprocess.run(
        [sys.executable, "resize_to_1920_column.py"],
        capture_output=False,
        text=True
    )
except Exception as e:
    print(f"错误: {e}")

input("\n按回车键关闭窗口...")
