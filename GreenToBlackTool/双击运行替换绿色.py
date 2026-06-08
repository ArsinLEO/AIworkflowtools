#!/usr/bin/env python3
"""双击运行：将 PNG 中的绿色像素替换为纯黑色。"""

import os
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    result = subprocess.run(
        [sys.executable, "green_to_black.py", "--auto"],
        capture_output=False,
        text=True
    )
except Exception as e:
    print(f"错误: {e}")

input("\n按回车键关闭窗口...")
