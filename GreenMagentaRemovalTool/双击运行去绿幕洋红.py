#!/usr/bin/env python3
"""双击运行：去除序列帧中的绿色和洋红色像素（转为透明）。"""

import os
import subprocess
import sys

# 获取脚本所在目录，确保无论从哪双击都能找到核心脚本
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    result = subprocess.run(
        [sys.executable, "auto_remove_green_magenta.py"],
        capture_output=False,
        text=True
    )
except Exception as e:
    print(f"错误: {e}")

input("\n按回车键关闭窗口...")
