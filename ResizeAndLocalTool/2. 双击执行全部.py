#!/usr/bin/env python3
"""双击运行第二阶段：应用对齐到全部帧并输出最终结果。"""

import os
import subprocess
import sys

# 获取脚本所在目录，确保无论从哪双击都能找到核心脚本
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    result = subprocess.run(
        [sys.executable, "auto_align_frames.py", "--apply"],
        capture_output=False,
        text=True
    )
    print("\n" + "="*60)
    print("全部完成！结果在 output/ 文件夹中。")
    print("="*60)
except Exception as e:
    print(f"错误: {e}")

input("\n按回车键关闭窗口...")
