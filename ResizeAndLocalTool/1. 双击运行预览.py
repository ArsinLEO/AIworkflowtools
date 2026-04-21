#!/usr/bin/env python3
"""双击运行第一阶段：生成预览图和参数。"""

import os
import subprocess
import sys

# 获取脚本所在目录，确保无论从哪双击都能找到核心脚本
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    result = subprocess.run(
        [sys.executable, "auto_align_frames.py"],
        capture_output=False,
        text=True
    )
    print("\n" + "="*60)
    print("预览已生成！请检查 output/_preview/ 文件夹。")
    print("确认无误后，双击【2. 双击执行全部.py】")
    print("="*60)
except Exception as e:
    print(f"错误: {e}")

input("\n按回车键关闭窗口...")
