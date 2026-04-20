#!/usr/bin/env python3
"""双击运行第一阶段：生成预览图和参数。"""

import subprocess
import sys

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
