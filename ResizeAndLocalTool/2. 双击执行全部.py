#!/usr/bin/env python3
"""双击运行第二阶段：应用对齐到全部帧并输出最终结果。"""

import subprocess
import sys

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
