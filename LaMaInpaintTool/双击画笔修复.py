#!/usr/bin/env python3
"""双击运行：打开网页画笔界面，在图上涂抹擦除区域，一键修复。"""

import os
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("正在启动 LaMa 画笔修复界面...")
print("浏览器将自动打开，如未打开请手动访问: http://127.0.0.1:7880")

try:
    result = subprocess.run(
        [sys.executable, "web_ui.py"],
        capture_output=False,
        text=True,
    )
except KeyboardInterrupt:
    print("\n已关闭。")
except Exception as e:
    print(f"错误: {e}")
    input("\n按回车键关闭窗口...")
