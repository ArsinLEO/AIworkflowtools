#!/usr/bin/env python3
"""双击运行即可去除 input/ 中PNG序列帧的白色背景。"""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "auto_remove_white.py"])
print("\n按回车键退出...")
input()
