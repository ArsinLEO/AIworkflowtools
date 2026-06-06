import os
import sys
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
subprocess.run([sys.executable, "black_white_matting.py"])
input("\n按回车键关闭窗口...")
