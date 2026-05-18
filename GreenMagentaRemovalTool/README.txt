绿幕/洋红幕去除工具
====================

功能：批量移除PNG序列帧中的绿色或洋红色背景像素，转为透明。

适用场景：
  - 去除绿幕背景（chroma key green）
  - 去除洋红幕背景（chroma key magenta，常用于游戏精灵图集）
  - 清除角色边缘的绿/洋红溢出（spill removal）

三个独立脚本：
  auto_remove_green.py          — 仅去除绿色背景
  auto_remove_magenta.py        — 仅去除洋红色背景
  auto_remove_green_magenta.py  — 同时去除绿色和洋红色背景（原始版本）

双击运行快捷方式：
  双击运行去绿幕.py        → 调用 auto_remove_green.py
  双击运行去洋红幕.py      → 调用 auto_remove_magenta.py
  双击运行去绿幕洋红.py    → 调用 auto_remove_green_magenta.py

使用方法：
  1. 将PNG序列帧放入 input/ 文件夹（支持子文件夹）
  2. 双击对应的 "双击运行去X幕.py"
  3. 处理结果输出到 output/ 文件夹

检测原理：
  greenness = G - max(R, B)  纯绿(0,255,0)=255，自然物体通常<20
  magentaness = min(R, B) - G  纯洋红(255,0,255)=255

  阶段1 — 高置信度：greenness/magentaness >= HIGH 直接移除
  阶段2 — 边缘溢出：从"深背景"（距角色>DISTANCE_THRESHOLD px的透明区）泛洪，
           清除 greenness/magentaness >= LOW 的邻近像素。
           角色内部透明空洞距角色近，不会成为泛洪种子。

各脚本参数调整（编辑对应 .py 文件开头）：
  HIGH = 40                 - 阶段1阈值（高置信度直接移除）
  LOW = 25                  - 阶段2阈值（边缘溢出，越高越保护角色内部）
  DISTANCE_THRESHOLD = 15   - 深背景判定距离（距角色>此值为深背景）
  SPILL_ITERATIONS = 12     - 泛洪轮数（越大清理越深但可能误伤）

调参指南：
  - 右下角绿色没清干净 → 降低 LOW（如20）
  - 角色绿色衣服被误删 → 提高 LOW（如30）
  - 边缘绿边残留太多 → 增大 SPILL_ITERATIONS（如15）或降低 DISTANCE_THRESHOLD
  - 只想清除背景不想动边缘 → 减小 SPILL_ITERATIONS（如5）
  - 只要精确匹配纯绿(0,255,0) → HIGH=250, LOW=250
