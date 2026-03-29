"""
auto_screenshot.py - 游戏实时定时截屏工具
用于采集 YOLO 训练数据

功能：
  - 每 N 秒自动截取游戏窗口
  - 保存到 datasets/images/ 目录
  - 文件名包含时间戳，方便排序
  - 按 q 退出，按 p 暂停/继续
  - 显示实时预览 + 截屏倒计时
"""

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import os
import time
import cv2
from screen_capture import ScreenCapture, list_and_select_window
from config import CAPTURE_WINDOW_TITLE

# ---- 配置 ----
SCREENSHOT_INTERVAL = 1        # 截屏间隔（秒）
SAVE_DIR = "datasets/images"   # 保存目录
IMAGE_FORMAT = ".png"          # 图片格式 (.png 无损, .jpg 有损但小)
JPG_QUALITY = 95               # JPG 质量 (仅 .jpg 格式时生效)
MAX_IMAGES = 5000              # 最大截屏数量（防止磁盘爆满）


def main():
    print("=" * 60)
    print("  游戏截屏采集工具 (YOLO 训练数据)")
    print("=" * 60)
    print(f"  截屏间隔: {SCREENSHOT_INTERVAL} 秒")
    print(f"  保存目录: {SAVE_DIR}")
    print(f"  图片格式: {IMAGE_FORMAT}")
    print(f"  快捷键: q=退出  p=暂停/继续  s=立即截屏")
    print("=" * 60)

    # 选择游戏窗口
    game_window = list_and_select_window(title_hint=CAPTURE_WINDOW_TITLE)
    if game_window is None:
        print("[ERROR] 未选择目标，退出")
        return

    capture = ScreenCapture(window_obj=game_window)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 统计已有图片数量
    existing = len([f for f in os.listdir(SAVE_DIR) if f.endswith(IMAGE_FORMAT)])
    print(f"  已有图片: {existing}")
    print(f"  开始采集...\n")

    count = existing
    last_save_time = time.time()
    paused = False
    fps = 0.0
    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            frame = capture.grab()
            if frame is None:
                time.sleep(0.1)
                continue

            now = time.time()
            display = frame.copy()
            h, w = display.shape[:2]

            # FPS
            frame_count += 1
            if now - fps_start >= 1.0:
                fps = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now

            # 倒计时
            if not paused:
                remaining = max(0, SCREENSHOT_INTERVAL - (now - last_save_time))
            else:
                remaining = -1

            # ---- 自动截屏 ----
            saved_this_frame = False
            if not paused and (now - last_save_time) >= SCREENSHOT_INTERVAL:
                if count < MAX_IMAGES:
                    saved_this_frame = _save_screenshot(frame, count)
                    if saved_this_frame:
                        count += 1
                        last_save_time = now
                else:
                    print(f"[WARN] 已达到最大数量 {MAX_IMAGES}，停止采集")
                    paused = True

            # ---- 绘制预览 UI ----
            # 顶部状态栏背景
            cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)

            # FPS
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 截屏计数
            cv2.putText(display, f"Screenshots: {count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 状态
            if paused:
                status = "PAUSED"
                status_color = (0, 0, 255)
            else:
                status = f"Next in {remaining:.1f}s"
                status_color = (0, 255, 0)
            cv2.putText(display, status, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # 倒计时进度条
            if not paused:
                bar_x = 250
                bar_w = 300
                bar_y = 60
                bar_h = 15
                progress = 1.0 - (remaining / SCREENSHOT_INTERVAL)
                fill_w = int(bar_w * progress)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
                bar_color = (0, 255, 0) if progress < 0.8 else (0, 255, 255)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

            # 刚保存时闪烁提示
            if saved_this_frame:
                cv2.putText(display, "SAVED!", (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
                cv2.putText(display, "SAVED!", (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # 右上角快捷键提示
            cv2.putText(display, "q:quit p:pause s:snap", (w - 300, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # 无窗口模式，只打印进度
            time.sleep(0.01)

    except KeyboardInterrupt:
        print(f"\n[INFO] Ctrl+C，共采集 {count} 张截图")
    finally:
        capture.release()


def _save_screenshot(frame, index):
    """保存截屏到文件"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"game_{ts}_{index:05d}{IMAGE_FORMAT}"
    filepath = os.path.join(SAVE_DIR, filename)

    try:
        if IMAGE_FORMAT == ".jpg":
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
        else:
            cv2.imwrite(filepath, frame)
        print(f"  [{index:05d}] {filename}")
        return True
    except Exception as e:
        print(f"  [ERROR] 保存失败: {e}")
        return False


if __name__ == "__main__":
    main()
