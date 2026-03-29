"""
screen_capture.py - 游戏窗口截取模块
支持两种模式：
  1. 选择显示器直接截取（推荐，稳定）
  2. 选择窗口截取（可能因编码问题不稳定）
"""

import numpy as np
import mss
import cv2

try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except ImportError:
    HAS_PYGETWINDOW = False


def list_and_select_window(title_hint=""):
    """
    列出所有显示器和可见窗口，让用户选择截取目标

    Returns:
        dict: {"mode": "monitor"|"window", "data": ...}
    """
    sct = mss.mss()
    monitors = sct.monitors  # [0]=虚拟桌面, [1]=主显示器, [2]=副显示器...

    print("\n" + "=" * 60)
    print("  请选择截取目标:")
    print("=" * 60)

    options = []

    # 列出显示器（从 1 开始，跳过 0 虚拟桌面）
    for i in range(1, len(monitors)):
        m = monitors[i]
        label = "主显示器" if i == 1 else f"副显示器 {i}"
        print(f"  [{len(options)}] [显示器] {label}  "
              f"{m['width']}x{m['height']}  pos=({m['left']},{m['top']})")
        options.append({"mode": "monitor", "data": m, "index": i})

    # 列出窗口
    if HAS_PYGETWINDOW:
        all_windows = gw.getAllWindows()
        visible = [w for w in all_windows
                   if w.title.strip() and w.visible and w.width > 100 and w.height > 100]
        for w in visible:
            try:
                title = w.title[:40]
            except Exception:
                title = "(编码错误)"
            print(f"  [{len(options)}] [窗口]   {title}  "
                  f"{w.width}x{w.height}  pos=({w.left},{w.top})")
            options.append({"mode": "window", "data": w})

    print("=" * 60)

    sct.close()

    while True:
        try:
            choice = input(f"输入编号 (0-{len(options)-1}): ").strip()
            idx = int(choice)
            if 0 <= idx < len(options):
                opt = options[idx]
                if opt["mode"] == "monitor":
                    m = opt["data"]
                    print(f"[INFO] 已选择显示器 {opt['index']}: "
                          f"{m['width']}x{m['height']} pos=({m['left']},{m['top']})\n")
                else:
                    w = opt["data"]
                    print(f"[INFO] 已选择窗口: {w.width}x{w.height} "
                          f"pos=({w.left},{w.top})\n")
                return opt
            else:
                print(f"  请输入 0 到 {len(options)-1} 之间的数字")
        except ValueError:
            print("  请输入数字")
        except (KeyboardInterrupt, EOFError):
            return None


class ScreenCapture:
    """屏幕截取器 - 支持显示器模式和窗口模式"""

    def __init__(self, window_obj=None):
        """
        Args:
            window_obj: list_and_select_window() 返回的 dict
        """
        self.sct = mss.mss()
        self.window_left = 0
        self.window_top = 0

        if window_obj is None:
            self.mode = None
            self.monitor_data = None
            self.window_data = None
        elif window_obj["mode"] == "monitor":
            self.mode = "monitor"
            self.monitor_data = window_obj["data"]
            self.window_left = self.monitor_data["left"]
            self.window_top = self.monitor_data["top"]
        else:
            self.mode = "window"
            self.window_data = window_obj["data"]

    def grab(self):
        """
        截取一帧图像

        Returns:
            numpy.ndarray: BGR 格式图像，失败返回 None
        """
        try:
            if self.mode == "monitor":
                monitor = self.monitor_data
            elif self.mode == "window":
                win = self.window_data
                try:
                    if win.isMinimized:
                        return None
                except Exception:
                    pass

                left = max(0, win.left)
                top = max(0, win.top)
                width = win.width - (left - win.left)
                height = win.height - (top - win.top)

                if width < 50 or height < 50:
                    return None

                self.window_left = left
                self.window_top = top
                monitor = {"left": left, "top": top, "width": width, "height": height}
            else:
                return None

            screenshot = self.sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame

        except Exception as e:
            print(f"[ERROR] 截屏失败: {e}")
            return None

    def release(self):
        """释放资源"""
        try:
            self.sct.close()
        except Exception:
            pass
