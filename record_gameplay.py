"""
record_gameplay.py - 玩家操作录制工具
录制鼠标点击、键盘按键、游戏截屏，用于分析攻击模式。

使用方式：
  python record_gameplay.py
  正常玩游戏打怪，录制结束后按 Ctrl+C

输出：
  datasets/recordings/session_XXXXXXXX/
    ├─ events.jsonl     # 鼠标/键盘事件（带时间戳）
    ├─ frames/          # 截屏（每 0.5 秒一张）
    └─ summary.json     # 录制摘要
"""

import os
import sys
import json
import time
import threading
import cv2
import numpy as np
from datetime import datetime

# 截屏
from screen_capture import ScreenCapture, list_and_select_window
from config import CAPTURE_WINDOW_TITLE

# 键鼠监听
try:
    from pynput import mouse, keyboard
except ImportError:
    print("需要安装 pynput: pip install pynput")
    sys.exit(1)


class GameplayRecorder:
    """录制游戏操作"""

    def __init__(self, capture, save_dir, frame_interval=0.5):
        self.capture = capture
        self.save_dir = save_dir
        self.frame_interval = frame_interval

        # 事件记录
        self.events = []
        self.start_time = time.time()
        self.frame_count = 0
        self.running = False

        # 创建目录
        self.frames_dir = os.path.join(save_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        # 事件文件
        self.event_file = open(os.path.join(save_dir, "events.jsonl"), "w", encoding="utf-8")

    def _timestamp(self):
        return round(time.time() - self.start_time, 3)

    def _log_event(self, event_type, data):
        event = {
            "t": self._timestamp(),
            "type": event_type,
            **data,
        }
        self.events.append(event)
        self.event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.event_file.flush()

    # ---- 鼠标回调 ----
    def on_mouse_click(self, x, y, button, pressed):
        if not self.running:
            return
        self._log_event("mouse_click", {
            "x": x, "y": y,
            "button": str(button),
            "pressed": pressed,
        })
        action = "按下" if pressed else "松开"
        print(f"  [{self._timestamp():.1f}s] 鼠标{action} {button} ({x},{y})")

    def on_mouse_move(self, x, y):
        # 鼠标移动太频繁，每 100ms 记一次
        if not self.running:
            return
        t = self._timestamp()
        if not hasattr(self, '_last_move_t') or t - self._last_move_t >= 0.1:
            self._log_event("mouse_move", {"x": x, "y": y})
            self._last_move_t = t

    # ---- 键盘回调 ----
    def on_key_press(self, key):
        if not self.running:
            return
        try:
            key_name = key.char if hasattr(key, 'char') and key.char else str(key)
        except Exception:
            key_name = str(key)
        self._log_event("key_down", {"key": key_name})
        print(f"  [{self._timestamp():.1f}s] 键盘按下 {key_name}")

    def on_key_release(self, key):
        if not self.running:
            return
        try:
            key_name = key.char if hasattr(key, 'char') and key.char else str(key)
        except Exception:
            key_name = str(key)
        self._log_event("key_up", {"key": key_name})

    # ---- 截屏线程 ----
    def _frame_loop(self):
        while self.running:
            frame = self.capture.grab()
            if frame is not None:
                t = self._timestamp()
                fname = f"frame_{self.frame_count:06d}.jpg"
                path = os.path.join(self.frames_dir, fname)
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                self._log_event("frame", {
                    "file": fname,
                    "frame_id": self.frame_count,
                })
                self.frame_count += 1

            time.sleep(self.frame_interval)

    def start(self):
        """开始录制"""
        self.running = True
        self.start_time = time.time()

        # 启动键鼠监听（后台线程）
        self.mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click,
            on_move=self.on_mouse_move,
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release,
        )
        self.mouse_listener.start()
        self.keyboard_listener.start()

        print(f"录制开始！截屏间隔: {self.frame_interval}s")
        print("正常玩游戏打怪，按 Ctrl+C 结束录制\n")

    def capture_frame(self):
        """主线程调用：截屏一帧（mss 不支持子线程）"""
        frame = self.capture.grab()
        if frame is not None:
            fname = f"frame_{self.frame_count:06d}.jpg"
            path = os.path.join(self.frames_dir, fname)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self._log_event("frame", {"file": fname, "frame_id": self.frame_count})
            self.frame_count += 1

    def stop(self):
        """停止录制，保存摘要"""
        self.running = False
        time.sleep(0.5)

        try:
            self.mouse_listener.stop()
            self.keyboard_listener.stop()
        except Exception:
            pass

        self.event_file.close()

        # 统计
        duration = self._timestamp()
        clicks = sum(1 for e in self.events if e["type"] == "mouse_click" and e.get("pressed"))
        keys = sum(1 for e in self.events if e["type"] == "key_down")

        summary = {
            "duration_sec": round(duration, 1),
            "total_events": len(self.events),
            "mouse_clicks": clicks,
            "key_presses": keys,
            "frames": self.frame_count,
            "frame_interval": self.frame_interval,
            "timestamp": datetime.now().isoformat(),
        }

        with open(os.path.join(self.save_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*50}")
        print(f"  录制完成!")
        print(f"  时长: {duration:.1f} 秒")
        print(f"  鼠标点击: {clicks} 次")
        print(f"  键盘按键: {keys} 次")
        print(f"  截屏: {self.frame_count} 张")
        print(f"  保存目录: {self.save_dir}")
        print(f"{'='*50}")


def main():
    print("=" * 50)
    print("  玩家操作录制工具")
    print("=" * 50)

    game_window = list_and_select_window(title_hint=CAPTURE_WINDOW_TITLE)
    if game_window is None:
        print("未选择窗口")
        return

    capture = ScreenCapture(window_obj=game_window)

    # 创建录制目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("datasets", "recordings", f"session_{ts}")

    recorder = GameplayRecorder(capture, save_dir, frame_interval=0.5)

    try:
        recorder.start()
        # 主线程循环截屏（mss 必须在主线程）
        last_frame_time = 0
        while True:
            now = time.time()
            if now - last_frame_time >= recorder.frame_interval:
                recorder.capture_frame()
                last_frame_time = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n停止录制...")
        recorder.stop()
    finally:
        capture.release()


if __name__ == "__main__":
    main()
