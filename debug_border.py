"""
debug_border.py - 调试黑色边框检测
截一帧游戏画面，显示黑色边框检测的中间结果
"""

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import cv2
import numpy as np
from screen_capture import ScreenCapture, list_and_select_window
from config import (
    EXCLUDE_LEFT_RATIO, EXCLUDE_BOTTOM_RATIO,
    EXCLUDE_MINIMAP_RIGHT_RATIO, EXCLUDE_MINIMAP_TOP_RATIO,
    EXCLUDE_TOP_RATIO, SELF_CENTER_X, SELF_CENTER_Y,
)


def main():
    game_window = list_and_select_window()
    if game_window is None:
        return

    capture = ScreenCapture(window_obj=game_window)

    print("按 空格 截取一帧分析，q 退出")

    while True:
        frame = capture.grab()
        if frame is None:
            continue

        cv2.imshow("Game", cv2.resize(frame, (960, 540)))
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        if key != ord(" "):
            continue

        print("\n分析中...")
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 角色周围 300px 区域
        cx, cy = SELF_CENTER_X, SELF_CENTER_Y
        r = 300
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)

        roi_gray = gray[y1:y2, x1:x2]
        roi_color = frame[y1:y2, x1:x2].copy()

        # 方法1: 亮度阈值（当前方法）
        _, dark_mask = cv2.threshold(roi_gray, 25, 255, cv2.THRESH_BINARY_INV)

        # 方法2: Canny 边缘
        edges = cv2.Canny(roi_gray, 50, 150)

        # 方法3: 自适应阈值
        adaptive = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 5)

        # 在每种方法上找矩形
        for name, mask in [("dark_thresh", dark_mask), ("canny", edges), ("adaptive", adaptive)]:
            # 闭运算连接
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0
            vis = roi_color.copy()
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # 血条边框大小范围
                if 12 <= bw <= 60 and 2 <= bh <= 8:
                    aspect = bw / bh
                    if aspect >= 2.5:
                        cnt_area = cv2.contourArea(cnt)
                        rect = cnt_area / (bw * bh) if bw * bh > 0 else 0
                        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
                        cv2.putText(vis, f"{bw}x{bh} r={rect:.2f}", (bx, by - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                        count += 1

            print(f"  {name}: {count} 个候选框")
            cv2.imshow(f"{name}_mask", cv2.resize(processed, (600, 600)))
            cv2.imshow(f"{name}_result", cv2.resize(vis, (600, 600)))

        # 还显示原始灰度
        cv2.imshow("roi_gray", cv2.resize(roi_gray, (600, 600)))

        print("  按空格继续截取，q 退出")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
