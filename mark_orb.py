"""
mark_orb.py - 标记血球位置工具
鼠标操作：
  1. 左键点击圆心
  2. 拖拽到边缘确定半径
  3. 按 Enter 保存，按 r 重画，按 q 退出
"""

import cv2
import numpy as np
from screen_capture import ScreenCapture, list_and_select_window
from config import CAPTURE_WINDOW_TITLE

state = {
    "center": None,
    "radius": 0,
    "drawing": False,
    "done": False,
}


def mouse_cb(event, x, y, flags, param):
    frame = param["frame"]
    scale = param["scale"]

    # 转换到原始坐标
    ox = int(x / scale)
    oy = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        state["center"] = (ox, oy)
        state["radius"] = 0
        state["drawing"] = True
        state["done"] = False

    elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
        if state["center"]:
            cx, cy = state["center"]
            state["radius"] = int(((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5)

    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"] = False
        state["done"] = True


def main():
    print("=" * 50)
    print("  血球位置标记工具")
    print("=" * 50)

    game_window = list_and_select_window(title_hint=CAPTURE_WINDOW_TITLE)
    if game_window is None:
        print("未选择窗口")
        return

    capture = ScreenCapture(window_obj=game_window)
    frame = capture.grab()
    if frame is None:
        print("截屏失败")
        return

    original = frame.copy()
    fh, fw = frame.shape[:2]

    # 缩放显示
    scale = min(1280 / fw, 720 / fh, 1.0)
    dw, dh = int(fw * scale), int(fh * scale)

    win_name = "Mark Orb - Click center, drag to edge"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    param = {"frame": original, "scale": scale}
    cv2.setMouseCallback(win_name, mouse_cb, param)

    print("\n操作说明:")
    print("  1. 左键点击血球圆心")
    print("  2. 拖拽到血球边缘确定半径")
    print("  3. Enter = 保存, r = 重画, q = 退出\n")

    while True:
        display = original.copy()

        if state["center"]:
            cx, cy = state["center"]
            r = state["radius"]

            # 画圆
            cv2.circle(display, (cx, cy), max(r, 1), (255, 255, 0), 2)
            # 中线
            cv2.line(display, (cx, cy - r), (cx, cy + r), (255, 255, 0), 1)
            # 圆心
            cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)

            # HP / MP 标签
            cv2.putText(display, "HP", (cx - r - 5, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(display, "MP", (cx + 5, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

            # 信息
            info = f"Center=({cx},{cy}) Radius={r}"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 缩放显示
        show = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_AREA)
        cv2.imshow(win_name, show)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state["center"] = None
            state["radius"] = 0
            state["done"] = False
        elif key == 13 and state["center"] and state["radius"] > 5:  # Enter
            cx, cy = state["center"]
            r = state["radius"]
            print(f"\n保存血球位置:")
            print(f"  POTION_ORB_CENTER_X = {cx}")
            print(f"  POTION_ORB_CENTER_Y = {cy}")
            print(f"  POTION_ORB_RADIUS = {r}")

            # 写入 config
            try:
                with open("config.py", "r", encoding="utf-8") as f:
                    content = f.read()

                import re
                content = re.sub(r'POTION_ORB_CENTER_X\s*=\s*\d+',
                                 f'POTION_ORB_CENTER_X = {cx}', content)
                content = re.sub(r'POTION_ORB_CENTER_Y\s*=\s*\d+',
                                 f'POTION_ORB_CENTER_Y = {cy}', content)
                content = re.sub(r'POTION_ORB_RADIUS\s*=\s*\d+',
                                 f'POTION_ORB_RADIUS = {r}', content)

                with open("config.py", "w", encoding="utf-8") as f:
                    f.write(content)

                print("  ✓ 已写入 config.py")
            except Exception as e:
                print(f"  写入失败: {e}")
                print(f"  请手动修改 config.py")
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
