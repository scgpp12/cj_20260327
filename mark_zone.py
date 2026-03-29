"""
mark_zone.py - 标记检测区域工具
框选战斗检测范围，自动写入 config.py

操作:
    1. 选择游戏窗口
    2. 鼠标拖动框选检测区域
    3. 按 s 保存
    4. 按 r 重新截屏
    5. 按 q 退出
"""

import re
import cv2
from screen_capture import ScreenCapture, list_and_select_window

drawing = False
start_x, start_y = 0, 0
end_x, end_y = 0, 0
selection_done = False


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, selection_done
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        selection_done = False
        start_x, start_y = x, y
        end_x, end_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_x, end_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        selection_done = True


def main():
    global start_x, start_y, end_x, end_y, selection_done

    print("=" * 60)
    print("  标记检测区域（战斗范围）")
    print("  拖动框选区域 -> 按 s 保存 -> 按 q 退出")
    print("=" * 60)

    window_title = list_and_select_window()
    if window_title is None:
        return

    capture = ScreenCapture(window_obj=window_title)
    frame = capture.grab()
    if frame is None:
        print("[ERROR] 截屏失败")
        capture.release()
        return

    fh, fw = frame.shape[:2]
    scale = min(1280 / fw, 720 / fh, 1.0)
    dw, dh = int(fw * scale), int(fh * scale)
    display_base = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)

    # 读取已有的 SELF 区域并显示
    try:
        from config import (EXCLUDE_CENTER_ENABLED, EXCLUDE_CENTER_X1,
                            EXCLUDE_CENTER_Y1, EXCLUDE_CENTER_X2, EXCLUDE_CENTER_Y2)
        has_self = EXCLUDE_CENTER_ENABLED
    except ImportError:
        has_self = False

    win_name = "Mark Zone"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    while True:
        display = display_base.copy()

        # 显示已有的 SELF 排除区域（缩放后坐标）
        if has_self:
            sx1 = int(EXCLUDE_CENTER_X1 * scale)
            sy1 = int(EXCLUDE_CENTER_Y1 * scale)
            sx2 = int(EXCLUDE_CENTER_X2 * scale)
            sy2 = int(EXCLUDE_CENTER_Y2 * scale)
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
            cv2.putText(display, "SELF", (sx1, sy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 画选区
        if drawing or selection_done:
            cv2.rectangle(display, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            ox1 = int(min(start_x, end_x) / scale)
            oy1 = int(min(start_y, end_y) / scale)
            ox2 = int(max(start_x, end_x) / scale)
            oy2 = int(max(start_y, end_y) / scale)

            info = f"Zone: ({ox1},{oy1})-({ox2},{oy2}) = {ox2-ox1}x{oy2-oy1}px"
            cv2.putText(display, info, (10, dh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display, info, (10, dh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display, "Drag = detection zone (green) | s=save | r=recapture | q=quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(display, "Drag = detection zone (green) | s=save | r=recapture | q=quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.imshow(win_name, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            new_frame = capture.grab()
            if new_frame is not None:
                frame = new_frame
                fh, fw = frame.shape[:2]
                scale = min(1280 / fw, 720 / fh, 1.0)
                dw, dh = int(fw * scale), int(fh * scale)
                display_base = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
                selection_done = False
                print("[INFO] 已重新截屏")
        elif key == ord("s") and selection_done:
            ox1 = int(min(start_x, end_x) / scale)
            oy1 = int(min(start_y, end_y) / scale)
            ox2 = int(max(start_x, end_x) / scale)
            oy2 = int(max(start_y, end_y) / scale)

            if ox2 - ox1 < 20 or oy2 - oy1 < 20:
                print("[WARNING] 选区太小，请重新框选")
                continue

            print(f"\n[OK] 检测区域: ({ox1},{oy1}) - ({ox2},{oy2})")
            print(f"     尺寸: {ox2-ox1}x{oy2-oy1}px")

            config_path = "config.py"
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "DETECT_ZONE_X1" in content:
                content = re.sub(r'DETECT_ZONE_X1 = \d+', f'DETECT_ZONE_X1 = {ox1}', content)
                content = re.sub(r'DETECT_ZONE_Y1 = \d+', f'DETECT_ZONE_Y1 = {oy1}', content)
                content = re.sub(r'DETECT_ZONE_X2 = \d+', f'DETECT_ZONE_X2 = {ox2}', content)
                content = re.sub(r'DETECT_ZONE_Y2 = \d+', f'DETECT_ZONE_Y2 = {oy2}', content)
            else:
                zone_block = (
                    f"\n# ============================================================\n"
                    f"# 检测区域（战斗范围），仅在此区域内检测运动和血条\n"
                    f"# ============================================================\n"
                    f"DETECT_ZONE_ENABLED = True\n"
                    f"DETECT_ZONE_X1 = {ox1}\n"
                    f"DETECT_ZONE_Y1 = {oy1}\n"
                    f"DETECT_ZONE_X2 = {ox2}\n"
                    f"DETECT_ZONE_Y2 = {oy2}\n"
                )
                # 插入到运动检测开关之前
                content = content.replace(
                    "# ============================================================\n# 运动检测开关",
                    zone_block + "# ============================================================\n# 运动检测开关"
                )

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"[OK] 已保存到 {config_path}")
            print("[INFO] 重新运行 main.py 即可生效\n")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
