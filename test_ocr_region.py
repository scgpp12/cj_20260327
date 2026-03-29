"""
OCR 区域标记工具：截取游戏画面，鼠标拖拽框选 OCR 区域，输出坐标
操作：
  - 鼠标左键拖拽画框
  - 按 R 重新选择
  - 按 Enter 确认并保存
  - 按 ESC 退出
"""
import cv2
import numpy as np
from screen_capture import ScreenCapture, list_and_select_window
from config import CAPTURE_WINDOW_TITLE

# 全局状态
drawing = False
start_x, start_y = 0, 0
end_x, end_y = 0, 0
frame_copy = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        end_x, end_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x, end_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y


def main():
    global frame_copy, start_x, start_y, end_x, end_y

    game_window = list_and_select_window(title_hint=CAPTURE_WINDOW_TITLE)
    if game_window is None:
        print("未找到游戏窗口")
        return
    cap = ScreenCapture(window_obj=game_window)
    frame = cap.grab()
    if frame is None:
        print("截图失败")
        return

    h, w = frame.shape[:2]
    print(f"画面尺寸: {w}x{h}")
    print("操作说明:")
    print("  鼠标左键拖拽 → 框选坐标区域")
    print("  R → 重新选择")
    print("  Enter → 确认保存")
    print("  ESC → 退出")

    original = frame.copy()
    frame_copy = frame.copy()

    cv2.namedWindow("Select OCR Region", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select OCR Region", min(w, 1400), min(h, 900))
    cv2.setMouseCallback("Select OCR Region", mouse_callback)

    while True:
        display = original.copy()

        # 画当前选框
        if start_x != end_x and start_y != end_y:
            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)
            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 显示坐标
            label = f"({x1},{y1})-({x2},{y2}) {x2-x1}x{y2-y1}px"
            cv2.putText(display, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示鼠标位置
        cv2.putText(display, "Drag to select, Enter=OK, R=Reset, ESC=Quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Select OCR Region", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            start_x, start_y = 0, 0
            end_x, end_y = 0, 0
        elif key == 13:  # Enter
            if start_x != end_x and start_y != end_y:
                x1 = min(start_x, end_x)
                y1 = min(start_y, end_y)
                x2 = max(start_x, end_x)
                y2 = max(start_y, end_y)

                print(f"\n===== OCR 区域确认 =====")
                print(f"  左上角: ({x1}, {y1})")
                print(f"  右下角: ({x2}, {y2})")
                print(f"  尺寸: {x2-x1} x {y2-y1} px")
                print(f"========================")
                print(f"\n配置代码:")
                print(f"  OCR_X1, OCR_Y1 = {x1}, {y1}")
                print(f"  OCR_X2, OCR_Y2 = {x2}, {y2}")

                # 保存裁剪区域放大图
                roi = original[y1:y2, x1:x2]
                roi_big = cv2.resize(roi, (roi.shape[1] * 4, roi.shape[0] * 4),
                                     interpolation=cv2.INTER_NEAREST)
                cv2.imwrite("output/ocr_region_crop.png", roi_big)
                print(f"\n裁剪放大已保存: output/ocr_region_crop.png")
                break
            else:
                print("请先拖拽框选区域")

    cv2.destroyAllWindows()


def test_ocr():
    """快速测试 OCR 读取"""
    from coordinate_reader import CoordinateReader

    game_window = list_and_select_window(title_hint=CAPTURE_WINDOW_TITLE)
    if game_window is None:
        print("未找到游戏窗口")
        return
    cap = ScreenCapture(window_obj=game_window)
    reader = CoordinateReader()

    print("连续读取 10 次坐标...")
    for i in range(10):
        frame = cap.grab()
        if frame is None:
            print(f"  [{i}] 截图失败")
            continue
        coord = reader.read(frame)
        print(f"  [{i}] 坐标: {coord}")
        cv2.waitKey(200)

    print(f"成功率: {reader.success_rate:.0%}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ocr":
        test_ocr()
    else:
        main()
