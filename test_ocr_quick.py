# -*- coding: utf-8 -*-
"""快速 OCR 测试：用游戏窗口截图测试坐标读取"""
import sys
import cv2
import mss
import numpy as np
import time

sys.stdout.reconfigure(encoding='utf-8')

from coordinate_reader import CoordinateReader, OCR_X1, OCR_Y1, OCR_X2, OCR_Y2

def main():
    sct = mss.mss()
    reader = CoordinateReader()

    # 游戏窗口在第二显示器 pos=(3832,-8) size=1936x1048
    # 实际客户区大约从 (3840, 0) 开始，大小约 1920x1032
    monitor = {"left": 3840, "top": 0, "width": 1920, "height": 1032}
    print(f"截取区域: {monitor}")
    print(f"OCR 区域: ({OCR_X1},{OCR_Y1})-({OCR_X2},{OCR_Y2})")

    for i in range(10):
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        print(f"\n[{i}] 帧尺寸: {frame.shape}")

        # 检查 OCR 区域是否在范围内
        h, w = frame.shape[:2]
        if OCR_Y2 > h or OCR_X2 > w:
            print(f"    !! OCR 区域超出帧范围 ({w}x{h})")
            # 尝试在底部找坐标文字
            bottom = frame[h-50:h, 0:300]
            cv2.imwrite(f"output/ocr_bottom_{i}.png", bottom)
            print(f"    已保存底部区域到 output/ocr_bottom_{i}.png")
        else:
            roi = frame[OCR_Y1:OCR_Y2, OCR_X1:OCR_X2]
            print(f"    ROI 尺寸: {roi.shape}, 平均亮度: {roi.mean():.0f}")

            coord = reader.read(frame)
            print(f"    OCR 结果: {coord}")

            # 保存 ROI
            roi_big = cv2.resize(roi, (roi.shape[1]*4, roi.shape[0]*4),
                                 interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"output/ocr_roi_{i}.png", roi_big)

        time.sleep(0.5)

    sct.close()

if __name__ == "__main__":
    main()
