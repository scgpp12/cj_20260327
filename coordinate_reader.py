"""
coordinate_reader.py — OCR 读取游戏世界坐标
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【职责】
  从游戏画面左下角状态栏裁剪"236:239"格式坐标文字，
  用 Tesseract OCR 识别后解析为 (x, y) 整数元组。

【调用方式】
  reader = CoordinateReader()
  coord = reader.read(frame)   # 返回 (x,y) 或上次成功值

【常见故障排查】
  1. 坐标长时间不变 → OCR_X1/Y1/X2/Y2 区域位置偏移，需要重新校准
  2. 识别率低 → 调整二值化阈值（threshold=180，暗文字降低）
  3. 快速验证 → 运行 test_ocr_quick.py 查看截图效果

【关键常量】
  OCR_X1/Y1/X2/Y2   游戏左下角坐标栏的像素范围（换分辨率后需改）
  坐标系：(0,0)=左上，X向右，Y向下，地图300×300
"""

import re
import cv2
import pytesseract

# Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR 裁剪区域（游戏窗口客户区像素坐标）
OCR_X1, OCR_Y1 = 11, 1013
OCR_X2, OCR_Y2 = 174, 1032

# 坐标解析正则：支持中文冒号和英文冒号
_COORD_PATTERN = re.compile(r'(\d+)\s*[：:,.]\s*(\d+)')


class CoordinateReader:
    """OCR 读取游戏世界坐标"""

    def __init__(self):
        self._last_coord = None      # 上次成功读取的坐标
        self._fail_count = 0         # 连续失败次数
        self._total_reads = 0
        self._total_success = 0
        self._reject_count = 0       # 连续被异常值过滤次数

    def read(self, frame, hint=None):
        """
        从游戏画面读取坐标

        Args:
            frame: BGR 格式游戏画面
            hint:  (hx, hy) 期望坐标（当前路线目标点），有多个候选时取最近的

        Returns:
            (x, y) 整数元组，失败返回上次成功值或 None
        """
        self._total_reads += 1

        h, w = frame.shape[:2]
        # 防越界
        x1 = max(0, min(OCR_X1, w))
        y1 = max(0, min(OCR_Y1, h))
        x2 = max(0, min(OCR_X2, w))
        y2 = max(0, min(OCR_Y2, h))

        if x2 <= x1 or y2 <= y1:
            self._fail_count += 1
            return self._last_coord

        # 裁剪 OCR 区域
        roi = frame[y1:y2, x1:x2]

        # 预处理：灰度 → 二值化（白字提取）
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # 放大2倍提高 OCR 精度
        binary = cv2.resize(binary, (binary.shape[1] * 2, binary.shape[0] * 2),
                           interpolation=cv2.INTER_LINEAR)

        # OCR 识别
        try:
            text = pytesseract.image_to_string(
                binary,
                config='--psm 7'
            ).strip()
        except Exception as e:
            self._fail_count += 1
            if self._fail_count >= 10:
                print(f"[OCR] 连续{self._fail_count}次失败: {e}")
            return self._last_coord

        # 提取所有候选坐标（OCR可能产生多组数字对）
        candidates = []
        for m in _COORD_PATTERN.findall(text):
            x, y = int(m[0]), int(m[1])
            if 0 <= x <= 300 and 0 <= y <= 300:
                candidates.append((x, y))

        if candidates:
            # 有 hint（当前路线目标点）时，选距离最近的候选值
            if hint is not None and len(candidates) > 1:
                hx, hy = hint
                best = min(candidates, key=lambda c: abs(c[0] - hx) + abs(c[1] - hy))
            else:
                best = candidates[0]

            # 异常值过滤：和上次坐标差距过大则丢弃（OCR误读）
            # 角色每帧移动 ≤ 3格，阈值20格已足够宽松
            if self._last_coord is not None:
                lx, ly = self._last_coord
                if abs(best[0] - lx) + abs(best[1] - ly) > 20:
                    self._reject_count += 1
                    if self._reject_count < 5:
                        # 连续5次内：丢弃异常值，返回上次坐标
                        return self._last_coord
                    else:
                        # 连续5次都被过滤：可能是真实的大位移（路线重定位后）
                        # 接受新值并重置计数
                        print(f"[OCR] 强制接受坐标({best[0]},{best[1]}) 已连续拒绝{self._reject_count}次")
            self._reject_count = 0
            self._last_coord = best
            self._fail_count = 0
            self._total_success += 1
            return best

        # 解析失败
        self._fail_count += 1
        if self._fail_count == 5:
            print(f"[OCR] 连续5次解析失败, 原始文本: '{text}'")
        return self._last_coord

    @property
    def success_rate(self):
        if self._total_reads == 0:
            return 0
        return self._total_success / self._total_reads

    @property
    def last_coord(self):
        return self._last_coord
