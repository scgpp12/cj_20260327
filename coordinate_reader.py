"""
coordinate_reader.py - OCR 读取游戏世界坐标

从游戏画面左下角状态栏读取角色坐标（格式: 236:239）
坐标系: 0:0 = 左上角, X向右增大, Y向下增大, 地图300x300
"""

import re
import cv2
import pytesseract

# Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR 裁剪区域（游戏窗口客户区像素坐标）
OCR_X1, OCR_Y1 = 87, 1009
OCR_X2, OCR_Y2 = 148, 1032

# 坐标解析正则：支持中文冒号和英文冒号
_COORD_PATTERN = re.compile(r'(\d+)\s*[：:,.]\s*(\d+)')


class CoordinateReader:
    """OCR 读取游戏世界坐标"""

    def __init__(self):
        self._last_coord = None      # 上次成功读取的坐标
        self._fail_count = 0         # 连续失败次数
        self._total_reads = 0
        self._total_success = 0

    def read(self, frame):
        """
        从游戏画面读取坐标

        Args:
            frame: BGR 格式游戏画面

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
                config='--psm 7 -c tessedit_char_whitelist=0123456789:.'
            ).strip()
        except Exception as e:
            self._fail_count += 1
            if self._fail_count >= 10:
                print(f"[OCR] 连续{self._fail_count}次失败: {e}")
            return self._last_coord

        # 解析坐标
        match = _COORD_PATTERN.search(text)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))

            # 范围校验（地图 300x300）
            if 0 <= x <= 300 and 0 <= y <= 300:
                self._last_coord = (x, y)
                self._fail_count = 0
                self._total_success += 1
                return (x, y)

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
