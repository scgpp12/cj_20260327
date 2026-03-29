"""
motion_detector.py - 运动检测模块
支持限定检测区域 + 排除自身角色
"""

import cv2
import numpy as np
from config import (
    MOTION_BLUR_KSIZE,
    MOTION_THRESHOLD,
    MOTION_DILATE_ITERATIONS,
    MOTION_MIN_AREA,
    MOTION_MAX_AREA,
    EXCLUDE_CENTER_ENABLED,
)

# 检测区域
try:
    from config import DETECT_ZONE_ENABLED, DETECT_ZONE_X1, DETECT_ZONE_Y1, DETECT_ZONE_X2, DETECT_ZONE_Y2
    _HAS_ZONE = DETECT_ZONE_ENABLED
except ImportError:
    _HAS_ZONE = False

# 自身排除区域
try:
    from config import EXCLUDE_CENTER_X1, EXCLUDE_CENTER_Y1, EXCLUDE_CENTER_X2, EXCLUDE_CENTER_Y2
    _SELF_ABSOLUTE = True
except ImportError:
    _SELF_ABSOLUTE = False
    try:
        from config import EXCLUDE_CENTER_WIDTH_RATIO, EXCLUDE_CENTER_HEIGHT_RATIO
    except ImportError:
        pass


def _boxes_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """检查两个矩形是否有重叠"""
    if ax1 >= bx2 or ax2 <= bx1:
        return False
    if ay1 >= by2 or ay2 <= by1:
        return False
    return True


class MotionDetector:
    """基于帧差法的运动检测器，支持区域限定和自身排除"""

    def __init__(self):
        self.prev_gray = None

    def detect(self, frame):
        """
        检测运动区域

        Returns:
            list of (x, y, w, h): 运动目标框列表（已排除自身）
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (MOTION_BLUR_KSIZE, MOTION_BLUR_KSIZE), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return []

        # 帧差
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        # 如果有检测区域限定，只保留区域内的差分
        if _HAS_ZONE:
            mask_zone = np.zeros_like(diff)
            mask_zone[DETECT_ZONE_Y1:DETECT_ZONE_Y2, DETECT_ZONE_X1:DETECT_ZONE_X2] = 255
            diff = cv2.bitwise_and(diff, mask_zone)

        # 二值化
        _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

        # 膨胀 + 闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=MOTION_DILATE_ITERATIONS)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤
        frame_h, frame_w = frame.shape[:2]
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MOTION_MIN_AREA:
                continue
            if area > MOTION_MAX_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # 排除与自身重叠的运动区域
            if EXCLUDE_CENTER_ENABLED and self._overlaps_self(x, y, w, h, frame_w, frame_h):
                continue

            boxes.append((x, y, w, h))

        return boxes

    def _overlaps_self(self, x, y, w, h, frame_w, frame_h):
        """检查运动框是否与自身角色区域重叠"""
        if _SELF_ABSOLUTE:
            return _boxes_overlap(
                x, y, x + w, y + h,
                EXCLUDE_CENTER_X1, EXCLUDE_CENTER_Y1,
                EXCLUDE_CENTER_X2, EXCLUDE_CENTER_Y2,
            )
        else:
            cx, cy = frame_w // 2, frame_h // 2
            try:
                half_w = int(frame_w * EXCLUDE_CENTER_WIDTH_RATIO / 2)
                half_h = int(frame_h * EXCLUDE_CENTER_HEIGHT_RATIO / 2)
                return _boxes_overlap(
                    x, y, x + w, y + h,
                    cx - half_w, cy - half_h,
                    cx + half_w, cy + half_h,
                )
            except NameError:
                return False

    def reset(self):
        """重置状态"""
        self.prev_gray = None
