"""
hp_detector.py - 血条检测模块 (ver04 精简版)
检测方式：红色颜色填充 + 黑色边框
"""

import os
import cv2
import numpy as np

from config import (
    HP_COLOR_RANGES,
    HP_MIN_WIDTH, HP_MAX_WIDTH, HP_MAX_HEIGHT,
    HP_MIN_ASPECT_RATIO, HP_COLOR_FILL_RATIO,
    EXCLUDE_LEFT_RATIO, EXCLUDE_BOTTOM_RATIO,
    EXCLUDE_MINIMAP_RIGHT_RATIO, EXCLUDE_MINIMAP_TOP_RATIO,
    EXCLUDE_TOP_RATIO,
)


class HPDetector:
    """血条检测器：红色填充 + 黑色边框"""

    def scan_full_frame(self, frame):
        """
        全画面扫描血条

        Returns:
            list of dict: [{"hp_box": (x,y,w,h), "target_box": (x,y,w,h), "source": str}, ...]
        """
        results = []

        # 红色填充检测
        for bar in self._scan_color_bars(frame):
            results.append({"hp_box": bar, "source": "color"})

        # 黑色边框检测
        for bar in self._scan_black_borders(frame):
            # 去重：如果和颜色检测重叠就跳过
            if not self._overlaps_any(bar, [r["hp_box"] for r in results]):
                results.append({"hp_box": bar, "source": "border"})

        # 为每个血条推测怪物区域（血条下方）
        fh, fw = frame.shape[:2]
        for info in results:
            bx, by, bw, bh = info["hp_box"]
            tx = max(0, bx - 10)
            ty = by + bh
            tw = min(bw + 20, fw - tx)
            th = min(int(bw * 1.5), fh - ty)
            if th < 10:
                th = 50
            info["target_box"] = (tx, ty, tw, th)

        return results

    def _apply_ui_mask(self, mask, h, w):
        """排除 UI 区域"""
        mask[:int(h * EXCLUDE_TOP_RATIO), :] = 0
        mask[:, :int(w * EXCLUDE_LEFT_RATIO)] = 0
        mask[int(h * (1.0 - EXCLUDE_BOTTOM_RATIO)):, :] = 0
        mask[:int(h * EXCLUDE_MINIMAP_TOP_RATIO),
             int(w * (1.0 - EXCLUDE_MINIMAP_RIGHT_RATIO)):] = 0

    def _scan_color_bars(self, frame):
        """全图颜色扫描，找红色横条"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        for cr in HP_COLOR_RANGES:
            lower = np.array(cr["lower"], dtype=np.uint8)
            upper = np.array(cr["upper"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        self._apply_ui_mask(mask, h, w)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bars = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if not self._check_bar_shape(bw, bh):
                continue

            # 填充率
            region = mask[by:by + bh, bx:bx + bw]
            fill = np.count_nonzero(region) / (bw * bh) if bw * bh > 0 else 0
            if fill < HP_COLOR_FILL_RATIO:
                continue

            bars.append((bx, by, bw, bh))

        return bars

    def _scan_black_borders(self, frame):
        """
        全图扫描血条黑色边框 — 用 Canny 边缘 + 矩形轮廓检测
        即使血条为空（全黑），边框的锐利边缘也能被检测到
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Canny 边缘检测（对锐利矩形边框效果好）
        edges = cv2.Canny(gray, 50, 150)
        self._apply_ui_mask(edges, h, w)

        # 横向闭运算，连接边框的上下边缘
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bars = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)

            # 血条边框尺寸：宽 15-50，高 3-8
            if bw < 15 or bw > 50 or bh < 3 or bh > 8:
                continue
            if bw / bh < 2.5:
                continue

            # 矩形度：轮廓近似矩形
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            # 验证：边框区域内的平均亮度应该较低（黑色边框+空/少量填充）
            roi = gray[by:by + bh, bx:bx + bw]
            mean_val = float(roi.mean())
            if mean_val > 80:  # 太亮不是血条
                continue

            # 验证：边框正下方应该有非黑色像素（怪物身体）
            below_y = min(by + bh + 5, h - 1)
            below_region = gray[by + bh:below_y, bx:bx + bw] if below_y > by + bh else None
            if below_region is not None and below_region.size > 0:
                below_mean = float(below_region.mean())
                if below_mean < 15:  # 下方也是纯黑 → 可能是深渊边缘，不是血条
                    continue

            # 验证：边框周围有一定的对比度（边框比周围暗）
            # 取边框上方 3px 区域
            above_y = max(0, by - 3)
            above_region = gray[above_y:by, bx:bx + bw] if by > 0 else None
            if above_region is not None and above_region.size > 0:
                above_mean = float(above_region.mean())
                # 上方比边框亮至少 10（说明有对比度，不是大面积暗区）
                if above_mean - mean_val < 5:
                    continue

            bars.append((bx, by, bw, bh))

        return self._nms_boxes(bars)

    def _check_bar_shape(self, bw, bh):
        """检查是否符合血条形状"""
        if bw < HP_MIN_WIDTH or bw > HP_MAX_WIDTH:
            return False
        if bh > HP_MAX_HEIGHT or bh == 0:
            return False
        if bw / bh < HP_MIN_ASPECT_RATIO:
            return False
        return True

    def _overlaps_any(self, box, existing, threshold=0.5):
        """检查 box 是否和 existing 中任何一个重叠"""
        bx, by, bw, bh = box
        for ex, ey, ew, eh in existing:
            ix = max(0, min(bx + bw, ex + ew) - max(bx, ex))
            iy = max(0, min(by + bh, ey + eh) - max(by, ey))
            inter = ix * iy
            area_min = min(bw * bh, ew * eh)
            if area_min > 0 and inter / area_min > threshold:
                return True
        return False

    def _nms_boxes(self, boxes, overlap_thresh=0.5):
        """非极大值抑制"""
        if not boxes:
            return []

        arr = np.array(boxes)
        x1, y1 = arr[:, 0], arr[:, 1]
        x2, y2 = x1 + arr[:, 2], y1 + arr[:, 3]
        areas = arr[:, 2] * arr[:, 3]
        order = areas.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / np.minimum(areas[i], areas[order[1:]])
            order = order[np.where(iou < overlap_thresh)[0] + 1]

        return [boxes[i] for i in keep]
