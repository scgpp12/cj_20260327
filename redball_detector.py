"""
redball_detector.py - 红球怪物检测
怪物模型被替换为红色球体，用 HSV 红色阈值 + 轮廓检测识别
"""

import cv2
import numpy as np


class RedBallDetector:
    """红色球体怪物检测器"""

    def __init__(self,
                 red_lower1=(0, 150, 100),
                 red_upper1=(10, 255, 255),
                 red_lower2=(170, 150, 100),
                 red_upper2=(180, 255, 255),
                 min_area=300,
                 max_area=15000,
                 min_circularity=0.3,
                 self_center_x=965,
                 self_center_y=444,
                 self_radius=30):
        """
        Args:
            red_lower1/upper1: HSV 红色范围1 (H=0~10)
            red_lower2/upper2: HSV 红色范围2 (H=170~180)
            min_area: 最小面积（过滤噪点）
            max_area: 最大面积（过滤 UI 元素）
            min_circularity: 最小圆形度 (0~1, 1=完美圆形)
            self_center_x/y: 角色画面中心坐标
            self_radius: 角色自身排除半径
        """
        self.red_lower1 = np.array(red_lower1)
        self.red_upper1 = np.array(red_upper1)
        self.red_lower2 = np.array(red_lower2)
        self.red_upper2 = np.array(red_upper2)
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.self_cx = self_center_x
        self.self_cy = self_center_y
        self.self_radius = self_radius

        # UI 排除区域（比例）
        self.exclude_left = 0.15
        self.exclude_bottom = 0.10           # 缩小：只排底部UI栏，不挡怪物
        self.exclude_right_top = (0.85, 0.20)  # (x比例, y比例)

    def detect(self, frame):
        """
        检测画面中的红色球体

        Args:
            frame: BGR 图像

        Returns:
            list of dict: [{
                "box": (x, y, w, h),      # 边界框
                "center": (cx, cy),         # 中心点
                "area": int,                # 面积
                "circularity": float,       # 圆形度
                "dist": float,              # 到角色的距离
            }, ...]
            按距离排序（最近在前）
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 双范围红色掩码
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask = mask1 | mask2

        # 排除 UI 区域
        left_px = int(w * self.exclude_left)
        bottom_px = int(h * (1 - self.exclude_bottom))
        mask[:, :left_px] = 0                          # 左侧面板
        mask[bottom_px:, :] = 0                        # 底部 UI
        rt_x = int(w * self.exclude_right_top[0])
        rt_y = int(h * self.exclude_right_top[1])
        mask[:rt_y, rt_x:] = 0                         # 右上小地图

        # 角色中心死区已取消（不再排除，避免漏检身边的怪）

        # 形态学处理：闭运算连接碎片 + 开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            # 圆形度 = 4π × 面积 / 周长²
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            # 边界框
            x, y, bw, bh = cv2.boundingRect(cnt)

            # 中心点
            cx = x + bw // 2
            cy = y + bh // 2

            # 到角色的距离
            dist = ((cx - self.self_cx) ** 2 + (cy - self.self_cy) ** 2) ** 0.5

            results.append({
                "box": (x, y, bw, bh),
                "center": (cx, cy),
                "area": int(area),
                "circularity": circularity,
                "dist": dist,
            })

        # 按距离排序
        results.sort(key=lambda r: r["dist"])
        return results
