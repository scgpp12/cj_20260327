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
        full_mask = mask1 | mask2

        # 排除 UI 区域
        left_px = int(w * self.exclude_left)
        bottom_px = int(h * (1 - self.exclude_bottom))
        full_mask[:, :left_px] = 0
        full_mask[bottom_px:, :] = 0
        rt_x = int(w * self.exclude_right_top[0])
        rt_y = int(h * self.exclude_right_top[1])
        full_mask[:rt_y, rt_x:] = 0

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

        # ======= 分两阶段检测 =======
        # 阶段1: 近身优先区 (100×100 绿框)
        NEAR_HALF = 50
        nx1 = max(0, self.self_cx - NEAR_HALF)
        ny1 = max(0, self.self_cy - NEAR_HALF)
        nx2 = min(w, self.self_cx + NEAR_HALF)
        ny2 = min(h, self.self_cy + NEAR_HALF)

        # 保存近身框坐标供可视化
        self.near_box = (nx1, ny1, nx2, ny2)

        near_results = self._detect_from_mask(full_mask, nx1, ny1, nx2, ny2)
        if near_results:
            near_results.sort(key=lambda r: r["dist"])
            return near_results

        # 阶段2: 全屏检测（近身区以外）
        all_results = self._detect_from_mask(full_mask, 0, 0, w, h)
        all_results.sort(key=lambda r: r["dist"])
        return all_results

    def _detect_from_mask(self, full_mask, x1, y1, x2, y2):
        """从指定区域的 mask 中检测红球"""
        # 裁剪区域
        roi_mask = np.zeros_like(full_mask)
        roi_mask[y1:y2, x1:x2] = full_mask[y1:y2, x1:x2]

        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            if area <= self.max_area:
                # 小面积：面积>5000跳过圆形度（大块必定是红球不是噪点）
                if area < 5000:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter < 1:
                        continue
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < self.min_circularity:
                        continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = x + bw // 2
                cy = y + bh // 2
                dist = ((cx - self.self_cx) ** 2 + (cy - self.self_cy) ** 2) ** 0.5
                results.append({
                    "box": (x, y, bw, bh),
                    "center": (cx, cy),
                    "area": int(area),
                    "circularity": 0,
                    "dist": dist,
                })
            else:
                # 大面积粘连：距离变换拆分
                x, y, bw, bh = cv2.boundingRect(cnt)
                blob_mask = full_mask[y:y+bh, x:x+bw].copy()

                dist_transform = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
                _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
                if max_val < 3:
                    continue
                _, sure_fg = cv2.threshold(dist_transform, max_val * 0.5, 255, 0)
                sure_fg = np.uint8(sure_fg)

                n_labels, labels = cv2.connectedComponents(sure_fg)
                for label_id in range(1, n_labels):
                    pts = np.where(labels == label_id)
                    if len(pts[0]) < 10:
                        continue
                    cy_local = int(np.mean(pts[0]))
                    cx_local = int(np.mean(pts[1]))
                    cx_abs = x + cx_local
                    cy_abs = y + cy_local
                    dist = ((cx_abs - self.self_cx) ** 2 + (cy_abs - self.self_cy) ** 2) ** 0.5
                    ball_r = 20
                    results.append({
                        "box": (cx_abs - ball_r, cy_abs - ball_r, ball_r * 2, ball_r * 2),
                        "center": (cx_abs, cy_abs),
                        "area": int(area // max(n_labels - 1, 1)),
                        "circularity": 0,
                        "dist": dist,
                    })

        return results
