"""
grid_navigator.py - 网格覆盖寻路系统（扫地机器人模式）
动态网格地图 + 相位相关定位 + BFS 前沿覆盖
不需要预知地图大小，边走边建图。
"""

import math
import numpy as np
import cv2
from collections import deque

from config import (
    SELF_CENTER_X, SELF_CENTER_Y,
    PATROL_DARK_THRESHOLD,
    GRID_CELL_SIZE, GRID_PHASE_ROI_W, GRID_PHASE_ROI_H,
    GRID_MAX_SHIFT_PER_FRAME,
)

# 格子状态
CELL_UNKNOWN = 0   # 未探索
CELL_OPEN = 1      # 可走（扫描到亮色）
CELL_VISITED = 2   # 已走过
CELL_WALL = 3      # 墙壁/深渊

# 8 方向（和 patrol_controller 一致）
DIRECTIONS = {
    "UP":         (0, -1),
    "DOWN":       (0, 1),
    "LEFT":       (-1, 0),
    "RIGHT":      (1, 0),
    "UP_LEFT":    (-1, -1),
    "UP_RIGHT":   (1, -1),
    "DOWN_LEFT":  (-1, 1),
    "DOWN_RIGHT": (1, 1),
}

# 8 邻居偏移
NEIGHBORS_8 = [(-1, -1), (0, -1), (1, -1),
               (-1, 0),           (1, 0),
               (-1, 1),  (0, 1),  (1, 1)]


class GridMap:
    """动态网格地图，用 dict 存储，自动扩展"""

    def __init__(self, cell_size=64):
        self.cell_size = cell_size
        self.cells = {}  # {(gx, gy): state}

    def world_to_grid(self, wx, wy):
        """世界坐标 → 网格坐标"""
        gx = int(math.floor(wx / self.cell_size))
        gy = int(math.floor(wy / self.cell_size))
        return gx, gy

    def grid_to_world(self, gx, gy):
        """网格坐标 → 世界坐标（格子中心）"""
        wx = gx * self.cell_size + self.cell_size / 2
        wy = gy * self.cell_size + self.cell_size / 2
        return wx, wy

    def get_cell(self, gx, gy):
        return self.cells.get((gx, gy), CELL_UNKNOWN)

    def set_cell(self, gx, gy, state):
        current = self.cells.get((gx, gy), CELL_UNKNOWN)
        # 不降级：VISITED 不会被覆盖为 OPEN
        if current == CELL_VISITED and state == CELL_OPEN:
            return
        self.cells[(gx, gy)] = state

    def mark_visited(self, wx, wy):
        gx, gy = self.world_to_grid(wx, wy)
        self.cells[(gx, gy)] = CELL_VISITED

    def mark_walls_from_terrain(self, wx, wy, terrain_scores, dark_threshold):
        """
        根据 8 方向地形亮度扫描结果，标记周围格子。
        terrain_scores: {"UP": brightness, "DOWN": ..., ...}
        """
        sample_distances = [80, 150, 220]

        for d_name, (dx, dy) in DIRECTIONS.items():
            brightness = terrain_scores.get(d_name, 0)

            for dist in sample_distances:
                # 方向上的世界坐标
                sample_wx = wx + dx * dist
                sample_wy = wy + dy * dist
                gx, gy = self.world_to_grid(sample_wx, sample_wy)

                current = self.get_cell(gx, gy)
                if current == CELL_VISITED:
                    continue  # 已走过的不改

                if brightness < dark_threshold:
                    self.set_cell(gx, gy, CELL_WALL)
                else:
                    if current == CELL_UNKNOWN:
                        self.set_cell(gx, gy, CELL_OPEN)

    def coverage_ratio(self):
        """已覆盖比例"""
        visited = sum(1 for v in self.cells.values() if v == CELL_VISITED)
        walkable = sum(1 for v in self.cells.values() if v in (CELL_OPEN, CELL_VISITED))
        if walkable == 0:
            return 0.0
        return visited / walkable

    def get_bounds(self):
        """获取已知区域的边界"""
        if not self.cells:
            return 0, 0, 0, 0
        xs = [k[0] for k in self.cells]
        ys = [k[1] for k in self.cells]
        return min(xs), min(ys), max(xs), max(ys)


class PositionTracker:
    """用相位相关估算角色世界位置"""

    def __init__(self, roi_w=600, roi_h=400, max_shift=30):
        self.world_x = 0.0
        self.world_y = 0.0
        self.prev_gray = None
        self.confidence = 0.0
        self.roi_w = roi_w
        self.roi_h = roi_h
        self.max_shift = max_shift

        # ROI 区域（居中于角色，避开 UI）
        self.roi_x1 = SELF_CENTER_X - roi_w // 2
        self.roi_y1 = SELF_CENTER_Y - roi_h // 2
        self.roi_x2 = self.roi_x1 + roi_w
        self.roi_y2 = self.roi_y1 + roi_h

        # Hanning 窗口（减少边缘效应）
        self.hanning = cv2.createHanningWindow((roi_w, roi_h), cv2.CV_32F)

    def update(self, frame):
        """
        更新位置估算。返回 (dx, dy, confidence)
        dx, dy 是世界坐标偏移（像素）
        """
        h, w = frame.shape[:2]

        # 裁切 ROI，确保不越界
        x1 = max(0, self.roi_x1)
        y1 = max(0, self.roi_y1)
        x2 = min(w, self.roi_x2)
        y2 = min(h, self.roi_y2)

        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        # 调整 Hanning 窗口尺寸（如果 ROI 被裁切了）
        gh, gw = gray.shape
        if gw != self.roi_w or gh != self.roi_h:
            hanning = cv2.createHanningWindow((gw, gh), cv2.CV_32F)
        else:
            hanning = self.hanning

        gray_windowed = gray * hanning

        if self.prev_gray is None or self.prev_gray.shape != gray_windowed.shape:
            self.prev_gray = gray_windowed
            return 0.0, 0.0, 0.0

        # 相位相关：检测画面平移
        (dx, dy), confidence = cv2.phaseCorrelate(self.prev_gray, gray_windowed)

        self.prev_gray = gray_windowed
        self.confidence = confidence

        # 画面向右移 = 角色向右走 = world_x 增加
        # phaseCorrelate 返回的是 prev 到 curr 的偏移
        # 如果画面右移（世界左移），dx > 0，角色实际向右走

        # 限制最大偏移（防止异常跳变）
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > self.max_shift:
            # 异常大的偏移，可能是场景切换，忽略
            return 0.0, 0.0, 0.0

        # 低置信度也忽略
        if confidence < 0.05:
            return 0.0, 0.0, 0.0

        # 累加到世界坐标
        self.world_x += dx
        self.world_y += dy

        return dx, dy, confidence

    def on_stuck(self):
        """撞墙时调用，不更新位置"""
        pass


class CoveragePlanner:
    """BFS 前沿搜索覆盖规划"""

    def __init__(self, grid_map):
        self.grid = grid_map

    def find_nearest_frontier(self, wx, wy, max_search=2000):
        """
        BFS 找最近的未探索前沿格子。
        返回 (gx, gy) 或 None（全部探索完）。
        """
        start = self.grid.world_to_grid(wx, wy)
        queue = deque([start])
        visited_bfs = {start}
        searched = 0

        while queue and searched < max_search:
            gx, gy = queue.popleft()
            searched += 1

            for ngx, ngy in self._neighbors(gx, gy):
                if (ngx, ngy) in visited_bfs:
                    continue
                visited_bfs.add((ngx, ngy))

                state = self.grid.get_cell(ngx, ngy)

                if state == CELL_UNKNOWN:
                    # 找到前沿！
                    return (ngx, ngy)

                if state in (CELL_OPEN, CELL_VISITED):
                    queue.append((ngx, ngy))
                # CELL_WALL: 不穿过

        return None  # 全部探索完或被墙包围

    def _neighbors(self, gx, gy):
        """8 方向邻居"""
        for dx, dy in NEIGHBORS_8:
            yield gx + dx, gy + dy


class GridNavigator:
    """
    网格覆盖导航器（外部接口）
    结合 GridMap + PositionTracker + CoveragePlanner
    """

    def __init__(self, cell_size=None, dark_threshold=None):
        cs = cell_size or GRID_CELL_SIZE
        dt = dark_threshold or PATROL_DARK_THRESHOLD

        self.grid = GridMap(cs)
        self.tracker = PositionTracker(
            roi_w=GRID_PHASE_ROI_W,
            roi_h=GRID_PHASE_ROI_H,
            max_shift=GRID_MAX_SHIFT_PER_FRAME,
        )
        self.planner = CoveragePlanner(self.grid)
        self.dark_threshold = dt

        self._waypoint = None         # 目标网格坐标 (gx, gy)
        self._waypoint_world = None   # 目标世界坐标 (wx, wy)
        self._arrived_threshold = cs * 0.7

        # 统计
        self.total_steps = 0
        self.frontier_count = 0

    def track_frame(self, frame):
        """
        每帧调用，更新位置和网格（轻量操作）。
        不做路径规划。
        """
        dx, dy, conf = self.tracker.update(frame)
        wx, wy = self.tracker.world_x, self.tracker.world_y

        # 标记当前位置为已走过
        self.grid.mark_visited(wx, wy)

    def get_direction(self, frame, terrain_scores=None):
        """
        获取下一步方向。返回方向名（如 "UP_LEFT"）或 None。
        terrain_scores: _scan_terrain() 的返回值，用于标记墙壁。
        """
        wx, wy = self.tracker.world_x, self.tracker.world_y

        # 用地形扫描结果更新网格墙壁
        if terrain_scores:
            self.grid.mark_walls_from_terrain(wx, wy, terrain_scores, self.dark_threshold)

        # 检查是否到达航点
        if self._waypoint_world is not None:
            dist = math.sqrt(
                (wx - self._waypoint_world[0]) ** 2 +
                (wy - self._waypoint_world[1]) ** 2
            )
            if dist < self._arrived_threshold:
                self._waypoint = None
                self._waypoint_world = None
                self.total_steps += 1

        # 需要新航点
        if self._waypoint is None:
            frontier = self.planner.find_nearest_frontier(wx, wy)
            if frontier is None:
                return None  # 全部探索完

            self._waypoint = frontier
            self._waypoint_world = self.grid.grid_to_world(*frontier)
            self.frontier_count += 1

        # 计算方向
        twx, twy = self._waypoint_world
        return self._direction_to(wx, wy, twx, twy)

    def on_stuck(self, current_dir=None):
        """撞墙：在移动方向标记墙壁 + 扇形扩展，重新规划"""
        wx, wy = self.tracker.world_x, self.tracker.world_y

        if current_dir and current_dir in DIRECTIONS:
            dx, dy = DIRECTIONS[current_dir]

            # 标记前方 1-5 格为墙
            for dist_mult in range(1, 6):
                wall_wx = wx + dx * self.grid.cell_size * dist_mult
                wall_wy = wy + dy * self.grid.cell_size * dist_mult
                gx, gy = self.grid.world_to_grid(wall_wx, wall_wy)
                current = self.grid.get_cell(gx, gy)
                if current != CELL_VISITED:
                    self.grid.set_cell(gx, gy, CELL_WALL)

            # 也标记相邻方向的前方格子（扇形扩展，防止 BFS 从旁边绕过来选同一个前沿）
            for ndx, ndy in NEIGHBORS_8:
                # 只选和撞墙方向相近的邻居
                dot = dx * ndx + dy * ndy
                if dot > 0:  # 方向相似
                    for dist_mult in range(1, 4):
                        wall_wx = wx + ndx * self.grid.cell_size * dist_mult
                        wall_wy = wy + ndy * self.grid.cell_size * dist_mult
                        gx, gy = self.grid.world_to_grid(wall_wx, wall_wy)
                        current = self.grid.get_cell(gx, gy)
                        if current != CELL_VISITED:
                            self.grid.set_cell(gx, gy, CELL_WALL)

        # 如果当前航点就在被标记为墙的方向上，也清掉
        if self._waypoint is not None:
            wp_state = self.grid.get_cell(*self._waypoint)
            if wp_state == CELL_WALL:
                self._waypoint = None
                self._waypoint_world = None

        # 清除航点，强制重新规划
        self._waypoint = None
        self._waypoint_world = None
        self.tracker.on_stuck()

        # 打印调试
        walls = sum(1 for v in self.grid.cells.values() if v == CELL_WALL)
        visited = sum(1 for v in self.grid.cells.values() if v == CELL_VISITED)
        print(f"[GRID] 地图状态: {len(self.grid.cells)} 格 (已走:{visited} 墙:{walls})")

    def get_viz_data(self):
        """返回可视化数据"""
        return {
            "cells": dict(self.grid.cells),  # 复制
            "cell_size": self.grid.cell_size,
            "world_pos": (self.tracker.world_x, self.tracker.world_y),
            "waypoint": self._waypoint_world,
            "coverage": self.grid.coverage_ratio(),
            "total_cells": len(self.grid.cells),
            "total_steps": self.total_steps,
        }

    def _direction_to(self, from_x, from_y, to_x, to_y):
        """计算从 (from) 到 (to) 的 8 方向"""
        dx = to_x - from_x
        dy = to_y - from_y

        if abs(dx) < 1 and abs(dy) < 1:
            return "DOWN"  # 原地不动，默认向下

        angle = math.atan2(dy, dx)
        deg = math.degrees(angle)

        # 角度 → 8 方向（右=0°, 下=90°, 左=±180°, 上=-90°）
        dir_angles = {
            "RIGHT": 0, "DOWN_RIGHT": 45, "DOWN": 90, "DOWN_LEFT": 135,
            "LEFT": 180, "UP_LEFT": -135, "UP": -90, "UP_RIGHT": -45,
        }

        best_dir = "RIGHT"
        best_diff = 999
        for d, a in dir_angles.items():
            diff = abs(deg - a)
            if diff > 180:
                diff = 360 - diff
            if diff < best_diff:
                best_diff = diff
                best_dir = d

        return best_dir
