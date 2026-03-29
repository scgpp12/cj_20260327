"""
grid_navigator.py - 网格覆盖寻路系统（OCR精确坐标版）

基于 OCR 读取游戏世界坐标（300×300地图），实现30分钟不走回头路。
坐标系: 0:0=左上角, X向右, Y向下
"""

import math
import numpy as np
import cv2
from collections import deque

from config import (
    SELF_CENTER_X, SELF_CENTER_Y,
    PATROL_DARK_THRESHOLD,
)

# 格子状态
CELL_UNKNOWN = 0   # 未探索
CELL_VISITED = 1   # 已走过
CELL_WALL = 2      # 墙壁/深渊

# 8 方向
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

NEIGHBORS_8 = [(-1, -1), (0, -1), (1, -1),
               (-1, 0),           (1, 0),
               (-1, 1),  (0, 1),  (1, 1)]


class GridMap:
    """300×300 网格地图，1格 = 1个游戏坐标"""

    MAP_SIZE = 300

    def __init__(self):
        self.visited = set()     # 已走过的坐标 {(x, y)}
        self.walls = set()       # 墙壁坐标 {(x, y)}

    def mark_visited(self, x, y):
        """标记坐标及周围9格(3×3)为已走过"""
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx <= self.MAP_SIZE and 0 <= ny <= self.MAP_SIZE:
                    self.visited.add((nx, ny))

    def mark_wall(self, x, y):
        """标记坐标为墙壁"""
        if 0 <= x <= self.MAP_SIZE and 0 <= y <= self.MAP_SIZE:
            if (x, y) not in self.visited:  # 已走过的不标墙
                self.walls.add((x, y))

    def is_visited(self, x, y):
        return (x, y) in self.visited

    def is_wall(self, x, y):
        return (x, y) in self.walls

    def is_walkable(self, x, y):
        """可走 = 在地图范围内 且 不是墙"""
        return (0 <= x <= self.MAP_SIZE and
                0 <= y <= self.MAP_SIZE and
                (x, y) not in self.walls)

    def coverage_ratio(self):
        """覆盖率 = 已走 / (已走 + 未走可走区域)"""
        total = len(self.visited)
        if total == 0:
            return 0.0
        # 简单用已走格子数 / 地图总面积（粗略估计）
        return total / (self.MAP_SIZE * self.MAP_SIZE)

    def get_stats(self):
        return {
            "visited": len(self.visited),
            "walls": len(self.walls),
            "coverage": self.coverage_ratio(),
        }


class CoveragePlanner:
    """蛇形扫描 + BFS 前沿搜索"""

    def __init__(self, grid_map):
        self.grid = grid_map
        # 蛇形扫描状态
        self._scan_y = 0         # 当前扫描行
        self._scan_right = True  # True=从左到右, False=从右到左
        self._scan_x = 0        # 当前扫描列

    def find_next_target(self, cur_x, cur_y, max_search=3000):
        """
        找下一个要走的未访问坐标。
        优先蛇形扫描，失败则 BFS 找最近未访问格。

        Returns:
            (x, y) 或 None
        """
        # 策略1：蛇形扫描 — 找当前行最近的未访问格
        target = self._snake_scan(cur_x, cur_y)
        if target:
            return target

        # 策略2：BFS 找最近未访问格
        return self._bfs_frontier(cur_x, cur_y, max_search)

    def _snake_scan(self, cur_x, cur_y):
        """蛇形行扫描：找当前或附近行的未访问格"""
        grid = self.grid

        # 从当前 Y 坐标开始搜索
        for y_offset in range(0, grid.MAP_SIZE):
            for sign in [0, 1, -1]:  # 先当前行，再上下
                scan_y = cur_y + y_offset * (sign if sign != 0 else 1)
                if scan_y < 0 or scan_y > grid.MAP_SIZE:
                    continue

                # 该行扫描方向
                if scan_y % 2 == 0:
                    x_range = range(0, grid.MAP_SIZE + 1)  # 左到右
                else:
                    x_range = range(grid.MAP_SIZE, -1, -1)  # 右到左

                for x in x_range:
                    if not grid.is_visited(x, scan_y) and grid.is_walkable(x, scan_y):
                        return (x, scan_y)

                if y_offset == 0:
                    break  # y_offset=0 时只搜当前行

        return None

    def _bfs_frontier(self, cur_x, cur_y, max_search):
        """BFS 找最近的未访问可走格"""
        start = (cur_x, cur_y)
        queue = deque([start])
        visited_bfs = {start}
        searched = 0

        while queue and searched < max_search:
            x, y = queue.popleft()
            searched += 1

            for dx, dy in NEIGHBORS_8:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited_bfs:
                    continue
                visited_bfs.add((nx, ny))

                if not self.grid.is_walkable(nx, ny):
                    continue

                if not self.grid.is_visited(nx, ny):
                    return (nx, ny)  # 找到未访问格

                queue.append((nx, ny))

        return None


class GridNavigator:
    """
    网格覆盖导航器（OCR坐标版）
    结合 GridMap + OCR CoordinateReader + CoveragePlanner
    """

    def __init__(self):
        self.grid = GridMap()
        self.planner = CoveragePlanner(self.grid)
        self.dark_threshold = PATROL_DARK_THRESHOLD

        # OCR 坐标读取器
        from coordinate_reader import CoordinateReader
        self.coord_reader = CoordinateReader()

        # 当前位置
        self.world_x = -1
        self.world_y = -1

        # 目标航点
        self._waypoint = None    # (x, y) 游戏坐标

        # 统计
        self.total_steps = 0

    def track_frame(self, frame):
        """
        每帧调用：OCR 读取坐标，标记已访问。
        """
        coord = self.coord_reader.read(frame)
        if coord is not None:
            old_x, old_y = self.world_x, self.world_y
            self.world_x, self.world_y = coord
            self.grid.mark_visited(self.world_x, self.world_y)
            # 每10次成功读取打印一次
            if self.coord_reader._total_success % 10 == 1:
                stats = self.grid.get_stats()
                print(f"[OCR] 坐标:({self.world_x},{self.world_y}) "
                      f"已走:{stats['visited']}格 成功率:{self.coord_reader.success_rate:.0%}")
        else:
            if self.coord_reader._fail_count == 1:
                print(f"[OCR] 读取失败, 上次坐标:({self.world_x},{self.world_y})")

    def get_direction(self, frame, terrain_scores=None):
        """
        获取下一步方向。

        Returns:
            方向名（如 "UP_LEFT"）或 None（全覆盖完）
        """
        if self.world_x < 0:
            return None  # OCR 还没读到坐标

        # 用地形亮度标墙
        if terrain_scores:
            self._mark_walls_from_terrain(terrain_scores)

        # 检查是否到达航点
        if self._waypoint is not None:
            dist = abs(self.world_x - self._waypoint[0]) + abs(self.world_y - self._waypoint[1])
            if dist <= 2:  # 曼哈顿距离 ≤ 2 视为到达
                self._waypoint = None
                self.total_steps += 1

        # 需要新航点
        if self._waypoint is None:
            target = self.planner.find_next_target(self.world_x, self.world_y)
            if target is None:
                return None  # 全部探索完
            self._waypoint = target

        # 计算方向
        return self._direction_to(
            self.world_x, self.world_y,
            self._waypoint[0], self._waypoint[1]
        )

    def on_direction_failed(self, direction):
        """A*规划失败：标记该方向前方为墙，重新规划航点"""
        if self.world_x < 0 or direction not in DIRECTIONS:
            return

        dx, dy = DIRECTIONS[direction]
        # 标记前方 3-8 格为墙（A*看到的障碍通常比撞墙远）
        for i in range(3, 9):
            wx = self.world_x + dx * i
            wy = self.world_y + dy * i
            self.grid.mark_wall(wx, wy)

        # 清除航点，下次 get_direction 会重新规划
        self._waypoint = None

        stats = self.grid.get_stats()
        print(f"[GRID] A*失败标墙({direction}) 墙:{stats['walls']} 覆盖:{stats['coverage']:.1%}")

    def on_stuck(self, current_dir=None):
        """撞墙：标记前方为墙，重新规划"""
        if self.world_x < 0:
            return

        if current_dir and current_dir in DIRECTIONS:
            dx, dy = DIRECTIONS[current_dir]

            # 标记前方 1-5 格为墙
            for i in range(1, 6):
                wx = self.world_x + dx * i
                wy = self.world_y + dy * i
                self.grid.mark_wall(wx, wy)

            # 扇形扩展
            for ndx, ndy in NEIGHBORS_8:
                dot = dx * ndx + dy * ndy
                if dot > 0:
                    for i in range(1, 3):
                        wx = self.world_x + ndx * i
                        wy = self.world_y + ndy * i
                        self.grid.mark_wall(wx, wy)

        # 清除航点，重新规划
        self._waypoint = None

        stats = self.grid.get_stats()
        print(f"[GRID] 撞墙标墙 已走:{stats['visited']} 墙:{stats['walls']} 覆盖:{stats['coverage']:.1%}")

    def _mark_walls_from_terrain(self, terrain_scores):
        """根据地形亮度标墙"""
        if self.world_x < 0:
            return

        # 每个方向的亮度 → 估计几格外是墙
        # 采样距离 80,150,220px ≈ 游戏坐标 2,4,6 格（粗略估计）
        dist_to_grids = [2, 4, 6]

        for d_name, (dx, dy) in DIRECTIONS.items():
            brightness = terrain_scores.get(d_name, 255)
            if brightness < self.dark_threshold:
                for g in dist_to_grids:
                    wx = self.world_x + dx * g
                    wy = self.world_y + dy * g
                    self.grid.mark_wall(wx, wy)

    def get_viz_data(self):
        """返回可视化数据"""
        return {
            "visited": self.grid.visited.copy(),
            "walls": self.grid.walls.copy(),
            "map_size": self.grid.MAP_SIZE,
            "world_pos": (self.world_x, self.world_y),
            "waypoint": self._waypoint,
            "coverage": self.grid.coverage_ratio(),
            "total_steps": self.total_steps,
            "ocr_rate": self.coord_reader.success_rate,
        }

    def _direction_to(self, from_x, from_y, to_x, to_y):
        """计算 8 方向"""
        dx = to_x - from_x
        dy = to_y - from_y

        if abs(dx) < 1 and abs(dy) < 1:
            return "DOWN"

        angle = math.atan2(dy, dx)
        deg = math.degrees(angle)

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
