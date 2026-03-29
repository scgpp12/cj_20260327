"""
pathfinder.py - A* 寻路模块
在当前帧画面上实时规划绕障路径。

用法：
    pf = Pathfinder()
    waypoints = pf.find_path(frame, target_x, target_y)
    # waypoints = [(x1,y1), (x2,y2), ...] 像素坐标拐点列表
"""

import cv2
import numpy as np
import heapq
import math

from config import (
    SELF_CENTER_X, SELF_CENTER_Y,
    PATROL_DARK_THRESHOLD, PATROL_WALL_EXPAND,
)

# 网格参数
GRID_STEP = 16  # 每格像素

# 8 方向
MOVES = [(-1, -1), (0, -1), (1, -1),
         (-1, 0),           (1, 0),
         (-1, 1),  (0, 1),  (1, 1)]
MOVE_COSTS = [1.414, 1.0, 1.414,
              1.0,        1.0,
              1.414, 1.0, 1.414]


class Pathfinder:
    """A* 实时寻路"""

    def __init__(self, grid_step=GRID_STEP, dark_thresh=None, wall_expand=None):
        self.grid_step = grid_step
        self.dark_thresh = dark_thresh or PATROL_DARK_THRESHOLD
        self.wall_expand = wall_expand if wall_expand is not None else PATROL_WALL_EXPAND

        # 缓存
        self._grid = None
        self._grid_h = 0
        self._grid_w = 0
        self._last_path = None          # 上次完整路径（网格坐标）
        self._last_waypoints = None     # 上次简化路径（像素坐标）
        self._last_target = None        # 上次目标位置

    def find_path(self, frame, target_x, target_y):
        """
        从角色位置到目标位置规划路径。

        Args:
            frame: 当前帧 BGR
            target_x, target_y: 目标像素坐标

        Returns:
            list of (x, y): 像素坐标拐点列表（不含起点），或 None
        """
        # 构建网格
        self._build_grid(frame)

        # 转网格坐标
        start_gx = SELF_CENTER_X // self.grid_step
        start_gy = SELF_CENTER_Y // self.grid_step
        goal_gx = int(target_x) // self.grid_step
        goal_gy = int(target_y) // self.grid_step

        # 边界检查
        if not self._valid(start_gx, start_gy) or not self._valid(goal_gx, goal_gy):
            return None

        # 起点或终点是墙 → 找附近可走点
        if self._grid[start_gy, start_gx] == 0:
            start_gx, start_gy = self._nearest_walkable(start_gx, start_gy)
            if start_gx is None:
                return None

        if self._grid[goal_gy, goal_gx] == 0:
            goal_gx, goal_gy = self._nearest_walkable(goal_gx, goal_gy)
            if goal_gx is None:
                return None

        # A* 寻路
        path = self._astar((start_gx, start_gy), (goal_gx, goal_gy))
        if path is None:
            self._last_path = None
            self._last_waypoints = None
            return None

        self._last_path = path

        # 简化路径 → 拐点
        simplified = self._simplify(path)

        # 网格坐标 → 像素坐标（去掉起点）
        waypoints = []
        for gx, gy in simplified[1:]:  # 跳过起点
            px = gx * self.grid_step + self.grid_step // 2
            py = gy * self.grid_step + self.grid_step // 2
            waypoints.append((px, py))

        self._last_waypoints = waypoints
        self._last_target = (target_x, target_y)
        return waypoints

    def get_next_waypoint(self, frame, target_x, target_y, arrived_dist=40):
        """
        获取下一个要走到的拐点。如果已到达当前拐点就弹出下一个。

        Returns:
            (x, y) 下一个拐点像素坐标，或 None（已到达/无路径）
        """
        # 检查是否需要重新规划
        need_replan = (
            self._last_waypoints is None or
            len(self._last_waypoints) == 0 or
            self._last_target is None or
            abs(target_x - self._last_target[0]) > 100 or
            abs(target_y - self._last_target[1]) > 100
        )

        if need_replan:
            self.find_path(frame, target_x, target_y)

        if not self._last_waypoints:
            return None

        # 检查是否到达当前拐点
        wx, wy = self._last_waypoints[0]
        dist = math.sqrt((SELF_CENTER_X - wx) ** 2 + (SELF_CENTER_Y - wy) ** 2)

        if dist < arrived_dist:
            self._last_waypoints.pop(0)
            if not self._last_waypoints:
                return None
            wx, wy = self._last_waypoints[0]

        return (wx, wy)

    def draw_path(self, frame, color=(0, 255, 255), thickness=2):
        """在画面上绘制当前路径"""
        if not self._last_waypoints:
            return

        # 画从角色到第一个拐点的线
        pts = [(SELF_CENTER_X, SELF_CENTER_Y)] + list(self._last_waypoints)

        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, thickness)

        # 画拐点
        for wx, wy in self._last_waypoints:
            cv2.circle(frame, (wx, wy), 5, (0, 0, 255), -1)

    def _build_grid(self, frame):
        """构建可走性网格（深渊膨胀法）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        step = self.grid_step

        self._grid_h = h // step
        self._grid_w = w // step

        # 按亮度标记深渊核心
        raw = np.zeros((self._grid_h, self._grid_w), dtype=np.uint8)
        for gy in range(self._grid_h):
            for gx in range(self._grid_w):
                py = gy * step
                px = gx * step
                region = gray[py:py + step, px:px + step]
                raw[gy, gx] = 1 if region.mean() >= self.dark_thresh else 0

        # 深渊膨胀
        if self.wall_expand > 0:
            wall_mask = (raw == 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.wall_expand * 2 + 1, self.wall_expand * 2 + 1)
            )
            expanded = cv2.dilate(wall_mask, kernel, iterations=1)
            self._grid = (expanded == 0).astype(np.uint8)
        else:
            self._grid = raw

    def _valid(self, gx, gy):
        return 0 <= gx < self._grid_w and 0 <= gy < self._grid_h

    def _nearest_walkable(self, gx, gy, max_search=10):
        """找最近的可走格子"""
        for r in range(1, max_search):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if self._valid(nx, ny) and self._grid[ny, nx] == 1:
                        return nx, ny
        return None, None

    def _astar(self, start, goal):
        """A* 核心算法"""
        sx, sy = start
        gx, gy = goal

        def heuristic(x, y):
            dx = abs(x - gx)
            dy = abs(y - gy)
            return max(dx, dy) + 0.414 * min(dx, dy)

        open_set = []
        heapq.heappush(open_set, (heuristic(sx, sy), 0, sx, sy))
        came_from = {}
        g_score = {(sx, sy): 0}
        visited = set()

        while open_set:
            _, cost, cx, cy = heapq.heappop(open_set)

            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            if cx == gx and cy == gy:
                path = [(gx, gy)]
                while path[-1] != (sx, sy):
                    path.append(came_from[path[-1]])
                path.reverse()
                return path

            for i, (dx, dy) in enumerate(MOVES):
                nx, ny = cx + dx, cy + dy
                if self._valid(nx, ny) and self._grid[ny, nx] == 1:
                    new_cost = cost + MOVE_COSTS[i]
                    if (nx, ny) not in g_score or new_cost < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_cost
                        f = new_cost + heuristic(nx, ny)
                        heapq.heappush(open_set, (f, new_cost, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)

            # 搜索上限
            if len(visited) > 10000:
                return None

        return None

    def _simplify(self, path, min_angle=15):
        """去掉直线中间点，只保留拐点"""
        if not path or len(path) <= 2:
            return path

        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            px, py = path[i - 1]
            cx, cy = path[i]
            nx, ny = path[i + 1]

            a1 = math.atan2(cy - py, cx - px)
            a2 = math.atan2(ny - cy, nx - cx)
            diff = abs(math.degrees(a2 - a1))
            if diff > 180:
                diff = 360 - diff
            if diff >= min_angle:
                simplified.append(path[i])

        simplified.append(path[-1])
        return simplified
