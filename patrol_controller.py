"""
patrol_controller.py - 自动巡逻寻怪 (v2 预规划路线模式)
状态机：IDLE → PATROL → STUCK → COMBAT
预规划路线优先，A* 局部避障，撞墙跳过路线点
"""

import time
import os
import cv2
import numpy as np
import ctypes
import math

from config import (
    PATROL_IDLE_TIMEOUT,
    PATROL_MOVE_INTERVAL,
    PATROL_CLICK_DISTANCE,
    PATROL_STUCK_TIMEOUT,
    PATROL_STUCK_THRESHOLD,
    PATROL_DARK_THRESHOLD,
    SELF_CENTER_X,
    SELF_CENTER_Y,
    PATROL_USE_GRID,
)

# PostMessage 消息常量
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
MK_LBUTTON = 0x0001
MK_RBUTTON = 0x0002
user32 = ctypes.windll.user32
PostMessage = user32.PostMessageW


def _make_lparam(x, y):
    x, y = int(x), int(y)
    return (y << 16) | (x & 0xFFFF)


SELF_CX = SELF_CENTER_X
SELF_CY = SELF_CENTER_Y

# 8 个巡逻方向 (dx, dy)
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
DIR_NAMES = list(DIRECTIONS.keys())


class PatrolController:
    """
    自动巡逻控制器（预规划路线模式）

    状态机:
        IDLE    - 等待中，无怪物超过 idle_timeout 后进入 PATROL
        PATROL  - 沿预规划路线巡逻
        STUCK   - 撞墙了，跳过路线点
        COMBAT  - 战斗中（由外部控制）
    """

    def __init__(self):
        self.state = "IDLE"
        self.enabled = True

        # 时间戳
        self.last_target_time = time.time()
        self.last_move_time = 0
        self.move_frame = None
        self.move_frame_time = 0

        # 方向
        self.current_dir = "RIGHT"
        self.consecutive_stuck = 0

        # 右键点按状态
        self._rbutton_held = False
        self._held_hwnd = None
        self._held_lparam = 0

        # 追踪目标
        self._chase_target = None

        # 远处怪物位置提示
        self._monster_hints = []

        # ---- 预规划路线 ----
        self.route = []           # [(world_x, world_y), ...]
        self.route_index = 0      # 当前目标点索引
        self.route_mode = False   # 路线模式是否激活
        self._load_route()

        # ---- OCR坐标（通过 grid_nav 获取）----
        self.grid_nav = None
        if PATROL_USE_GRID:
            try:
                from grid_navigator import GridNavigator
                self.grid_nav = GridNavigator()
                print("[PATROL] 网格覆盖导航已启用")
            except Exception as e:
                print(f"[PATROL] 网格导航加载失败: {e}")

        # A* 寻路器
        self.pathfinder = None
        try:
            from pathfinder import Pathfinder
            self.pathfinder = Pathfinder()
            print("[PATROL] A* 寻路已启用")
        except Exception as e:
            print(f"[PATROL] A* 寻路不可用: {e}")

        # 巡逻信息（给可视化用）
        self.info = {
            "state": "IDLE",
            "direction": self.current_dir,
            "click_pos": None,
            "terrain": None,
            "route_index": 0,
            "route_total": len(self.route),
            "visited_centroid_screen": None,
        }

    def _load_route(self):
        """加载预规划路线"""
        route_path = os.path.join(os.path.dirname(__file__), "output", "map_patrol_route.txt")
        if not os.path.exists(route_path):
            print(f"[ROUTE] 路线文件不存在: {route_path}")
            return

        try:
            with open(route_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "," in line:
                        x, y = line.split(",")
                        self.route.append((int(x), int(y)))
            self.route_mode = len(self.route) > 0
            print(f"[ROUTE] 加载路线: {len(self.route)} 个点 (模式: {'ON' if self.route_mode else 'OFF'})")
        except Exception as e:
            print(f"[ROUTE] 加载路线失败: {e}")

    def _find_nearest_route_index(self, wx, wy):
        """在路线中找到离当前坐标最近的点"""
        best_idx = 0
        best_dist = 999999
        for i, (rx, ry) in enumerate(self.route):
            d = abs(rx - wx) + abs(ry - wy)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def set_chase_target(self, x, y):
        self._chase_target = (int(x), int(y))

    def clear_chase_target(self):
        self._chase_target = None

    def set_monster_hints(self, positions):
        self._monster_hints = positions

    def _release_rbutton(self):
        self._rbutton_held = False
        self._held_lparam = 0

    def on_target_found(self):
        """发现怪物 → 停止巡逻，进入战斗"""
        self._release_rbutton()
        self.state = "COMBAT"
        self.last_target_time = time.time()
        self.consecutive_stuck = 0
        self.info["state"] = "COMBAT"

    def on_target_lost(self):
        """怪物消失了"""
        if self.state == "COMBAT":
            self.state = "IDLE"
            self.last_target_time = time.time()
            self.info["state"] = "IDLE"

    def update(self, frame, game_hwnd):
        """每帧调用"""
        if not self.enabled or game_hwnd is None:
            return

        # 每帧更新 OCR 坐标
        if self.grid_nav is not None:
            self.grid_nav.track_frame(frame)

            # 首次获取坐标时，定位到路线最近点
            if self.route_mode and self.route_index == 0 and self.grid_nav.world_x >= 0:
                self.route_index = self._find_nearest_route_index(
                    self.grid_nav.world_x, self.grid_nav.world_y)
                print(f"[ROUTE] 定位到路线点 #{self.route_index}/{len(self.route)} "
                      f"({self.route[self.route_index][0]},{self.route[self.route_index][1]})")

        now = time.time()

        if self.state == "IDLE":
            if now - self.last_target_time >= PATROL_IDLE_TIMEOUT:
                self.state = "PATROL"
                self.info["state"] = "PATROL"
                self._update_direction()
                print(f"[PATROL] 开始巡逻，方向: {self.current_dir}")

        elif self.state == "PATROL":
            # 检查是否到达当前路线点
            self._check_route_arrival()

            if self._check_stuck(frame):
                self.state = "STUCK"
                self.info["state"] = "STUCK"
                self.consecutive_stuck += 1
                print(f"[PATROL] 撞墙! 连续{self.consecutive_stuck}次")
                self._handle_stuck(frame)
                self.state = "PATROL"
                self.info["state"] = "PATROL"
                return

            if now - self.last_move_time >= PATROL_MOVE_INTERVAL:
                self._do_move(frame, game_hwnd)

        elif self.state == "COMBAT":
            pass

    def _check_route_arrival(self):
        """检查是否到达当前路线点"""
        if not self.route_mode or self.grid_nav is None:
            return
        if self.grid_nav.world_x < 0:
            return
        if self.route_index >= len(self.route):
            # 路线走完，从头循环
            self.route_index = 0
            print(f"[ROUTE] 路线走完! 从头循环")
            return

        wx, wy = self.grid_nav.world_x, self.grid_nav.world_y
        rx, ry = self.route[self.route_index]
        dist = abs(wx - rx) + abs(wy - ry)

        if dist <= 3:
            self.route_index += 1
            self._update_direction()
            if self.route_index % 50 == 0:
                progress = self.route_index / len(self.route) * 100
                print(f"[ROUTE] 进度: {self.route_index}/{len(self.route)} ({progress:.0f}%)")

    def _update_direction(self):
        """根据路线更新当前方向"""
        if self.route_mode and self.grid_nav and self.grid_nav.world_x >= 0:
            direction = self._get_route_direction()
            if direction:
                self.current_dir = direction
                self.info["direction"] = self.current_dir
                self.info["route_index"] = self.route_index
                return

        # fallback: 用地形扫描
        self.info["direction"] = self.current_dir

    def _get_route_direction(self):
        """计算从当前坐标到路线点的 8 方向"""
        if not self.route or self.route_index >= len(self.route):
            return None
        if self.grid_nav is None or self.grid_nav.world_x < 0:
            return None

        wx, wy = self.grid_nav.world_x, self.grid_nav.world_y
        rx, ry = self.route[self.route_index]
        dx = rx - wx
        dy = ry - wy

        if dx == 0 and dy == 0:
            return None

        # 归一化到 8 方向
        ndx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        ndy = 1 if dy > 0 else (-1 if dy < 0 else 0)

        for d_name, (ddx, ddy) in DIRECTIONS.items():
            if ddx == ndx and ddy == ndy:
                return d_name
        return None

    def _do_move(self, frame, game_hwnd):
        """发送移动指令 — 路线模式 + A* 局部避障"""

        # 更新方向
        self._update_direction()

        # 确定目标点
        if self._chase_target is not None:
            click_x, click_y = self._chase_target
            mode = "CHASE"
        else:
            dx, dy = DIRECTIONS[self.current_dir]
            click_x = SELF_CX + int(dx * PATROL_CLICK_DISTANCE)
            click_y = SELF_CY + int(dy * PATROL_CLICK_DISTANCE)
            mode = "ROUTE" if self.route_mode else "PATROL"

        # A* 局部避障
        if self.pathfinder is not None:
            waypoints = self.pathfinder.find_path(frame, click_x, click_y)
            if waypoints:
                # 过滤太近的拐点
                waypoints = [(wx, wy) for wx, wy in waypoints
                             if ((wx - SELF_CX) ** 2 + (wy - SELF_CY) ** 2) ** 0.5 >= 200]
                if waypoints:
                    click_x, click_y = waypoints[0]
                    mode = f"A*{mode}"
            else:
                # A* 失败 → 直线走
                pass

        # 限制在画面内
        h, w = frame.shape[:2]
        click_x = max(10, min(w - 10, click_x))
        click_y = max(10, min(h - 10, click_y))

        try:
            lparam = _make_lparam(click_x, click_y)
            PostMessage(game_hwnd, WM_RBUTTONDOWN, MK_RBUTTON, lparam)
            PostMessage(game_hwnd, WM_RBUTTONUP, 0, lparam)
            self._held_hwnd = game_hwnd
        except Exception as e:
            print(f"[PATROL] 移动指令失败: {e}")
            return

        # 记录
        self.move_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.move_frame_time = time.time()
        self.last_move_time = time.time()
        self.info["click_pos"] = (click_x, click_y)

        if self.route_mode and self.route_index < len(self.route):
            rx, ry = self.route[self.route_index]
            print(f"[PATROL] {mode} {self.current_dir} -> click({click_x},{click_y}) "
                  f"目标点#{self.route_index}({rx},{ry})")
        else:
            print(f"[PATROL] {mode} {self.current_dir} -> click({click_x},{click_y})")

    def _check_stuck(self, frame):
        """检查是否撞墙"""
        if self.move_frame is None:
            return False
        elapsed = time.time() - self.move_frame_time
        if elapsed < PATROL_STUCK_TIMEOUT:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.move_frame, gray)
        if diff.mean() < PATROL_STUCK_THRESHOLD:
            return True

        self._moved_since_last_check = True
        return False

    def _handle_stuck(self, frame):
        """撞墙处理：盲点3下 + 跳过路线点"""
        self._release_rbutton()
        stuck_dir = self.current_dir

        # 朝移动方向盲点 3 下左键
        if self._held_hwnd is not None:
            dx, dy = DIRECTIONS.get(stuck_dir, (0, 0))
            is_diagonal = abs(dx) == 1 and abs(dy) == 1
            blind_dist = 50 if is_diagonal else 40
            origin_x = SELF_CX - 5
            origin_y = SELF_CY - 50 + 10
            bx = origin_x + int(dx * blind_dist)
            by = origin_y + int(dy * blind_dist)
            try:
                lparam = _make_lparam(bx, by)
                for _ in range(3):
                    PostMessage(self._held_hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
                    PostMessage(self._held_hwnd, WM_LBUTTONUP, 0, lparam)
                print(f"[PATROL] 朝 {stuck_dir} 盲点3下 ({bx},{by})")
            except Exception:
                pass

        # 清除 A* 缓存
        if self.pathfinder is not None:
            self.pathfinder._last_waypoints = None

        # 通知网格导航
        if self.grid_nav is not None:
            self.grid_nav.on_stuck(stuck_dir)

        # 路线模式：跳过被墙挡的路线点
        if self.route_mode and self.grid_nav and self.grid_nav.world_x >= 0:
            wx, wy = self.grid_nav.world_x, self.grid_nav.world_y
            skip_count = 0
            for i in range(self.route_index, min(self.route_index + 15, len(self.route))):
                rx, ry = self.route[i]
                # 如果路线点在撞墙方向上 → 跳过
                rdx = rx - wx
                rdy = ry - wy
                if rdx == 0 and rdy == 0:
                    skip_count += 1
                    continue
                ndx = 1 if rdx > 0 else (-1 if rdx < 0 else 0)
                ndy = 1 if rdy > 0 else (-1 if rdy < 0 else 0)
                sdx, sdy = DIRECTIONS.get(stuck_dir, (0, 0))
                if ndx == sdx and ndy == sdy:
                    skip_count += 1
                else:
                    break  # 不在撞墙方向了，停止跳过
            if skip_count > 0:
                self.route_index += skip_count
                print(f"[ROUTE] 撞墙跳过 {skip_count} 个点 → #{self.route_index}")
        else:
            # 无路线模式：用地形选方向
            terrain = self._scan_terrain(frame)
            best_dir = max(terrain, key=terrain.get)
            self.current_dir = best_dir

        if self.consecutive_stuck >= 5:
            self.consecutive_stuck = 0
            # 跳过更多路线点
            if self.route_mode:
                self.route_index = min(self.route_index + 20, len(self.route) - 1)
                print(f"[ROUTE] 连续撞墙5次，大跳到 #{self.route_index}")

        self._update_direction()
        self.move_frame = None

    def _scan_terrain(self, frame):
        """扫描 8 个方向的地面亮度"""
        frame_id = id(frame)
        if hasattr(self, '_terrain_cache_id') and self._terrain_cache_id == frame_id:
            return self._terrain_cache_result
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        scores = {}
        sample_distances = [80, 150, 220]
        sample_weights = [3.0, 2.0, 1.0]
        sample_size = 30

        for d_name in DIR_NAMES:
            dx, dy = DIRECTIONS[d_name]
            total_score = 0.0
            total_weight = 0.0
            for dist, weight in zip(sample_distances, sample_weights):
                cx = SELF_CX + int(dx * dist)
                cy = SELF_CY + int(dy * dist)
                x1 = max(0, cx - sample_size)
                y1 = max(0, cy - sample_size)
                x2 = min(w, cx + sample_size)
                y2 = min(h, cy + sample_size)
                if x2 - x1 < 5 or y2 - y1 < 5:
                    total_weight += weight
                    continue
                region = gray[y1:y2, x1:x2]
                total_score += float(region.mean()) * weight
                total_weight += weight
            scores[d_name] = total_score / total_weight if total_weight > 0 else 0

        self._terrain_cache_id = frame_id
        self._terrain_cache_result = scores
        return scores
