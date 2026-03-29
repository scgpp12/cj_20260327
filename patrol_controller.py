"""
patrol_controller.py - 自动巡逻寻怪 + 障碍物躲避
状态机：IDLE → PATROL → STUCK → COMBAT
使用 PostMessage 点击地面移动角色
"""

import time
import random
import cv2
import numpy as np
import ctypes

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

# 8 个巡逻方向 (dx, dy) — 等角视角下的方向偏移
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

# 相邻方向表（撞墙后排除用）
ADJACENT = {
    "UP":         ["UP_LEFT", "UP_RIGHT"],
    "DOWN":       ["DOWN_LEFT", "DOWN_RIGHT"],
    "LEFT":       ["UP_LEFT", "DOWN_LEFT"],
    "RIGHT":      ["UP_RIGHT", "DOWN_RIGHT"],
    "UP_LEFT":    ["UP", "LEFT"],
    "UP_RIGHT":   ["UP", "RIGHT"],
    "DOWN_LEFT":  ["DOWN", "LEFT"],
    "DOWN_RIGHT": ["DOWN", "RIGHT"],
}


class PatrolController:
    """
    自动巡逻控制器

    状态机:
        IDLE    - 等待中，无怪物超过 idle_timeout 后进入 PATROL
        PATROL  - 巡逻中，随机方向移动
        STUCK   - 撞墙了，换方向
        COMBAT  - 战斗中（由外部控制）
    """

    def __init__(self):
        self.state = "IDLE"
        self.enabled = True

        # 时间戳
        self.last_target_time = time.time()  # 上次看到怪物的时间
        self.last_move_time = 0              # 上次发出移动指令的时间
        self.move_frame = None               # 发出移动指令时的帧（用于撞墙检测）
        self.move_frame_time = 0             # 发出指令的时间

        # 方向
        self.current_dir = random.choice(DIR_NAMES)
        self.blocked_dirs = set()            # 被阻挡的方向
        self.consecutive_stuck = 0           # 连续撞墙次数

        # 右键点按状态（点一下走一步，约80px/1秒）
        self._rbutton_held = False           # 兼容：攻击时释放用
        self._held_hwnd = None               # 上次点击的窗口句柄
        self._held_lparam = 0                # 兼容保留

        # ---- 路径记忆 ----
        self.dir_visit_count = {d: 0 for d in DIR_NAMES}
        self.dir_history = []
        self.DIR_HISTORY_MAX = 12  # 记住最近 12 步，更强防回头

        # 追踪目标（范围外的怪物坐标）
        self._chase_target = None
        self._chase_bypass = False
        self._bypass_steps = 0

        # 远处怪物位置提示（巡逻方向加分用）
        self._monster_hints = []
        self._monster_stuck_count = 0  # 朝怪物方向连续撞墙次数

        # 墙壁掩码（给可视化用）
        self.wall_mask = None

        # ---- 网格覆盖导航（扫地机器人模式）----
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

        # 寻路拐点列表
        self._path_waypoints = []  # [(px, py), ...]
        self._path_target = None   # 寻路目标

        # 巡逻信息（给可视化用）
        self.info = {
            "state": "IDLE",
            "direction": self.current_dir,
            "click_pos": None,
            "grid_data": None,
        }

    def set_chase_target(self, x, y):
        """设置追踪目标（范围外的怪物位置）"""
        self._chase_target = (int(x), int(y))
        self._path_target = self._chase_target

    def clear_chase_target(self):
        """清除追踪目标"""
        self._chase_target = None

    def set_monster_hints(self, positions):
        """设置远处怪物位置提示（每帧从 main.py 刷新，不持久化）
        Args:
            positions: [(x, y), ...] 怪物画面坐标列表，空列表=没有远处怪物
        """
        self._monster_hints = positions
        if not positions:
            self._monster_stuck_count = 0  # 没有怪物时重置撞墙计数

    def _release_rbutton(self):
        """停止移动（点按模式下主要是标记状态，兼容攻击器调用）"""
        self._rbutton_held = False
        self._held_lparam = 0

    def on_target_found(self):
        """外部调用：发现怪物了 → 停止跑步，进入战斗"""
        self._release_rbutton()
        self.state = "COMBAT"
        self.last_target_time = time.time()
        self.consecutive_stuck = 0
        self._monster_stuck_count = 0
        self._monster_hints = []  # 战斗中清除远处怪物提示
        self.blocked_dirs.clear()
        self.info["state"] = "COMBAT"

    def on_target_lost(self):
        """外部调用：怪物消失了"""
        if self.state == "COMBAT":
            self.state = "IDLE"
            self.last_target_time = time.time()
            self.info["state"] = "IDLE"

    def update(self, frame, game_hwnd):
        """
        每帧调用，根据状态执行巡逻逻辑
        """
        if not self.enabled or game_hwnd is None:
            return

        # 每帧刷新墙壁掩码（跟随地图滚动）
        self._update_wall_mask(frame)

        # 每帧更新网格导航的位置追踪
        if self.grid_nav is not None:
            self.grid_nav.track_frame(frame)
            self.info["grid_data"] = self.grid_nav.get_viz_data()

        now = time.time()

        if self.state == "IDLE":
            if now - self.last_target_time >= PATROL_IDLE_TIMEOUT:
                self._pick_direction(frame)
                self.state = "PATROL"
                self.info["state"] = "PATROL"
                print(f"[PATROL] 开始巡逻，方向: {self.current_dir}")

        elif self.state == "PATROL":
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
                # 保持当前方向直走，不换方向（撞墙/A*失败时才换）
                self._do_move(frame, game_hwnd)

        elif self.state == "COMBAT":
            pass

    def _opposite(self, d):
        """获取反方向"""
        opp = {
            "UP": "DOWN", "DOWN": "UP",
            "LEFT": "RIGHT", "RIGHT": "LEFT",
            "UP_LEFT": "DOWN_RIGHT", "DOWN_RIGHT": "UP_LEFT",
            "UP_RIGHT": "DOWN_LEFT", "DOWN_LEFT": "UP_RIGHT",
        }
        return opp.get(d)

    def _pick_direction(self, frame):
        """
        选择巡逻方向（简洁版）：
        A: 不走回头路（排除反方向）
        B: 远离墙体（地形最亮）
        C: 排除 blocked_dirs
        """

        # 简洁版：地形最亮(远离墙) + 不回头 + 不走blocked
        terrain = self._scan_terrain(frame)

        # 可选方向：排除 blocked + 排除回头
        opposite = self._opposite(self.current_dir)
        available = [d for d in DIR_NAMES
                     if d not in self.blocked_dirs and d != opposite]
        if not available:
            # 全堵了，只排除回头
            available = [d for d in DIR_NAMES if d != opposite]
        if not available:
            self.blocked_dirs.clear()
            available = DIR_NAMES[:]

        # 按地形亮度排序（最亮 = 远离墙体 = 优先）
        scored = [(d, terrain[d]) for d in available]
        scored.sort(key=lambda x: x[1], reverse=True)

        self.current_dir = scored[0][0]

        top_info = " | ".join(f"{d}:{s:.0f}" for d, s in scored[:4])
        print(f"[PATROL] 选方向: {top_info} → {self.current_dir}")

        # 记录
        self.dir_visit_count[self.current_dir] = \
            self.dir_visit_count.get(self.current_dir, 0) + 1
        self.dir_history.append(self.current_dir)
        if len(self.dir_history) > self.DIR_HISTORY_MAX:
            self.dir_history.pop(0)

        self.info["direction"] = self.current_dir
        self.info["terrain"] = terrain

    def _do_move(self, frame, game_hwnd):
        """发送移动指令（右键点按走路，一步约80px/1秒）— 优先用 A* 规划"""

        # 确定目标点
        if self._chase_target is not None:
            target_x, target_y = self._chase_target
            mode = "CHASE"
        else:
            dx, dy = DIRECTIONS[self.current_dir]
            target_x = SELF_CX + int(dx * PATROL_CLICK_DISTANCE)
            target_y = SELF_CY + int(dy * PATROL_CLICK_DISTANCE)
            mode = "MONSTER" if self._monster_hints else "PATROL"

        # A* 规划路径
        if self.pathfinder is not None:
            waypoints = self.pathfinder.find_path(frame, target_x, target_y)
            if waypoints and len(waypoints) > 0:
                # 过滤所有拐点：离角色中心 < 200px 的全部跳过
                waypoints = [(wx, wy) for wx, wy in waypoints
                             if ((wx - SELF_CX) ** 2 + (wy - SELF_CY) ** 2) ** 0.5 >= 200]

                if not waypoints:
                    # 所有拐点都太近 → 直接用方向目标点
                    click_x, click_y = target_x, target_y
                    self.info["direction"] = mode
                    print(f"[PATROL] A*拐点都太近 → 直线走({click_x},{click_y})")
                else:
                    click_x, click_y = waypoints[0]
                n_wp = len(waypoints)
                self.info["direction"] = f"A*{mode}"
                print(f"[PATROL] A*{mode} -> walk({click_x},{click_y}) ({n_wp}个拐点)")
            else:
                # A* 失败 → 告诉Grid标墙，换方向重试
                print(f"[PATROL] A*{mode} 失败，换方向")
                if self.grid_nav is not None:
                    self.grid_nav.on_direction_failed(self.current_dir)
                self.blocked_dirs.add(self.current_dir)
                self._pick_direction(frame)
                dx, dy = DIRECTIONS[self.current_dir]
                target_x = SELF_CX + int(dx * PATROL_CLICK_DISTANCE)
                target_y = SELF_CY + int(dy * PATROL_CLICK_DISTANCE)

                waypoints = self.pathfinder.find_path(frame, target_x, target_y)
                if waypoints and len(waypoints) > 0:
                    click_x, click_y = waypoints[0]
                    self.info["direction"] = f"A*{mode}"
                    print(f"[PATROL] A*重试 -> walk({click_x},{click_y})")
                else:
                    # 还是失败 → 最亮方向直线走
                    terrain = self._scan_terrain(frame)
                    best_dir = max(terrain, key=terrain.get)
                    dx, dy = DIRECTIONS[best_dir]
                    click_x = SELF_CX + int(dx * PATROL_CLICK_DISTANCE)
                    click_y = SELF_CY + int(dy * PATROL_CLICK_DISTANCE)
                    self.current_dir = best_dir
                    self.info["direction"] = f"FALLBACK({best_dir})"
                    print(f"[PATROL] FALLBACK({best_dir}) -> walk({click_x},{click_y})")
        else:
            click_x, click_y = target_x, target_y
            self.info["direction"] = mode

        # 确保点击坐标在画面内
        h, w = frame.shape[:2]
        click_x = max(10, min(w - 10, click_x))
        click_y = max(10, min(h - 10, click_y))

        try:
            lparam = _make_lparam(click_x, click_y)

            # 右键点按（按下+松开 = 走一步）
            PostMessage(game_hwnd, WM_RBUTTONDOWN, MK_RBUTTON, lparam)
            PostMessage(game_hwnd, WM_RBUTTONUP, 0, lparam)
            self._held_hwnd = game_hwnd
        except Exception as e:
            print(f"[PATROL] 移动指令失败: {e}")
            return

        # 记录移动帧（用于撞墙检测）
        self.move_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.move_frame_time = time.time()
        self.last_move_time = time.time()
        self.info["click_pos"] = (click_x, click_y)

        print(f"[PATROL] WALK {self.current_dir} -> click({click_x},{click_y})")

    def _check_stuck(self, frame):
        """检查是否撞墙（画面静止检测）"""
        if self.move_frame is None:
            return False

        now = time.time()
        elapsed = now - self.move_frame_time

        # 至少等 stuck_timeout 秒才判定
        if elapsed < PATROL_STUCK_TIMEOUT:
            return False

        # 比较当前帧和移动指令帧的差异
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.move_frame, gray)
        mean_diff = diff.mean()

        if mean_diff < PATROL_STUCK_THRESHOLD:
            return True

        # 画面有变化 = 角色在移动 → 通知攻击器清除误检标记
        self._moved_since_last_check = True
        return False

    def _get_chase_direction(self):
        """根据追踪目标计算理想方向名"""
        if self._chase_target is None:
            return None
        tx, ty = self._chase_target
        dx = tx - SELF_CX
        dy = ty - SELF_CY

        # 计算角度 → 映射到 8 方向
        import math
        angle = math.atan2(dy, dx)  # -π ~ π
        deg = math.degrees(angle)   # -180 ~ 180

        # 角度 → 方向名（右=0°, 下=90°, 左=180°, 上=-90°）
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

    # 扇形展开顺序：从目标方向开始，左右交替扩展
    FAN_ORDER = {
        "UP":         ["UP", "UP_LEFT", "UP_RIGHT", "LEFT", "RIGHT", "DOWN_LEFT", "DOWN_RIGHT", "DOWN"],
        "DOWN":       ["DOWN", "DOWN_LEFT", "DOWN_RIGHT", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "UP"],
        "LEFT":       ["LEFT", "UP_LEFT", "DOWN_LEFT", "UP", "DOWN", "UP_RIGHT", "DOWN_RIGHT", "RIGHT"],
        "RIGHT":      ["RIGHT", "UP_RIGHT", "DOWN_RIGHT", "UP", "DOWN", "UP_LEFT", "DOWN_LEFT", "LEFT"],
        "UP_LEFT":    ["UP_LEFT", "UP", "LEFT", "UP_RIGHT", "DOWN_LEFT", "RIGHT", "DOWN", "DOWN_RIGHT"],
        "UP_RIGHT":   ["UP_RIGHT", "UP", "RIGHT", "UP_LEFT", "DOWN_RIGHT", "LEFT", "DOWN", "DOWN_LEFT"],
        "DOWN_LEFT":  ["DOWN_LEFT", "DOWN", "LEFT", "DOWN_RIGHT", "UP_LEFT", "RIGHT", "UP", "UP_RIGHT"],
        "DOWN_RIGHT": ["DOWN_RIGHT", "DOWN", "RIGHT", "DOWN_LEFT", "UP_RIGHT", "LEFT", "UP", "UP_LEFT"],
    }

    def _handle_stuck(self, frame):
        """撞墙处理：先朝前方盲点3下（可能有未检测到的怪挡路），再标记墙壁"""
        self._release_rbutton()
        stuck_dir = self.current_dir

        # 0. 朝移动方向盲点 3 下左键（打掉可能挡路的隐形怪）
        # 距离参考攻击模式：直方向40px，斜方向50px
        if self._held_hwnd is not None:
            dx, dy = DIRECTIONS.get(stuck_dir, (0, 0))
            is_diagonal = abs(dx) == 1 and abs(dy) == 1
            blind_dist = 50 if is_diagonal else 40
            # 攻击圆心在角色上方50px
            origin_x = SELF_CX - 5
            origin_y = SELF_CY - 50 + 10
            click_x = origin_x + int(dx * blind_dist)
            click_y = origin_y + int(dy * blind_dist)
            try:
                lparam = _make_lparam(click_x, click_y)
                for _ in range(3):
                    PostMessage(self._held_hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
                    PostMessage(self._held_hwnd, WM_LBUTTONUP, 0, lparam)
                print(f"[PATROL] 撞墙! 朝 {stuck_dir} 方向盲点3下 ({click_x},{click_y})")
            except Exception as e:
                print(f"[PATROL] 盲点失败: {e}")

        # 1. 清除 A* 缓存路径
        if self.pathfinder is not None:
            self.pathfinder._last_waypoints = None

        # 2. 通知网格导航（如果启用）
        if self.grid_nav is not None:
            self.grid_nav.on_stuck(stuck_dir)

        # 3. 朝怪物方向撞墙计数
        if self._monster_hints:
            self._monster_stuck_count += 1
            if self._monster_stuck_count >= 3:
                print(f"[PATROL] 朝怪物方向撞墙3次，放弃怪物方向")
                self._monster_hints = []
                self._monster_stuck_count = 0

        # 4. 换方向
        self.blocked_dirs.add(stuck_dir)
        if self.consecutive_stuck >= 3:
            self.blocked_dirs.clear()
            self.consecutive_stuck = 0
        # _pick_direction 内部也有 blocked_dirs 全满时自动清除的逻辑
        self._pick_direction(frame)

        self.move_frame = None
        print(f"[PATROL] 换方向: {self.current_dir}")

    def _scan_terrain(self, frame):
        """
        扫描 8 个方向的地面亮度。带缓存，同一帧只扫一次。
        """
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
                    total_score += 0
                    total_weight += weight
                    continue

                region = gray[y1:y2, x1:x2]
                brightness = float(region.mean())
                total_score += brightness * weight
                total_weight += weight

            avg_score = total_score / total_weight if total_weight > 0 else 0
            scores[d_name] = avg_score

        # 缓存结果
        self._terrain_cache_id = frame_id
        self._terrain_cache_result = scores
        return scores

    def _update_wall_mask(self, frame):
        """每帧刷新墙壁掩码，跟随地图滚动实时更新"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        radius = 300
        y1 = max(0, SELF_CY - radius)
        y2 = min(h, SELF_CY + radius)
        x1 = max(0, SELF_CX - radius)
        x2 = min(w, SELF_CX + radius)
        roi = gray[y1:y2, x1:x2]
        _, wall_roi = cv2.threshold(roi, PATROL_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        self.wall_mask = np.zeros((h, w), dtype=np.uint8)
        self.wall_mask[y1:y2, x1:x2] = wall_roi

    def _is_dark_ahead(self, frame, direction):
        """检测指定方向前方是否是深渊"""
        scores = self._scan_terrain(frame)
        return scores.get(direction, 0) < PATROL_DARK_THRESHOLD
