"""
patrol_controller.py — 自动巡逻控制器（预规划路线模式）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【职责】
  按 output/map_patrol_route.txt 中预存的路线点顺序巡逻全图。
  通过 OCR 读取角色世界坐标，精确计算朝路线点的方向并右键点击移动。

【状态机】
  IDLE   → 无怪等待，超过 PATROL_IDLE_TIMEOUT 秒后进入 PATROL
  PATROL → 沿路线点顺序移动（主要工作状态）
  STUCK  → 撞墙处理（盲点3下 + 跳过路线点），处理完立即回 PATROL
  COMBAT → 外部调用 on_target_found() 触发，巡逻暂停

【移动方式】
  精确角度右键点击：用世界坐标差(rx-wx, ry-wy)归一化后
  在角色屏幕中心 ± PATROL_CLICK_DISTANCE 像素处发送 WM_RBUTTONDOWN

【卡死救援（两层）】
  1. 画面帧差检测（_check_stuck）：1.5s 画面不动 → 贴墙滑行 / 盲点
  2. 全局看门狗（_watchdog）：20s 总位移 < 10格 → 全局重定位+前推30点

【关键参数】（全部在 config.py 修改）
  PATROL_MOVE_INTERVAL   每次移动间隔（秒）
  PATROL_CLICK_DISTANCE  右键点击半径（像素）
  PATROL_STUCK_TIMEOUT   帧差卡死判定秒数
  PATROL_STUCK_THRESHOLD 帧差阈值（越大越不敏感）

【路线文件】output/map_patrol_route.txt
  格式：每行 "x,y"，坐标 = 游戏世界坐标 = map_centerline.png 像素坐标
  验证坐标是否在路上：gray[y,x] > 60 = 道路，否则 = 墙壁
"""

import time
import os
import cv2
import ctypes
import math

from config import (
    PATROL_IDLE_TIMEOUT,
    PATROL_MOVE_INTERVAL,
    PATROL_CLICK_DISTANCE,
    PATROL_STUCK_TIMEOUT,
    PATROL_STUCK_THRESHOLD,
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

        # 路线断点防抖（防止单帧OCR误读触发重定位）
        self._disconnect_count = 0  # 连续断点帧数，达到3才触发重定位

        # 贴墙滑行（绕过墙角）
        self._corner_slide_count = 0  # 当前目标点已滑行次数，超过4次才放弃转跳点
        self._slide_steps_left = 0    # 剩余滑行步数（>0时_do_move持续朝垂直方向走）
        self._slide_perp = (0.0, 0.0) # 当前滑行的垂直方向单位向量

        # OCR 坐标卡住检测
        self._last_coord = (-1, -1)
        self._last_coord_time = time.time()
        self._coord_stuck_timeout = 2.0  # 坐标N秒不变 → 触发贴墙滑行

        # 全局进度看门狗（独立于单次卡死处理，专门应对"困在死角"场景）
        self._watchdog_coord = None
        self._watchdog_time = time.time()
        self._watchdog_timeout = 20.0  # 20秒内位移 < 10格 → 强制全局重定位
        self._watchdog_min = 10

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

        # 巡逻信息（给可视化用）
        self.info = {
            "state": "IDLE",
            "direction": self.current_dir,
            "click_pos": None,
            "route_index": 0,
            "route_total": len(self.route),
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

    def _find_nearest_route_index(self, wx, wy, full_search=False):
        """
        在路线中找到离当前坐标最近的点。
        full_search=True  → 搜索全部（仅初始定位时用）
        full_search=False → 只向前搜索（当前index起，最多+100），永不后退
        """
        if full_search or self.route_index == 0:
            search_start = 0
            search_end = len(self.route)
        else:
            search_start = self.route_index
            search_end = min(self.route_index + 100, len(self.route))

        best_idx = search_start
        best_dist = 999999
        for i in range(search_start, search_end):
            rx, ry = self.route[i]
            d = abs(rx - wx) + abs(ry - wy)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

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

        # 每帧更新 OCR 坐标，传入当前路线目标点作为 hint，用于多候选OCR值中选最近的
        if self.grid_nav is not None:
            hint = None
            if self.route_mode and self.route_index < len(self.route):
                hint = self.route[self.route_index]
            self.grid_nav.track_frame(frame, hint=hint)

            # 首次获取坐标时，定位到路线最近点
            if self.route_mode and self.route_index == 0 and self.grid_nav.world_x >= 0:
                self.route_index = self._find_nearest_route_index(
                    self.grid_nav.world_x, self.grid_nav.world_y, full_search=True)
                print(f"[ROUTE] 定位到路线点 #{self.route_index}/{len(self.route)} "
                      f"({self.route[self.route_index][0]},{self.route[self.route_index][1]})")

        now = time.time()

        if self.state == "IDLE":
            if now - self.last_target_time >= PATROL_IDLE_TIMEOUT:
                self.state = "PATROL"
                self.info["state"] = "PATROL"
                self._update_direction()
                # 进入巡逻时重置看门狗
                self._watchdog_coord = None
                self._watchdog_time = now
                print(f"[PATROL] 开始巡逻，方向: {self.current_dir}")

        elif self.state == "PATROL":
            # 检查是否到达当前路线点
            self._check_route_arrival()

            # 更新 OCR 坐标变化时间
            if self.grid_nav is not None and self.grid_nav.world_x >= 0:
                cur_coord = (self.grid_nav.world_x, self.grid_nav.world_y)
                if cur_coord != self._last_coord:
                    self._last_coord = cur_coord
                    self._last_coord_time = now
                    # 坐标变化说明角色在正常移动，重置撞墙计数和滑行计数
                    self.consecutive_stuck = 0
                    self._corner_slide_count = 0
                    self._slide_steps_left = 0

            # 全局进度看门狗：独立计时，不被单次卡死处理重置
            # 专门应对"困在死角"场景
            if self.grid_nav is not None and self.grid_nav.world_x >= 0:
                wx, wy = self.grid_nav.world_x, self.grid_nav.world_y
                if self._watchdog_coord is None:
                    self._watchdog_coord = (wx, wy)
                    self._watchdog_time = now
                elif now - self._watchdog_time >= self._watchdog_timeout:
                    ox, oy = self._watchdog_coord
                    total_moved = abs(wx - ox) + abs(wy - oy)
                    if total_moved < self._watchdog_min:
                        print(f"[PATROL] ★ {self._watchdog_timeout:.0f}秒位移仅{total_moved}格(困在死角)，全局重定位")
                        new_idx = self._find_nearest_route_index(wx, wy, full_search=True)
                        # 跳过当前附近点，往前推30个
                        new_idx = min(new_idx + 30, len(self.route) - 1)
                        self.route_index = new_idx
                        self.consecutive_stuck = 0
                        self._update_direction()
                        print(f"[ROUTE] 全局重定位 → #{self.route_index}{self.route[self.route_index]}")
                    # 每次到期都更新
                    self._watchdog_coord = (wx, wy)
                    self._watchdog_time = now

            # OCR坐标N秒不变 → 触发贴墙滑行（不跳点）
            coord_stuck = (
                self.grid_nav is not None
                and self.grid_nav.world_x >= 0
                and self._slide_steps_left == 0
                and now - self._last_coord_time > self._coord_stuck_timeout
            )
            if coord_stuck and self.route_mode and self.route_index < len(self.route):
                wx, wy = self.grid_nav.world_x, self.grid_nav.world_y
                rx, ry = self.route[self.route_index]
                if self._corner_slide_count < 4:
                    self._do_wall_slide(game_hwnd)
                    self._corner_slide_count += 1
                    self._last_coord_time = now  # 给滑行留出时间
                    return

            if self._check_stuck(frame):
                # 距目标较近 → 优先尝试贴墙滑行
                if (self.route_mode
                        and self.grid_nav is not None
                        and self.grid_nav.world_x >= 0
                        and self.route_index < len(self.route)):
                    wx, wy = self.grid_nav.world_x, self.grid_nav.world_y
                    rx, ry = self.route[self.route_index]
                    world_dist = abs(wx - rx) + abs(wy - ry)
                    if world_dist <= 25 and self._corner_slide_count < 4:
                        self._do_wall_slide(game_hwnd)
                        self._corner_slide_count += 1
                        return

                # 贴墙滑行用尽 / 距离远 → 盲点处理
                self._corner_slide_count = 0
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

        # 路线断点检测：动态阈值，撞墙越多越灵敏
        # consecutive_stuck=0 → 50, =1 → 30, >=2 → 15
        if self.consecutive_stuck == 0:
            _thresh = 50
        elif self.consecutive_stuck == 1:
            _thresh = 30
        else:
            _thresh = 15

        if dist > _thresh:
            # 防抖：连续3帧距离异常才触发重定位，防止OCR单帧误读（如37→97）导致级联失败
            self._disconnect_count += 1
            if self._disconnect_count >= 3:
                new_idx = self._find_nearest_route_index(wx, wy)
                if new_idx != self.route_index:
                    print(f"[ROUTE] 路线断点(dist={dist})，重定位 #{self.route_index}({rx},{ry})"
                          f" → #{new_idx}{self.route[new_idx]}")
                    self.route_index = new_idx
                    self._update_direction()
                self._disconnect_count = 0
            return
        else:
            self._disconnect_count = 0  # 距离正常，重置防抖计数

        # 到达判定：Manhattan ≤ 8，或 Chebyshev ≤ 1（x y 各 ±1 范围内即算到达）
        arrived = dist <= 8 or (abs(wx - rx) <= 1 and abs(wy - ry) <= 1)

        # 越过检测：如果下一个路线点比当前目标更近，说明已经冲过了
        if not arrived and self.route_index + 1 < len(self.route):
            nx, ny = self.route[self.route_index + 1]
            if abs(wx - nx) + abs(wy - ny) < dist:
                arrived = True

        if arrived:
            self.route_index += 1
            self._corner_slide_count = 0
            self._slide_steps_left = 0
            self._update_direction()
            if self.route_index % 50 == 0:
                progress = self.route_index / len(self.route) * 100
                print(f"[ROUTE] 进度: {self.route_index}/{len(self.route)} ({progress:.0f}%)")

    def _do_wall_slide(self, game_hwnd):
        """贴墙滑行：设定垂直方向滑行步数，由 _do_move 持续执行。
        步数 = max(至少2秒对应步数, world_dist/4)，奇偶次交替两侧方向。
        """
        if self.grid_nav is None or self.grid_nav.world_x < 0:
            return
        if self.route_index >= len(self.route):
            return

        wx, wy = self.grid_nav.world_x, self.grid_nav.world_y
        rx, ry = self.route[self.route_index]
        ddx = rx - wx
        ddy = ry - wy
        world_dist = abs(ddx) + abs(ddy)
        euclid = math.sqrt(ddx * ddx + ddy * ddy)
        if euclid == 0:
            return

        # 精确方向向量（目标方向）
        ndx = ddx / euclid
        ndy = ddy / euclid

        # 原移动方向（current_dir 8方向，与精确目标方向略有偏差）
        odx, ody = DIRECTIONS.get(self.current_dir, (0, 0))
        odist = math.sqrt(odx * odx + ody * ody)
        if odist > 0:
            v_orig_x = odx / odist
            v_orig_y = ody / odist
        else:
            v_orig_x, v_orig_y = ndx, ndy

        # 叉积判断目标方向相对原方向的偏转侧
        # cross < 0 → 目标在原方向的逆时针侧（屏幕坐标 y 向下）
        # cross > 0 → 目标在原方向的顺时针侧
        cross = v_orig_x * ndy - v_orig_y * ndx

        # 主方向：选让"原方向→滑行方向"扇形包含目标方向的那侧垂直方向
        if cross <= 0:
            # 目标偏逆时针 → 滑行选顺时针垂直方向
            primary_perp   = (ndy, -ndx)   # 顺时针 90°
            secondary_perp = (-ndy, ndx)   # 逆时针 90°（备用）
        else:
            # 目标偏顺时针 → 滑行选逆时针垂直方向
            primary_perp   = (-ndy, ndx)   # 逆时针 90°
            secondary_perp = (ndy, -ndx)   # 顺时针 90°（备用）

        if self._corner_slide_count % 2 == 0:
            perp_x, perp_y = primary_perp
            side = "主方向"
        else:
            perp_x, perp_y = secondary_perp
            side = "备用方向"

        # 滑行步数：至少2秒，或 world_dist/4 步，取较大值
        min_steps = math.ceil(2.0 / PATROL_MOVE_INTERVAL)   # 至少 ceil(2/0.6)=4 步
        dist_steps = max(1, round(world_dist / 4))
        slide_steps = max(min_steps, dist_steps)

        self._slide_steps_left = slide_steps
        self._slide_perp = (perp_x, perp_y)
        print(f"[PATROL] 贴墙滑行启动({self._corner_slide_count + 1}/4) [{side}]"
              f" cross={cross:.2f} dist={world_dist}"
              f" → {slide_steps}步({slide_steps * PATROL_MOVE_INTERVAL:.1f}s)"
              f" 目标#{self.route_index}({rx},{ry})")

    def _update_direction(self):
        """根据路线更新当前方向"""
        if self.route_mode and self.grid_nav and self.grid_nav.world_x >= 0:
            direction = self._get_route_direction()
            if direction:
                self.current_dir = direction
                self.info["direction"] = self.current_dir
                self.info["route_index"] = self.route_index
                # 目标点信息（供可视化）
                if self.route_index < len(self.route):
                    rx, ry = self.route[self.route_index]
                    wx, wy = self.grid_nav.world_x, self.grid_nav.world_y
                    self.info["route_target"] = (rx, ry)
                    self.info["route_dist"] = abs(rx - wx) + abs(ry - wy)
                return

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
        """发送移动指令 — 精确角度朝路线点右键点击"""

        # 贴墙滑行中：优先朝垂直方向走，走完再恢复原方向
        if self._slide_steps_left > 0:
            perp_x, perp_y = self._slide_perp
            click_x = int(SELF_CX + perp_x * PATROL_CLICK_DISTANCE)
            click_y = int(SELF_CY + perp_y * PATROL_CLICK_DISTANCE)
            h, w = frame.shape[:2]
            click_x = max(10, min(w - 10, click_x))
            click_y = max(10, min(h - 10, click_y))
            self._slide_steps_left -= 1
            try:
                lparam = _make_lparam(click_x, click_y)
                PostMessage(game_hwnd, WM_RBUTTONDOWN, MK_RBUTTON, lparam)
                PostMessage(game_hwnd, WM_RBUTTONUP, 0, lparam)
                self._held_hwnd = game_hwnd
            except Exception as e:
                print(f"[PATROL] 滑行移动失败: {e}")
                return
            self.move_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.move_frame_time = time.time()
            self.last_move_time = time.time()
            self.info["click_pos"] = (click_x, click_y)
            remaining_s = self._slide_steps_left * PATROL_MOVE_INTERVAL
            print(f"[PATROL] 滑行中 剩余{self._slide_steps_left}步({remaining_s:.1f}s)"
                  f" → click({click_x},{click_y})")
            return

        # 更新方向（供显示用）
        self._update_direction()

        # 精确角度点击：用世界坐标差值算真实方向向量
        mode = "ROUTE"
        if (self.route_mode
                and self.grid_nav is not None
                and self.grid_nav.world_x >= 0
                and self.route_index < len(self.route)):
            wx, wy = self.grid_nav.world_x, self.grid_nav.world_y
            rx, ry = self.route[self.route_index]
            ddx = rx - wx
            ddy = ry - wy
            dist = math.sqrt(ddx * ddx + ddy * ddy)
            if dist > 0:
                ndx = ddx / dist
                ndy = ddy / dist
                click_x = int(SELF_CX + ndx * PATROL_CLICK_DISTANCE)
                click_y = int(SELF_CY + ndy * PATROL_CLICK_DISTANCE)
            else:
                # 已在目标点上，用当前方向
                dx, dy = DIRECTIONS[self.current_dir]
                click_x = SELF_CX + int(dx * PATROL_CLICK_DISTANCE)
                click_y = SELF_CY + int(dy * PATROL_CLICK_DISTANCE)
        else:
            # fallback：8方向
            mode = "PATROL"
            dx, dy = DIRECTIONS[self.current_dir]
            click_x = SELF_CX + int(dx * PATROL_CLICK_DISTANCE)
            click_y = SELF_CY + int(dy * PATROL_CLICK_DISTANCE)

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
        """撞墙处理：盲点3下"""
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

        # 通知网格导航
        if self.grid_nav is not None:
            self.grid_nav.on_stuck(stuck_dir)

        self._update_direction()
        self.move_frame = None

