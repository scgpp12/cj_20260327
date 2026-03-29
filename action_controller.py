"""
action_controller.py - 攻击控制器 (v3 重构)

状态机：
  IDLE    - 无目标，等待发现怪物
  BURST   - 快速连击锁敌（2下，间隔0.1s）
  WAITING - 等待自动攻击，持续跟踪目标

核心规则：
  - targets 为空 → 立即回 IDLE（不再等1秒确认）
  - 近距离(≤60px) → 锁死目标不跳，每1秒补点
  - 远距离(>60px) → 追踪位置变化，位移≥15px重新点击
"""

import time
import ctypes
import ctypes.wintypes

# Windows 消息常量
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
MK_LBUTTON = 0x0001
MK_RBUTTON = 0x0002

user32 = ctypes.windll.user32
PostMessage = user32.PostMessageW
FindWindow = user32.FindWindowW
EnumWindows = user32.EnumWindows
GetWindowText = user32.GetWindowTextW
GetWindowTextLength = user32.GetWindowTextLengthW
IsWindowVisible = user32.IsWindowVisible
GetWindowRect = user32.GetWindowRect


def _make_lparam(x, y):
    x, y = int(x), int(y)
    return (y << 16) | (x & 0xFFFF)


def find_game_hwnd(left_hint=3000):
    """查找游戏窗口句柄"""
    result = []

    def callback(hwnd, _):
        if not IsWindowVisible(hwnd):
            return True
        rect = ctypes.wintypes.RECT()
        GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if rect.left >= left_hint and w > 800 and h > 600:
            length = GetWindowTextLength(hwnd)
            title = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, title, length + 1)
            result.append((hwnd, rect.left, rect.top, w, h, title.value))
        return True

    EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND,
                                    ctypes.wintypes.LPARAM)(callback), 0)
    if not result:
        print("[WARNING] 未找到游戏窗口句柄")
        return None

    result.sort(key=lambda x: x[3] * x[4], reverse=True)
    hwnd, left, top, w, h, title = result[0]
    print(f"[INFO] 找到游戏窗口: hwnd={hwnd} pos=({left},{top}) size={w}x{h}")
    return hwnd


def _box_center(box):
    return box[0] + box[2] // 2, box[1] + box[3] // 2


class ActionController:
    """
    攻击控制器

    状态机:
        IDLE → BURST → WAITING → IDLE(怪死了) 或 → BURST(重新连击)
    """

    STATE_IDLE = "IDLE"
    STATE_BURST = "BURST"
    STATE_WAITING = "WAITING"

    def __init__(self, click_cooldown=1.0, game_hwnd=None):
        self.game_hwnd = game_hwnd
        self.enabled = False

        # 状态
        self.state = self.STATE_IDLE
        self.locked_target = None     # 当前锁定的目标 (x,y,w,h)
        self.lock_count = 0           # 总点击次数

        # 连击参数
        self.BURST_CLICKS = 2         # 连击次数
        self.BURST_INTERVAL = 0.10    # 连击间隔（秒）
        self._burst_done = 0
        self._burst_last_time = 0

        # 等待参数
        self._wait_start_time = 0
        self.WAIT_RECHECK_INTERVAL = 2.0  # 多久无声音重新连击
        self.WAIT_ABSOLUTE_MAX = 10.0     # 绝对超时

        # 目标消失确认：短暂丢失不立刻放弃，但也不点击
        self._target_gone_time = 0
        self.TARGET_GONE_CONFIRM = 0.5    # 消失0.5秒才确认死亡

        # 声音状态（由外部设置）
        self.is_attacking_audio = False

        # 巡逻控制器引用（攻击前释放右键）
        self._patrol_ref = None

        # 距离参数
        from config import SELF_CENTER_X, SELF_CENTER_Y
        self.self_cx = SELF_CENTER_X - 5
        self.self_cy = SELF_CENTER_Y

        # 9 个方位攻击点（60px）：8方向 + 中心
        # 攻击圆心 = 角色中心上移50px（角色模型头部位置）
        import math
        self.ATK_RADIUS_STRAIGHT = 40   # 上下左右
        self.ATK_RADIUS_DIAGONAL = 50   # 斜方向
        self.atk_origin_x = self.self_cx - 5
        self.atk_origin_y = self.self_cy - 50 + 10
        self.atk_directions = []  # [(name, x, y), ...]
        dir_names = ["UP", "UP_RIGHT", "RIGHT", "DOWN_RIGHT",
                     "DOWN", "DOWN_LEFT", "LEFT", "UP_LEFT"]
        for i, name in enumerate(dir_names):
            angle = math.radians(90 - i * 45)  # UP=90°, 顺时针递减
            is_diagonal = i % 2 == 1  # 奇数索引 = 斜方向
            radius = self.ATK_RADIUS_DIAGONAL if is_diagonal else self.ATK_RADIUS_STRAIGHT
            dx = int(radius * math.cos(angle))
            dy = int(-radius * math.sin(angle))  # y轴向下
            self.atk_directions.append((name, self.atk_origin_x + dx, self.atk_origin_y + dy))
        # 中心点
        self.atk_directions.append(("CENTER", self.atk_origin_x, self.atk_origin_y))

        # 点击可视化
        self.last_click_pos = None
        self.last_click_time = 0
        self.last_atk_dir = None      # 上次攻击方位名
        self.CLICK_SHOW_DURATION = 0.5

        # 远距离位置追踪
        self._prev_target_pos = None
        self.TARGET_POS_CHANGE_THRESH = 15
        self._last_reclick_time = 0

        # 方位重试：点击一个方位没打到，试相邻方位
        # 索引: 0=UP, 1=UP_RIGHT, 2=RIGHT, 3=DOWN_RIGHT, 4=DOWN, 5=DOWN_LEFT, 6=LEFT, 7=UP_LEFT
        self._atk_dir_index = None       # 当前基准方位索引
        self._atk_try_step = 0           # 0=原方位, 1=邻居1, 2=邻居2
        # 每个方位的重试邻居表：斜方向→两个直方向，直方向→两侧斜方向
        self._atk_neighbors = {
            0: [1, 7],  # UP       → UP_RIGHT, UP_LEFT
            1: [0, 2],  # UP_RIGHT → UP, RIGHT
            2: [1, 3],  # RIGHT    → UP_RIGHT, DOWN_RIGHT
            3: [4, 2],  # DOWN_RIGHT → DOWN, RIGHT
            4: [3, 5],  # DOWN     → DOWN_RIGHT, DOWN_LEFT
            5: [4, 6],  # DOWN_LEFT → DOWN, LEFT
            6: [5, 7],  # LEFT     → DOWN_LEFT, UP_LEFT
            7: [0, 6],  # UP_LEFT  → UP, LEFT
        }

    def set_hwnd(self, hwnd):
        self.game_hwnd = hwnd

    def set_audio_state(self, is_attacking):
        self.is_attacking_audio = is_attacking

    def set_visual_state(self, is_moving):
        pass  # 保留接口兼容，不再使用

    def update(self, targets):
        """
        每帧调用。

        Args:
            targets: 范围内的怪物列表 [(x,y,w,h), ...]，按距离排序（最近在前）

        Returns:
            dict: 状态信息
        """
        now = time.time()

        if not self.enabled or self.game_hwnd is None:
            return self._make_info()

        # ===== targets 为空处理：等0.5秒确认，期间不点击 =====
        if not targets:
            if self.state == self.STATE_IDLE:
                return self._make_info()
            # 正在攻击中目标消失 → 开始计时
            if self._target_gone_time == 0:
                self._target_gone_time = now
            elif now - self._target_gone_time >= self.TARGET_GONE_CONFIRM:
                # 消失超过0.5秒 → 确认目标死亡
                print(f"[ATK] 目标消失{self.TARGET_GONE_CONFIRM}s → 回到空闲 (攻击了{self.lock_count}次)")
                self._reset_to_idle()
            # 消失期间不做任何点击，等目标重新出现
            return self._make_info()

        # targets 非空 → 重置消失计时
        self._target_gone_time = 0

        # ===== 状态机（以下保证 targets 非空）=====

        if self.state == self.STATE_IDLE:
            # 释放巡逻右键
            if self._patrol_ref is not None:
                self._patrol_ref._release_rbutton()

            self.locked_target = targets[0]
            self._prev_target_pos = None
            self.lock_count = 0
            self._burst_done = 0
            self.state = self.STATE_BURST
            dist = self._dist_to_self(self.locked_target)
            print(f"[ATK] 发现目标 dist={dist:.0f} → 连击锁敌")

        elif self.state == self.STATE_BURST:
            # 更新目标位置
            self._update_locked_target(targets)

            if self._burst_done >= self.BURST_CLICKS:
                self.state = self.STATE_WAITING
                self._wait_start_time = now
                self._last_reclick_time = now
                print(f"[ATK] 连击完成({self.BURST_CLICKS}下) → 等待自动攻击")
            elif now - self._burst_last_time >= self.BURST_INTERVAL:
                self._click_target()
                self._burst_done += 1
                self._burst_last_time = now
                self.lock_count += 1

        elif self.state == self.STATE_WAITING:
            # 更新目标位置
            self._update_locked_target(targets)

            wait_elapsed = now - self._wait_start_time

            # 绝对超时 10s
            if wait_elapsed >= self.WAIT_ABSOLUTE_MAX:
                print(f"[ATK] 10秒超时 → 重新连击")
                self._burst_done = 0
                self.state = self.STATE_BURST
                return self._make_info()

            # 持续点击：近距离每1秒补点，远距离位置变化时点
            self._keep_clicking(targets, now)

            # 有声音 → 继续等
            if self.is_attacking_audio and wait_elapsed < 5.0:
                pass
            elif wait_elapsed >= self.WAIT_RECHECK_INTERVAL:
                print(f"[ATK] {wait_elapsed:.0f}秒无声音 → 重新连击")
                self._burst_done = 0
                self.state = self.STATE_BURST

        return self._make_info()

    # ===== 内部方法 =====

    def _update_locked_target(self, targets):
        """更新锁定目标位置：始终跟踪同一只怪的新位置，不切换到别的怪"""
        if not targets or self.locked_target is None:
            return

        # 找和锁定目标最接近的怪（同一只怪的新位置）
        same = self._find_same_target(targets)
        if same is not None:
            self.locked_target = same
        else:
            # 远距离找不到同一只 → 锁定最近的
            locked_dist = self._dist_to_self(self.locked_target)
            if locked_dist > 60:
                self.locked_target = targets[0]
                self._prev_target_pos = None

    def _keep_clicking(self, targets, now):
        """WAITING 期间持续点击"""
        if not targets or self.locked_target is None:
            return

        locked_dist = self._dist_to_self(self.locked_target)

        if locked_dist <= 60:
            # 近距离：每1秒补点一次
            if now - self._last_reclick_time >= 1.0:
                self._click_target()
                self._last_reclick_time = now
                self.lock_count += 1
        else:
            # 远距离（直接点怪物）：位置变了立刻点，没变每1秒补点
            new_cx, new_cy = _box_center(self.locked_target)
            need_click = False

            if self._prev_target_pos is not None:
                dx = new_cx - self._prev_target_pos[0]
                dy = new_cy - self._prev_target_pos[1]
                pos_delta = (dx ** 2 + dy ** 2) ** 0.5
                if pos_delta >= self.TARGET_POS_CHANGE_THRESH:
                    need_click = True
            else:
                self._prev_target_pos = (new_cx, new_cy)

            if need_click and now - self._last_reclick_time >= 0.3:
                self._click_target()
                self._last_reclick_time = now
                self._prev_target_pos = (new_cx, new_cy)
                self.lock_count += 1
            elif now - self._last_reclick_time >= 1.0:
                self._click_target()
                self._last_reclick_time = now
                self._prev_target_pos = (new_cx, new_cy)
                self.lock_count += 1

    def _find_same_target(self, targets):
        """在当前帧找和锁定目标最接近的怪（60px内匹配）
        多个怪在60px内 → 不更新，防跳动
        """
        if not targets or self.locked_target is None:
            return None
        lcx, lcy = _box_center(self.locked_target)

        candidates = []
        for t in targets:
            cx, cy = _box_center(t)
            d = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
            if d < 60:
                candidates.append((d, t))

        if len(candidates) == 0:
            return None
        if len(candidates) == 1:
            return candidates[0][1]
        # 多个怪在60px内 → 不更新
        return self.locked_target

    def _reset_to_idle(self):
        """清除所有状态，回到 IDLE"""
        self.state = self.STATE_IDLE
        self.locked_target = None
        self.lock_count = 0
        self._prev_target_pos = None
        self._burst_done = 0
        self._target_gone_time = 0
        self._atk_dir_index = None
        self._atk_try_step = 0

    def _make_info(self):
        return {
            "state": self.state,
            "locked": self.locked_target is not None,
            "target": self.locked_target,
            "count": self.lock_count,
            "audio": self.is_attacking_audio,
            "moving": False,
        }

    def _dist_to_self(self, box):
        cx, cy = _box_center(box)
        return ((cx - self.self_cx) ** 2 + (cy - self.self_cy) ** 2) ** 0.5

    def _nearest_atk_index(self, target_box):
        """找到离目标最近的方位索引(0-7，不含CENTER)"""
        tcx, tcy = _box_center(target_box)
        best_idx = 0
        best_dist = 999
        for i in range(8):  # 只看8个方向，不含CENTER
            name, ax, ay = self.atk_directions[i]
            d = ((tcx - ax) ** 2 + (tcy - ay) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _click_target(self):
        """左键点击目标：近距离用方位点，远距离直接点怪物"""
        if self.locked_target is None or self.game_hwnd is None:
            return

        locked_dist = self._dist_to_self(self.locked_target)

        if locked_dist <= 100:
            # === 近距离：用9方位点攻击 ===
            atk_idx = self._nearest_atk_index(self.locked_target)
            atk_name, ax, ay = self.atk_directions[atk_idx]
            self.last_atk_dir = atk_name
        else:
            # === 远距离：直接点怪物位置 ===
            ax, ay = _box_center(self.locked_target)
            self.last_atk_dir = f"DIRECT d={locked_dist:.0f}"

        try:
            lparam = _make_lparam(ax, ay)
            PostMessage(self.game_hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
            PostMessage(self.game_hwnd, WM_LBUTTONUP, 0, lparam)
            self.last_click_pos = (int(ax), int(ay))
            self.last_click_time = time.time()
        except Exception as e:
            print(f"[ATK] 点击失败: {e}")

    def _try_next_direction(self):
        """当前方位没打到 → 切换到下一个邻居"""
        if self._atk_dir_index is None:
            return

        old_step = self._atk_try_step
        neighbors = self._atk_neighbors.get(self._atk_dir_index, [])

        if self._atk_try_step < len(neighbors):
            self._atk_try_step += 1
        else:
            # 邻居都试过了 → 回到原方位重新来
            self._atk_try_step = 0

        # 打印切换信息
        base_name = self.atk_directions[self._atk_dir_index][0]
        if self._atk_try_step == 0:
            print(f"[ATK] 邻居都试过 → 回到 {base_name}")
        else:
            next_idx = neighbors[self._atk_try_step - 1]
            next_name = self.atk_directions[next_idx][0]
            print(f"[ATK] {base_name}没打到 → 试 {next_name}")

    def on_target_lost(self):
        """外部通知：所有目标消失"""
        if self.state != self.STATE_IDLE:
            print(f"[ATK] 外部通知目标消失 → 回到空闲")
            self._reset_to_idle()

    def clear_false_targets(self):
        """保留接口兼容"""
        pass
