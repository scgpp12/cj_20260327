"""
action_controller.py - 攻击控制器 (v5 清理)

状态机：
  IDLE    - 无目标，等待发现怪物
  BURST   - 快速连击锁敌（2下，间隔0.1s）
  WAITING - 等待，持续跟踪目标，2秒后重新连击

核心规则：
  - targets 为空超过0.5s → 回 IDLE
  - 锁定远怪时身边出现近怪 → 立刻切换（比例判断）
  - 近距离(≤60px)用方位点攻击，远距离直接点怪物
  - 3轮无效攻击 → 放弃目标（冷却3秒）
"""

import time
import ctypes
import ctypes.wintypes
import math

# Windows 消息常量
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
MK_LBUTTON = 0x0001

user32 = ctypes.windll.user32
PostMessage = user32.PostMessageW
EnumWindows = user32.EnumWindows
GetWindowText = user32.GetWindowTextW
GetWindowTextLength = user32.GetWindowTextLengthW
IsWindowVisible = user32.IsWindowVisible
GetWindowRect = user32.GetWindowRect


def _make_lparam(x, y):
    x, y = int(x), int(y)
    return (y << 16) | (x & 0xFFFF)


def _box_center(box):
    return box[0] + box[2] // 2, box[1] + box[3] // 2


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


class ActionController:
    """攻击控制器: IDLE → BURST → WAITING → IDLE/BURST"""

    STATE_IDLE = "IDLE"
    STATE_BURST = "BURST"
    STATE_WAITING = "WAITING"

    def __init__(self, game_hwnd=None):
        self.game_hwnd = game_hwnd
        self.enabled = False

        # 状态
        self.state = self.STATE_IDLE
        self.locked_target = None
        self.lock_count = 0

        # 连击参数
        self.BURST_CLICKS = 2
        self.BURST_INTERVAL = 0.10
        self._burst_done = 0
        self._burst_last_time = 0

        # 等待参数
        self._wait_start_time = 0
        self.WAIT_RECHECK_INTERVAL = 2.0
        self.WAIT_ABSOLUTE_MAX = 10.0

        # 目标消失确认
        self._target_gone_time = 0
        self.TARGET_GONE_CONFIRM = 0.5

        # 无效攻击检测
        self._ineffective_rounds = 0
        self.MAX_INEFFECTIVE_ROUNDS = 3
        self._giveup_time = 0
        self.GIVEUP_COOLDOWN = 3.0

        # 巡逻控制器引用
        self._patrol_ref = None

        # 角色中心
        from config import SELF_CENTER_X, SELF_CENTER_Y
        self.self_cx = SELF_CENTER_X - 5
        self.self_cy = SELF_CENTER_Y

        # 9 方位攻击点（8方向 + 中心）
        self.ATK_RADIUS_STRAIGHT = 40
        self.ATK_RADIUS_DIAGONAL = 50
        self.atk_origin_x = self.self_cx - 5
        self.atk_origin_y = self.self_cy - 50 + 10
        self.atk_directions = []
        dir_names = ["UP", "UP_RIGHT", "RIGHT", "DOWN_RIGHT",
                     "DOWN", "DOWN_LEFT", "LEFT", "UP_LEFT"]
        for i, name in enumerate(dir_names):
            angle = math.radians(90 - i * 45)
            is_diagonal = i % 2 == 1
            radius = self.ATK_RADIUS_DIAGONAL if is_diagonal else self.ATK_RADIUS_STRAIGHT
            dx = int(radius * math.cos(angle))
            dy = int(-radius * math.sin(angle))
            self.atk_directions.append((name, self.atk_origin_x + dx, self.atk_origin_y + dy))
        self.atk_directions.append(("CENTER", self.atk_origin_x, self.atk_origin_y))

        # 点击可视化
        self.last_click_pos = None
        self.last_click_time = 0
        self.last_atk_dir = None
        self.CLICK_SHOW_DURATION = 0.5

        # 远距离位置追踪
        self._prev_target_pos = None
        self.TARGET_POS_CHANGE_THRESH = 15
        self._last_reclick_time = 0

        # 近身优先：3帧投票，只要1帧绿框内有怪就优先攻击
        self._near_history = [False, False, False]  # 最近3帧是否有近怪
        self._near_frame_idx = 0

    def set_hwnd(self, hwnd):
        self.game_hwnd = hwnd

    def set_audio_state(self, is_attacking):
        pass  # 保留接口兼容

    def update(self, targets):
        """每帧调用。targets: [(x,y,w,h), ...]，按距离排序"""
        now = time.time()

        if not self.enabled or self.game_hwnd is None:
            return self._make_info()

        # ===== targets 为空：等0.5秒确认 =====
        if not targets:
            if self.state == self.STATE_IDLE:
                return self._make_info()
            if self._target_gone_time == 0:
                self._target_gone_time = now
            elif now - self._target_gone_time >= self.TARGET_GONE_CONFIRM:
                print(f"[ATK] 目标消失{self.TARGET_GONE_CONFIRM}s → 回到空闲 (攻击了{self.lock_count}次)")
                self._reset_to_idle()
            return self._make_info()

        self._target_gone_time = 0

        # ===== 近身优先投票（3帧中1帧有近怪就优先）=====
        near_targets = [t for t in targets if self._dist_to_self(t) <= 100]
        self._near_history[self._near_frame_idx % 3] = len(near_targets) > 0
        self._near_frame_idx += 1

        # 3帧中有任意1帧检测到近怪 → 只用近怪列表
        if any(self._near_history) and near_targets:
            targets = near_targets

        # ===== 状态机 =====
        if self.state == self.STATE_IDLE:
            # 冷却期间只攻击近怪
            if now - self._giveup_time < self.GIVEUP_COOLDOWN:
                near = [t for t in targets if self._dist_to_self(t) <= 100]
                if not near:
                    return self._make_info()
                targets = near

            # 释放巡逻右键
            if self._patrol_ref is not None:
                self._patrol_ref._release_rbutton()

            self.locked_target = targets[0]
            self._prev_target_pos = None
            self.lock_count = 0
            self._burst_done = 0
            self._ineffective_rounds = 0
            self.state = self.STATE_BURST
            print(f"[ATK] 发现目标 dist={self._dist_to_self(self.locked_target):.0f} → 连击锁敌")

        elif self.state == self.STATE_BURST:
            self._check_switch_closer(targets)
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
            self._check_switch_closer(targets)
            self._update_locked_target(targets)

            wait_elapsed = now - self._wait_start_time

            if wait_elapsed >= self.WAIT_ABSOLUTE_MAX:
                self._burst_done = 0
                self.state = self.STATE_BURST
                return self._make_info()

            # 持续补点
            self._keep_clicking(targets, now)

            # 2秒后重新连击
            if wait_elapsed >= self.WAIT_RECHECK_INTERVAL:
                self._ineffective_rounds += 1
                if self._ineffective_rounds >= self.MAX_INEFFECTIVE_ROUNDS:
                    locked_dist = self._dist_to_self(self.locked_target) if self.locked_target else 0
                    print(f"[ATK] {self._ineffective_rounds}轮无效攻击(dist={locked_dist:.0f}) → 放弃目标(冷却{self.GIVEUP_COOLDOWN:.0f}s)")
                    self._ineffective_rounds = 0
                    self._giveup_time = now
                    self.locked_target = None
                    self.state = self.STATE_IDLE
                else:
                    print(f"[ATK] {wait_elapsed:.0f}秒等待 → 重新连击({self._ineffective_rounds}/{self.MAX_INEFFECTIVE_ROUNDS})")
                    self._burst_done = 0
                    self.state = self.STATE_BURST

        return self._make_info()

    # ===== 内部方法 =====

    def _check_switch_closer(self, targets):
        """锁定远怪时，出现明显更近的怪 → 切换（比例判断）"""
        if not targets or self.locked_target is None:
            return
        locked_dist = self._dist_to_self(self.locked_target)
        nearest_dist = self._dist_to_self(targets[0])

        if locked_dist > 100 and nearest_dist < locked_dist * 0.5:
            print(f"[ATK] 出现更近的怪 {locked_dist:.0f}→{nearest_dist:.0f}px → 切换目标")
            self.locked_target = targets[0]
            self._prev_target_pos = None
            self._burst_done = 0
            self._ineffective_rounds = 0
            self.state = self.STATE_BURST

    def _update_locked_target(self, targets):
        """跟踪同一只怪的新位置，找不到则切最近的"""
        if not targets or self.locked_target is None:
            return

        same = self._find_same_target(targets)
        if same is not None:
            self.locked_target = same
        else:
            old_dist = self._dist_to_self(self.locked_target)
            self.locked_target = targets[0]
            self._prev_target_pos = None
            new_dist = self._dist_to_self(self.locked_target)
            print(f"[ATK] 目标丢失，切换到最近怪 {old_dist:.0f}→{new_dist:.0f}px")

    def _keep_clicking(self, targets, now):
        """WAITING 期间持续补点"""
        if not targets or self.locked_target is None:
            return

        locked_dist = self._dist_to_self(self.locked_target)

        if locked_dist <= 60:
            # 近距离：每1秒补点
            if now - self._last_reclick_time >= 1.0:
                self._click_target()
                self._last_reclick_time = now
                self.lock_count += 1
        else:
            # 远距离：位置变化时立刻点，否则每1秒补点
            new_cx, new_cy = _box_center(self.locked_target)
            need_click = False

            if self._prev_target_pos is not None:
                dx = new_cx - self._prev_target_pos[0]
                dy = new_cy - self._prev_target_pos[1]
                if (dx ** 2 + dy ** 2) ** 0.5 >= self.TARGET_POS_CHANGE_THRESH:
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
        """在当前帧找和锁定目标最接近的怪（60px内匹配）"""
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
        # 多个怪在60px内 → 保持不动防跳动
        return self.locked_target

    def _reset_to_idle(self):
        """清除所有状态，回到 IDLE"""
        self.state = self.STATE_IDLE
        self.locked_target = None
        self.lock_count = 0
        self._prev_target_pos = None
        self._burst_done = 0
        self._target_gone_time = 0

    def _make_info(self):
        return {
            "state": self.state,
            "locked": self.locked_target is not None,
            "target": self.locked_target,
            "count": self.lock_count,
            "moving": False,
        }

    def _dist_to_self(self, box):
        cx, cy = _box_center(box)
        return ((cx - self.self_cx) ** 2 + (cy - self.self_cy) ** 2) ** 0.5

    def _nearest_atk_index(self, target_box):
        """找到离目标最近的方位索引(0-7)"""
        tcx, tcy = _box_center(target_box)
        best_idx = 0
        best_dist = 999
        for i in range(8):
            _, ax, ay = self.atk_directions[i]
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

        if locked_dist <= 60:
            atk_idx = self._nearest_atk_index(self.locked_target)
            atk_name, ax, ay = self.atk_directions[atk_idx]
            self.last_atk_dir = atk_name
        else:
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

    def on_target_lost(self):
        """外部通知：所有目标消失"""
        if self.state != self.STATE_IDLE:
            self._reset_to_idle()
