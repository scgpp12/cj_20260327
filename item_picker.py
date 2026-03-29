"""
item_picker.py - 地面物品拾取模块
使用 HSV 紫色检测找到掉落物品（紫色球体）
"""

import time
import cv2
import numpy as np
import ctypes

from config import SELF_CENTER_X, SELF_CENTER_Y

# PostMessage 常量
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


class ItemPicker:
    """
    地面物品拾取器（紫色球体 HSV 检测版）

    状态机:
        IDLE        - 空闲，等待触发
        WALKING     - 走向物品中
        PICKING     - 到达物品位置，点击拾取
        WAIT_PICK   - 等待物品消失确认
    """

    STATE_IDLE = "IDLE"
    STATE_WALKING = "WALKING"
    STATE_PICKING = "PICKING"
    STATE_WAITING = "WAIT_PICK"

    def __init__(self, pick_range=350, arrive_dist=40, pick_timeout=2.0):
        self.enabled = True
        self.pick_range = pick_range
        self.arrive_dist = arrive_dist
        self.pick_timeout = pick_timeout

        self.state = self.STATE_IDLE
        self.target_pos = None
        self.walk_start_time = 0
        self.pick_start_time = 0
        self.walk_timeout = 5.0

        # WALKING 状态：基于坐标变化重新点击
        self._walk_prev_item_pos = None   # 上次物品屏幕坐标
        self._walk_prev_dist = None       # 上次玩家与物品距离
        self._walk_last_click_time = 0    # 上次点击时间
        self._walk_still_since = None     # 坐标停止变化的时刻
        self.WALK_CLICK_INTERVAL = 0.3    # 最小点击间隔(秒)
        self.WALK_POS_CHANGE_THRESH = 5   # 物品坐标变化阈值(px)
        self.WALK_NUDGE_DIST = 100        # 往上走一格的像素距离

        # 跳过列表
        self._skip_list = []
        self.SKIP_DURATION = 30.0
        self.SKIP_RADIUS = 50

        # ---- 紫色检测参数 ----
        # 紫色 HSV 范围
        self.PURPLE_H_LOW = 110
        self.PURPLE_H_HIGH = 160
        self.PURPLE_S_MIN = 100
        self.PURPLE_V_MIN = 50
        self.MIN_AREA = 150         # 最小面积
        self.MAX_AREA = 5000        # 最大面积
        self.MIN_CIRCULARITY = 0.3  # 最小圆形度
        self.SELF_EXCLUDE = 30      # 角色中心排除半径

        print("[PICK] 紫色球体检测已启用")

        # 点击可视化：记录最近一次左键点击位置和时间
        self.last_click_pos = None
        self.last_click_time = 0
        self.CLICK_SHOW_DURATION = 0.5  # 红点显示时长(秒)

        self.info = {
            "state": self.STATE_IDLE,
            "items_detected": 0,
            "target": None,
        }

    def detect_items(self, frame):
        """
        用 HSV 紫色检测画面中的掉落物品

        Returns:
            items: [(x, y, w, h), ...] 物品框列表，按距离排序
        """
        h, w = frame.shape[:2]
        now = time.time()

        # ROI：角色周围 pick_range 范围
        roi_x1 = max(0, SELF_CENTER_X - self.pick_range)
        roi_y1 = max(0, SELF_CENTER_Y - self.pick_range)
        roi_x2 = min(w, SELF_CENTER_X + self.pick_range)
        roi_y2 = min(h, SELF_CENTER_Y + self.pick_range)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # HSV 紫色阈值
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           (self.PURPLE_H_LOW, self.PURPLE_S_MIN, self.PURPLE_V_MIN),
                           (self.PURPLE_H_HIGH, 255, 255))

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        items = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue

            # 圆形度
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.MIN_CIRCULARITY:
                continue

            # 边界框
            bx, by, bw, bh = cv2.boundingRect(cnt)

            # 转回全图坐标
            abs_x = bx + roi_x1
            abs_y = by + roi_y1
            cx = abs_x + bw // 2
            cy = abs_y + bh // 2

            # 角色中心排除
            if abs(cx - SELF_CENTER_X) < self.SELF_EXCLUDE and \
               abs(cy - SELF_CENTER_Y) < self.SELF_EXCLUDE:
                continue

            # 距离
            dist = ((cx - SELF_CENTER_X) ** 2 + (cy - SELF_CENTER_Y) ** 2) ** 0.5
            if dist > self.pick_range:
                continue

            # 跳过列表
            if self._is_skipped(cx, cy, now):
                continue

            items.append((abs_x, abs_y, bw, bh, dist))

        # 按距离排序
        items.sort(key=lambda x: x[4])

        # 返回 (x, y, w, h)
        result_items = [(x, y, w, h) for x, y, w, h, d in items]

        self.info["items_detected"] = len(result_items)
        return result_items

    def _nms(self, items, iou_thresh=0.3):
        """非极大值抑制，去除重叠检测"""
        if not items:
            return []

        # 按分数排序（高 → 低）
        items.sort(key=lambda x: x[4], reverse=True)
        kept = []

        for item in items:
            x1, y1, w1, h1 = item[0], item[1], item[2], item[3]
            suppress = False

            for k in kept:
                x2, y2, w2, h2 = k[0], k[1], k[2], k[3]
                # 计算中心距离（简单去重）
                d = ((x1 + w1//2 - x2 - w2//2)**2 + (y1 + h1//2 - y2 - h2//2)**2) ** 0.5
                if d < max(w1, w2):  # 中心距 < 物品宽度 → 重复
                    suppress = True
                    break

            if not suppress:
                kept.append(item)

        return kept

    def update(self, frame, game_hwnd, has_combat_target=False):
        """每帧调用，管理拾取状态机"""
        if not self.enabled or game_hwnd is None:
            return {"state": self.STATE_IDLE, "picking": False, "target": None}

        if has_combat_target:
            if self.state != self.STATE_IDLE:
                self.state = self.STATE_IDLE
                self.target_pos = None
            return {"state": self.STATE_IDLE, "picking": False, "target": None}

        now = time.time()
        self._skip_list = [(x, y, t) for x, y, t in self._skip_list if t > now]

        if self.state == self.STATE_IDLE:
            items = self.detect_items(frame)
            if items:
                bx, by, bw, bh = items[0]
                self.target_pos = (bx + bw // 2, by + bh // 2)
                self.walk_start_time = now

                dist = ((self.target_pos[0] - SELF_CENTER_X) ** 2 +
                        (self.target_pos[1] - SELF_CENTER_Y) ** 2) ** 0.5

                # 先释放右键停步（巡逻可能还在跑）
                self._release_rbutton(game_hwnd)
                self._click_item(game_hwnd, self.target_pos)
                self._walk_prev_item_pos = self.target_pos
                self._walk_prev_dist = dist
                self._walk_last_click_time = now
                self._walk_still_since = None
                self.state = self.STATE_WALKING
                print(f"[PICK] 发现物品 dist={dist:.0f} → 停步+点击走过去 ({self.target_pos[0]},{self.target_pos[1]})")

        elif self.state == self.STATE_WALKING:
            if self.target_pos is None:
                self.state = self.STATE_IDLE
                return self._result()

            items = self.detect_items(frame)
            nearest = self._find_same_item(items)

            if nearest is None:
                # 物品消失 — 根据最后距离判断是成功还是丢失
                last_dist = self._walk_prev_dist if self._walk_prev_dist else 999
                if last_dist <= 80:
                    print(f"[PICK] 物品消失(dist={last_dist:.0f}) → 拾取成功!")
                else:
                    print(f"[PICK] 物品消失(dist={last_dist:.0f}) → 距离太远，不是拾取，重新扫描")
                self.state = self.STATE_IDLE
                self.target_pos = None
                self._walk_prev_item_pos = None
                self._walk_prev_dist = None
                return self._result()

            bx, by, bw, bh = nearest
            new_pos = (bx + bw // 2, by + bh // 2)
            dist = ((new_pos[0] - SELF_CENTER_X) ** 2 +
                    (new_pos[1] - SELF_CENTER_Y) ** 2) ** 0.5

            # 打印距离（调试用）
            if not hasattr(self, '_last_dist_log') or now - self._last_dist_log > 1.0:
                print(f"[PICK] WALKING dist={dist:.0f}px target=({new_pos[0]},{new_pos[1]})")
                self._last_dist_log = now

            if now - self.walk_start_time > self.walk_timeout:
                print(f"[PICK] 走路超时(dist={dist:.0f}) → 跳过")
                self._add_skip(new_pos)
                self.state = self.STATE_IDLE
                self.target_pos = None
                self._walk_prev_item_pos = None
                self._walk_prev_dist = None
            else:
                self.target_pos = new_pos

                # 判断物品坐标有没有变化
                pos_delta = 999
                if self._walk_prev_item_pos is not None:
                    pos_delta = ((new_pos[0] - self._walk_prev_item_pos[0]) ** 2 +
                                 (new_pos[1] - self._walk_prev_item_pos[1]) ** 2) ** 0.5

                if pos_delta >= self.WALK_POS_CHANGE_THRESH:
                    # 坐标变了 = 角色在移动 → 重新点击最新位置
                    self._walk_still_since = None
                    if now - self._walk_last_click_time >= self.WALK_CLICK_INTERVAL:
                        self._click_item(game_hwnd, new_pos)
                        self._walk_last_click_time = now
                        self._walk_prev_item_pos = new_pos
                        self._walk_prev_dist = dist
                else:
                    # 坐标没变 = 角色到达/停下 → 左键点击上方走一格
                    if self._walk_still_since is None:
                        self._walk_still_since = now
                    else:
                        # 左键点击角色上方100px，走一小步
                        nudge_x = SELF_CENTER_X
                        nudge_y = SELF_CENTER_Y - self.WALK_NUDGE_DIST
                        self._click_item(game_hwnd, (nudge_x, nudge_y))
                        print(f"[PICK] 坐标不变 → 左键上方走一格({nudge_x},{nudge_y})")
                        time.sleep(0.5)
                        self._walk_prev_item_pos = None  # 清空，下帧重新检测
                        self._walk_prev_dist = dist
                        self._walk_still_since = None  # 重置

        self.info["state"] = self.state
        self.info["target"] = self.target_pos
        return self._result()

    def _result(self):
        return {
            "state": self.state,
            "picking": self.state != self.STATE_IDLE,
            "target": self.target_pos,
        }

    def _release_rbutton(self, hwnd):
        """释放右键，停止巡逻跑步"""
        try:
            PostMessage(hwnd, WM_RBUTTONUP, 0, _make_lparam(SELF_CENTER_X, SELF_CENTER_Y))
        except Exception:
            pass

    def _nudge_walk(self, hwnd, x, y):
        """右键点击指定位置，走一小步"""
        try:
            lparam = _make_lparam(x, y)
            PostMessage(hwnd, WM_RBUTTONDOWN, MK_RBUTTON, lparam)
            time.sleep(0.15)
            PostMessage(hwnd, WM_RBUTTONUP, 0, lparam)
        except Exception as e:
            print(f"[PICK] 移动失败: {e}")

    def _click_item(self, hwnd, pos):
        """左键点击物品位置（往上偏移25px）"""
        try:
            click_x = pos[0]
            click_y = pos[1] - 25
            lparam = _make_lparam(click_x, click_y)
            PostMessage(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
            PostMessage(hwnd, WM_LBUTTONUP, 0, lparam)
            # 记录点击位置用于可视化
            self.last_click_pos = (int(click_x), int(click_y))
            self.last_click_time = time.time()
        except Exception as e:
            print(f"[PICK] 点击失败: {e}")

    def _find_same_item(self, items):
        """在检测结果中找到和当前目标最近的物品"""
        if not items or self.target_pos is None:
            return None

        tx, ty = self.target_pos
        best = None
        best_dist = 999

        for item in items:
            bx, by, bw, bh = item
            cx = bx + bw // 2
            cy = by + bh // 2
            d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best = item

        if best_dist < 80:
            return best
        return None

    def _is_skipped(self, cx, cy, now):
        for sx, sy, expire in self._skip_list:
            if ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5 < self.SKIP_RADIUS:
                if now < expire:
                    return True
        return False

    def _add_skip(self, pos):
        if pos:
            self._skip_list.append((pos[0], pos[1], time.time() + self.SKIP_DURATION))
            print(f"[PICK] 标记跳过 ({pos[0]},{pos[1]}) {self.SKIP_DURATION}s")
