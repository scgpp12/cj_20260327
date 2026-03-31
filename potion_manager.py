"""
potion_manager.py - 自动喝药模块
检测 HP/MP 血球填充比例，低于阈值自动按键喝药。
HP = 左半圆红色，MP = 右半圆蓝色
"""

import time
import math
import cv2
import numpy as np
import ctypes

from config import (
    POTION_HP_KEY, POTION_MP_KEY,
    POTION_HP_THRESHOLD, POTION_MP_THRESHOLD,
    POTION_COOLDOWN, POTION_MP_COOLDOWN,
    POTION_ORB_CENTER_X, POTION_ORB_CENTER_Y, POTION_ORB_RADIUS,
)
from voice_alert import get_alert

# PostMessage 按键
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
user32 = ctypes.windll.user32
PostMessage = user32.PostMessageW

# 虚拟键码映射
VK_MAP = {
    '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
    '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38,
    '9': 0x39, '0': 0x30,
    'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
    'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
}


class PotionManager:
    """自动喝药管理器"""

    def __init__(self):
        self.enabled = True
        self.last_hp_time = 0
        self.last_mp_time = 0
        self.hp_ratio = 1.0
        self.mp_ratio = 1.0

        # 血球参数
        self.cx = POTION_ORB_CENTER_X
        self.cy = POTION_ORB_CENTER_Y
        self.radius = POTION_ORB_RADIUS

        # 按键虚拟码
        self.hp_vk = VK_MAP.get(POTION_HP_KEY.lower(), 0x32)  # 默认 '2'
        self.mp_vk = VK_MAP.get(POTION_MP_KEY.lower(), 0x33)  # 默认 '3'

        # 校准系数
        self.hp_full_raw = 0.83
        self.mp_full_raw = 0.83

        # 无效喝药检测
        self._hp_drink_count = 0
        self._mp_drink_count = 0
        self._hp_before_drink = 1.0
        self._mp_before_drink = 1.0
        self.MAX_INEFFECTIVE_DRINKS = 5  # 连续喝5次没效果就暂停

    def update(self, frame, game_hwnd):
        """
        每帧调用：检测血球 → 判断是否喝药 → 发送按键
        返回 (hp_ratio, mp_ratio, action)
        action: None / "HP" / "MP"
        """
        if not self.enabled or game_hwnd is None:
            return self.hp_ratio, self.mp_ratio, None

        # 检测 HP 和 MP 比例（校准后）
        raw_hp = self._detect_hp(frame)
        raw_mp = self._detect_mp(frame)
        self.hp_ratio = min(1.0, raw_hp / self.hp_full_raw)
        self.mp_ratio = min(1.0, raw_mp / self.mp_full_raw)

        now = time.time()
        action = None

        # HP 优先
        if (self.hp_ratio < POTION_HP_THRESHOLD
                and now - self.last_hp_time >= POTION_COOLDOWN
                and self._hp_drink_count < self.MAX_INEFFECTIVE_DRINKS):
            # 检查上次喝药有没有效果
            if self._hp_drink_count > 0 and self.hp_ratio <= self._hp_before_drink + 0.05:
                self._hp_drink_count += 1
            else:
                self._hp_drink_count = 0
            self._hp_before_drink = self.hp_ratio
            self._press_key(game_hwnd, self.hp_vk)
            self.last_hp_time = now
            self._hp_drink_count += 1
            action = "HP"
            if self._hp_drink_count >= self.MAX_INEFFECTIVE_DRINKS:
                print(f"[POTION] 喝血药无效{self.MAX_INEFFECTIVE_DRINKS}次，暂停30秒")
                self.last_hp_time = now + 30  # 暂停30秒
                self._hp_drink_count = 0
            else:
                print(f"[POTION] 喝血药! HP={self.hp_ratio:.0%}")

        elif (self.mp_ratio < POTION_MP_THRESHOLD
                and now - self.last_mp_time >= POTION_MP_COOLDOWN
                and self._mp_drink_count < self.MAX_INEFFECTIVE_DRINKS):
            if self._mp_drink_count > 0 and self.mp_ratio <= self._mp_before_drink + 0.05:
                self._mp_drink_count += 1
            else:
                self._mp_drink_count = 0
            self._mp_before_drink = self.mp_ratio
            self._press_key(game_hwnd, self.mp_vk)
            self.last_mp_time = now
            self._mp_drink_count += 1
            action = "MP"
            if self._mp_drink_count >= self.MAX_INEFFECTIVE_DRINKS:
                print(f"[POTION] 喝蓝药无效{self.MAX_INEFFECTIVE_DRINKS}次，暂停30秒")
                self.last_mp_time = now + 30
                self._mp_drink_count = 0
            else:
                print(f"[POTION] 喝蓝药! MP={self.mp_ratio:.0%}")

        # HP 语音警告：持续提醒直到 HP 恢复
        if self.hp_ratio < 0.3:
            get_alert().say("HP严重不足，请立即处理", cooldown=5.0)
        elif self.hp_ratio < 0.5:
            get_alert().say("当前HP已经不足50%，请注意", cooldown=8.0)

        return self.hp_ratio, self.mp_ratio, action

    def _detect_hp(self, frame):
        """
        检测 HP 比例：RGB 通道比较法。
        有血 = 红色通道明显高于绿蓝（渐变红、暗红都能检测）
        空血 = 灰色金属（R ≈ G ≈ B）
        白色高光 = R ≈ G ≈ B 且都很高，不算红色，但满血时占比小可忽略
        """
        h, w = frame.shape[:2]
        cx, cy, r = self.cx, self.cy, self.radius

        if cx - r < 0 or cy - r < 0 or cx + r >= w or cy + r >= h:
            return 1.0

        roi = frame[cy - r:cy + r, cx - r:cx + r]
        b, g, red = roi[:, :, 0].astype(float), roi[:, :, 1].astype(float), roi[:, :, 2].astype(float)

        # 左半圆遮罩（缩小 80% 避免边框）
        inner_r = int(r * 0.80)
        left_mask = np.zeros((r * 2, r * 2), dtype=np.uint8)
        cv2.ellipse(left_mask, (r, r), (inner_r, inner_r), 0, 90, 270, 255, -1)

        # 红色判定：R > G * 1.3 且 R > B * 1.3 且 R > 50
        is_red = (red > g * 1.3) & (red > b * 1.3) & (red > 50)
        is_red = is_red.astype(np.uint8) * 255

        valid = is_red & left_mask
        total_pixels = np.count_nonzero(left_mask)
        red_pixels = np.count_nonzero(valid)

        if total_pixels == 0:
            return 1.0

        return red_pixels / total_pixels

    def _detect_mp(self, frame):
        """
        检测 MP 比例：RGB 通道比较法。
        有蓝 = 蓝色通道明显高于红绿
        空蓝 = 灰色金属（R ≈ G ≈ B）
        """
        h, w = frame.shape[:2]
        cx, cy, r = self.cx, self.cy, self.radius

        if cx - r < 0 or cy - r < 0 or cx + r >= w or cy + r >= h:
            return 1.0

        roi = frame[cy - r:cy + r, cx - r:cx + r]
        b, g, red = roi[:, :, 0].astype(float), roi[:, :, 1].astype(float), roi[:, :, 2].astype(float)

        # 右半圆遮罩
        inner_r = int(r * 0.80)
        right_mask = np.zeros((r * 2, r * 2), dtype=np.uint8)
        cv2.ellipse(right_mask, (r, r), (inner_r, inner_r), 0, -90, 90, 255, -1)

        # 蓝色判定：B > R * 1.3 且 B > G * 1.2 且 B > 50
        is_blue = (b > red * 1.3) & (b > g * 1.2) & (b > 50)
        is_blue = is_blue.astype(np.uint8) * 255

        valid = is_blue & right_mask
        total_pixels = np.count_nonzero(right_mask)
        blue_pixels = np.count_nonzero(valid)

        if total_pixels == 0:
            return 1.0

        return blue_pixels / total_pixels

    def _press_key(self, hwnd, vk_code):
        """PostMessage 发送按键"""
        try:
            PostMessage(hwnd, WM_KEYDOWN, vk_code, 0)
            PostMessage(hwnd, WM_KEYUP, vk_code, 0)
        except Exception as e:
            print(f"[POTION] 按键失败: {e}")

    def get_state(self):
        """返回当前状态（给可视化用）"""
        return {
            "enabled": self.enabled,
            "hp_ratio": self.hp_ratio,
            "mp_ratio": self.mp_ratio,
        }
