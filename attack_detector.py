"""
attack_detector.py - 检测玩家角色是否在执行攻击动画
通过监测角色区域（SELF zone）的像素变化量来判断：
  - 站立/跑步时变化较小且均匀
  - 攻击时有明显的亮度爆发（武器挥动、技能特效）
"""

import cv2
import numpy as np
import time

try:
    from config import EXCLUDE_CENTER_X1, EXCLUDE_CENTER_Y1, EXCLUDE_CENTER_X2, EXCLUDE_CENTER_Y2
    SELF_X1 = EXCLUDE_CENTER_X1
    SELF_Y1 = EXCLUDE_CENTER_Y1
    SELF_X2 = EXCLUDE_CENTER_X2
    SELF_Y2 = EXCLUDE_CENTER_Y2
except ImportError:
    SELF_X1 = 900
    SELF_Y1 = 380
    SELF_X2 = 1020
    SELF_Y2 = 500


class AttackDetector:
    """
    通过角色区域像素变化检测攻击动画

    原理：
    - 角色区域（SELF zone）+ 周围扩展区域
    - 攻击动画产生大面积高亮变化（武器光效、技能特效）
    - 帧差的均值/最大值超过阈值 → 判定为正在攻击
    """

    def __init__(self):
        self.prev_gray = None
        self.is_attacking = False
        self.attack_start_time = 0
        self.last_attack_time = 0

        # 参数
        self.EXPAND = 50            # 检测区域向外扩展像素
        self.DIFF_THRESHOLD = 25    # 帧差二值化阈值
        self.CHANGE_RATIO = 0.15    # 变化像素占比超过此值 → 攻击中
        self.ATTACK_SUSTAIN = 0.5   # 攻击状态持续时间（秒），避免闪烁
        self.BRIGHTNESS_SPIKE = 40  # 亮度突变阈值（帧差均值）

    def detect(self, frame):
        """
        检测角色是否在攻击

        Args:
            frame: 当前帧 (BGR)

        Returns:
            dict: {
                "is_attacking": bool,
                "change_ratio": float,  # 变化像素占比
                "diff_mean": float,     # 帧差均值
            }
        """
        h, w = frame.shape[:2]

        # 扩展检测区域（攻击特效通常比角色大）
        rx1 = max(0, SELF_X1 - self.EXPAND)
        ry1 = max(0, SELF_Y1 - self.EXPAND)
        rx2 = min(w, SELF_X2 + self.EXPAND)
        ry2 = min(h, SELF_Y2 + self.EXPAND)

        roi = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        result = {
            "is_attacking": False,
            "change_ratio": 0.0,
            "diff_mean": 0.0,
        }

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            return result

        # 帧差
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray.copy()

        # 二值化
        _, thresh = cv2.threshold(diff, self.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = np.count_nonzero(thresh)
        change_ratio = changed_pixels / total_pixels if total_pixels > 0 else 0
        diff_mean = float(diff.mean())

        result["change_ratio"] = change_ratio
        result["diff_mean"] = diff_mean

        now = time.time()

        # 判定：变化比例 > 阈值 OR 亮度突变
        if change_ratio >= self.CHANGE_RATIO or diff_mean >= self.BRIGHTNESS_SPIKE:
            self.is_attacking = True
            self.attack_start_time = now
            self.last_attack_time = now

        # 攻击状态持续
        if self.is_attacking and now - self.last_attack_time > self.ATTACK_SUSTAIN:
            self.is_attacking = False

        result["is_attacking"] = self.is_attacking
        return result
