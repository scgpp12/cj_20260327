"""
target_tracker.py - 目标追踪器
跨帧位置匹配 + 稳定性过滤（连续 N 帧才算有效目标）
防止帧间跳动、闪烁、误检
"""

import time


class TrackedTarget:
    """单个被追踪的目标"""

    def __init__(self, box, target_id):
        self.id = target_id
        self.box = box           # (x, y, w, h)
        self.frames_seen = 1     # 连续看到的帧数
        self.frames_lost = 0     # 连续丢失的帧数
        self.stable = False      # 是否稳定（连续 N 帧）
        self.last_seen = time.time()
        self.first_seen = time.time()

    @property
    def center(self):
        x, y, w, h = self.box
        return x + w // 2, y + h // 2

    def dist_to(self, other_box):
        """和另一个框的中心距离"""
        cx1, cy1 = self.center
        x2, y2, w2, h2 = other_box
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


class TargetTracker:
    """
    目标追踪器

    功能：
    1. 跨帧匹配：用位置距离匹配同一个目标
    2. 稳定过滤：连续 N 帧检测到才算有效
    3. 短暂消失容忍：消失 1-2 帧不立刻删除
    """

    def __init__(self, match_dist=60, stable_frames=3, lost_tolerance=2):
        """
        Args:
            match_dist: 位置匹配阈值（像素），两帧间同一目标的最大移动距离
            stable_frames: 连续多少帧才算稳定目标
            lost_tolerance: 消失几帧后才删除（容忍短暂遮挡）
        """
        self.match_dist = match_dist
        self.stable_frames = stable_frames
        self.lost_tolerance = lost_tolerance

        self._tracked = []       # List[TrackedTarget]
        self._next_id = 0

    def update(self, detections):
        """
        每帧调用，传入当前帧的检测结果

        Args:
            detections: [(x, y, w, h), ...] 当前帧检测到的所有目标框

        Returns:
            stable_targets: [(x, y, w, h), ...] 稳定目标列表（连续 N 帧）
        """
        # 1. 匹配：当前检测 ↔ 已有追踪目标
        matched_track_ids = set()
        matched_det_ids = set()

        # 构建距离矩阵，贪心匹配（最近优先）
        pairs = []
        for ti, track in enumerate(self._tracked):
            for di, det in enumerate(detections):
                dist = track.dist_to(det)
                if dist < self.match_dist:
                    pairs.append((dist, ti, di))

        pairs.sort(key=lambda x: x[0])  # 距离最近的先匹配

        for dist, ti, di in pairs:
            if ti in matched_track_ids or di in matched_det_ids:
                continue
            # 匹配成功：更新位置
            self._tracked[ti].box = detections[di]
            self._tracked[ti].frames_seen += 1
            self._tracked[ti].frames_lost = 0
            self._tracked[ti].last_seen = time.time()

            if self._tracked[ti].frames_seen >= self.stable_frames:
                self._tracked[ti].stable = True

            matched_track_ids.add(ti)
            matched_det_ids.add(di)

        # 2. 未匹配的检测 → 新目标
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                new_track = TrackedTarget(det, self._next_id)
                self._next_id += 1
                self._tracked.append(new_track)

        # 3. 未匹配的追踪 → 丢失计数
        for ti, track in enumerate(self._tracked):
            if ti not in matched_track_ids:
                track.frames_lost += 1

        # 4. 删除丢失太久的目标
        self._tracked = [t for t in self._tracked
                         if t.frames_lost <= self.lost_tolerance]

        # 5. 返回稳定目标
        stable = [t.box for t in self._tracked if t.stable]
        return stable

    def get_all_tracked(self):
        """获取所有追踪中的目标（包括不稳定的）"""
        return [(t.box, t.stable, t.frames_seen, t.id) for t in self._tracked]

    def get_nearest_stable(self, ref_x, ref_y):
        """获取距离参考点最近的稳定目标"""
        best = None
        best_dist = float('inf')
        for t in self._tracked:
            if not t.stable:
                continue
            cx, cy = t.center
            dist = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = t.box
        return best, best_dist

    def clear(self):
        """清除所有追踪"""
        self._tracked.clear()

    @property
    def stable_count(self):
        return sum(1 for t in self._tracked if t.stable)

    @property
    def total_count(self):
        return len(self._tracked)
