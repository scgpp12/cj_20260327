"""
main.py - 主循环 (ver05)
YOLO 检测（主） + 传统 CV 血条检测（备选） → 攻击最近目标 → 无怪巡逻
"""

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import os
import time
import cv2
import numpy as np
from config import (
    CAPTURE_WINDOW_TITLE, OUTPUT_DIR,
    AUTO_ATTACK_ENABLED, AUTO_ATTACK_COOLDOWN,
    PATROL_ENABLED, SELF_CENTER_X, SELF_CENTER_Y,
    YOLO_ENABLED, YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU,
    YOLO_MONSTER_CLASSES, YOLO_SELF_CLASSES,
)
from screen_capture import ScreenCapture, list_and_select_window
from hp_detector import HPDetector
from action_controller import ActionController, find_game_hwnd
from patrol_controller import PatrolController
from target_tracker import TargetTracker

# 红球检测器（可选）
redball_detector = None
try:
    from config import REDBALL_ENABLED
    if REDBALL_ENABLED:
        from redball_detector import RedBallDetector
        from config import (REDBALL_MIN_AREA, REDBALL_MAX_AREA,
                            REDBALL_MIN_CIRCULARITY)
        redball_detector = RedBallDetector(
            min_area=REDBALL_MIN_AREA,
            max_area=REDBALL_MAX_AREA,
            min_circularity=REDBALL_MIN_CIRCULARITY,
            self_center_x=SELF_CENTER_X,
            self_center_y=SELF_CENTER_Y,
        )
        print("[INFO] 红球检测已启用（怪物=红色球体）")
except Exception as e:
    print(f"[WARN] 红球检测不可用: {e}")
from visualizer import (
    draw_hp_box, draw_target_box_scan, draw_distance_lines,
    draw_attack_range, draw_exclude_zones, draw_patrol_info,
    draw_wall_overlay, draw_pathfinder_overlay, draw_yolo_all,
    draw_grid_overlay, draw_fps, draw_stats, resize_for_display,
)

# YOLO 检测器（可选）
yolo_detector = None
if YOLO_ENABLED:
    try:
        from yolo_detector import YoloDetector
        yolo_detector = YoloDetector(
            model_path=YOLO_MODEL_PATH,
            confidence=YOLO_CONFIDENCE,
            iou_threshold=YOLO_IOU,
        )
    except Exception as e:
        print(f"[WARN] YOLO 加载失败，回退到传统 CV: {e}")
        yolo_detector = None

# 音频检测器（可选）
audio_det = None
try:
    from audio_detector import AudioDetector
    audio_det = AudioDetector(match_threshold=0.65, cooldown=0.5)
    audio_det.start()
except Exception as e:
    print(f"[WARN] 音频检测不可用: {e}")

# 攻击动画检测器（可选）
atk_detector = None
try:
    from attack_detector import AttackDetector
    atk_detector = AttackDetector()
except Exception:
    pass

# 自动喝药（可选）
potion_mgr = None
try:
    from config import POTION_ENABLED
    if POTION_ENABLED:
        from potion_manager import PotionManager
        potion_mgr = PotionManager()
        print("[INFO] 自动喝药已启用")
except Exception as e:
    print(f"[WARN] 自动喝药不可用: {e}")

# 物品拾取（可选）
item_picker = None
try:
    from config import PICK_ENABLED
    if PICK_ENABLED:
        from item_picker import ItemPicker
        from config import PICK_RANGE, PICK_ARRIVE_DIST, PICK_TIMEOUT, PICK_WALK_TIMEOUT
        item_picker = ItemPicker(
            pick_range=PICK_RANGE,
            arrive_dist=PICK_ARRIVE_DIST,
            pick_timeout=PICK_TIMEOUT,
        )
        item_picker.walk_timeout = PICK_WALK_TIMEOUT
        print("[INFO] 物品拾取已启用")
except Exception as e:
    print(f"[WARN] 物品拾取不可用: {e}")


def _box_center(box):
    x, y, w, h = box
    return x + w // 2, y + h // 2


def _dist_to_self(box):
    cx, cy = _box_center(box)
    return ((cx - SELF_CENTER_X) ** 2 + (cy - SELF_CENTER_Y) ** 2) ** 0.5


def main():
    # 同时输出到控制台和日志文件
    import sys, io
    class TeeWriter:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try: s.write(data); s.flush()
                except: pass
        def flush(self):
            for s in self.streams:
                try: s.flush()
                except: pass
    log_file = open("output/main_log.txt", "w", encoding="utf-8")
    sys.stdout = TeeWriter(sys.__stdout__, log_file)

    print("=" * 60)
    print("  Game Detector ver05 (YOLO + CV)")
    print("=" * 60)

    game_window = list_and_select_window(title_hint=CAPTURE_WINDOW_TITLE)
    if game_window is None:
        print("[ERROR] 未选择目标，退出")
        return

    capture = ScreenCapture(window_obj=game_window)
    hp_detector = HPDetector()

    game_hwnd = find_game_hwnd()
    attacker = ActionController(click_cooldown=AUTO_ATTACK_COOLDOWN, game_hwnd=game_hwnd)
    if game_hwnd:
        attacker.set_hwnd(game_hwnd)
    attacker.enabled = AUTO_ATTACK_ENABLED

    patrol = PatrolController()
    patrol.enabled = PATROL_ENABLED

    # 目标追踪器（跨帧稳定，3帧确认）
    monster_tracker = TargetTracker(match_dist=60, stable_frames=3, lost_tolerance=2)

    # 让攻击器能释放巡逻的右键
    attacker._patrol_ref = patrol

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    paused = False
    last_display_frame = None
    debug_hsv = False
    mouse_pos = [0, 0]
    use_yolo = yolo_detector is not None
    zoom_level = [1.0]       # 缩放倍数（列表便于闭包修改）
    zoom_center = [0, 0]     # 缩放中心（鼠标位置）
    pan_offset = [0, 0]      # 平移偏移
    pan_dragging = [False]
    pan_start = [0, 0]

    def _mouse_cb(event, x, y, flags, param):
        mouse_pos[0] = x
        mouse_pos[1] = y

        # 滚轮缩放
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                zoom_level[0] = min(zoom_level[0] * 1.2, 4.0)  # 放大，最大4倍
            else:
                zoom_level[0] = max(zoom_level[0] / 1.2, 0.5)  # 缩小，最小0.5倍
            zoom_center[0] = x
            zoom_center[1] = y

        # 中键拖拽平移
        elif event == cv2.EVENT_MBUTTONDOWN:
            pan_dragging[0] = True
            pan_start[0] = x
            pan_start[1] = y
        elif event == cv2.EVENT_MOUSEMOVE and pan_dragging[0]:
            pan_offset[0] += x - pan_start[0]
            pan_offset[1] += y - pan_start[1]
            pan_start[0] = x
            pan_start[1] = y
        elif event == cv2.EVENT_MBUTTONUP:
            pan_dragging[0] = False

    mode_str = "YOLO" if use_yolo else "传统CV"
    print(f"  检测模式: {mode_str}")
    print(f"  ATK: {'ON' if AUTO_ATTACK_ENABLED else 'OFF'}  "
          f"PAT: {'ON' if PATROL_ENABLED else 'OFF'}")
    print(f"  快捷键: q=退出 p=暂停 a=攻击 r=巡逻 y=切换YOLO/CV d=HSV s=截图")
    print("=" * 60)

    try:
        loop_count = 0
        while True:
            if not paused:
                frame = capture.grab()
                if frame is None:
                    time.sleep(0.1)
                    continue

                if loop_count == 0:
                    print(f"[DEBUG] 第一帧: {frame.shape} mean={frame.mean():.1f}")
                    import sys; sys.stdout.flush()

                display_frame = frame.copy()
                if loop_count == 0: print("[DEBUG] 1-copy OK"); sys.stdout.flush()
                target_count = 0
                all_targets = []  # (x, y, w, h) 列表，供攻击系统使用

                # =========================================
                # 检测阶段：YOLO 或 传统 CV
                # =========================================
                monsters_yolo = []
                self_yolo = []

                if redball_detector is not None:
                    # ---- 红球检测（最高优先级）----
                    if loop_count == 0: print("[DEBUG] 2-redball开始"); sys.stdout.flush()
                    balls = redball_detector.detect(frame)
                    if loop_count == 0: print(f"[DEBUG] 2-redball完成: {len(balls)}个"); sys.stdout.flush()
                    for b in balls:
                        bx, by, bw, bh = b["box"]
                        cx, cy = b["center"]
                        dist = b["dist"]
                        circ = b["circularity"]

                        # 画红色框 + 标签
                        cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)
                        label = f"MON {dist:.0f}px c={circ:.2f}"
                        cv2.putText(display_frame, label, (bx, by - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                        cv2.putText(display_frame, label, (bx, by - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                        all_targets.append(b["box"])
                        target_count += 1

                elif use_yolo and yolo_detector is not None:
                    # ---- YOLO 检测 ----
                    monsters_yolo = yolo_detector.detect_monsters(frame, YOLO_MONSTER_CLASSES)
                    self_yolo = yolo_detector.detect_self(frame, YOLO_SELF_CLASSES)

                    # 可视化
                    draw_yolo_all(display_frame, monsters_yolo, self_yolo)

                    # 提取目标框
                    for m in monsters_yolo:
                        all_targets.append(m["target_box"])
                        target_count += 1

                else:
                    # ---- 传统 CV 血条检测 ----
                    if atk_detector:
                        atk_detector.detect(frame)

                    scan_results = hp_detector.scan_full_frame(frame)
                    for info in scan_results:
                        draw_hp_box(display_frame, info["hp_box"])
                        draw_target_box_scan(display_frame, info)
                        all_targets.append(info["target_box"])
                        target_count += 1

                # =========================================
                # 目标追踪 + 稳定性过滤
                # =========================================
                # 把原始检测结果送入追踪器，只返回连续3帧稳定的目标
                stable_targets = monster_tracker.update(all_targets)

                # 画不稳定目标（灰色虚框，不参与攻击）
                for box, is_stable, frames, tid in monster_tracker.get_all_tracked():
                    if not is_stable:
                        bx, by, bw, bh = box
                        cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh),
                                      (128, 128, 128), 1)
                        cv2.putText(display_frame, f"?{frames}f", (bx, by - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)

                # =========================================
                # 距离过滤：只攻击范围内的稳定怪物
                # =========================================
                min_dist = 0 if (redball_detector is not None or use_yolo) else 80
                in_range = [t for t in stable_targets
                            if min_dist <= _dist_to_self(t) <= 300]

                # 画角色中心（紫色圆点）
                cv2.circle(display_frame, (SELF_CENTER_X, SELF_CENTER_Y), 6, (255, 0, 255), -1)
                cv2.circle(display_frame, (SELF_CENTER_X, SELF_CENTER_Y), 8, (255, 0, 255), 1)

                if loop_count == 0: print("[DEBUG] 3-tracker完成"); sys.stdout.flush()
                if loop_count == 0: print("[DEBUG] 3a-声音检测开始"); sys.stdout.flush()
                # 画攻击范围 + 距离线
                draw_attack_range(display_frame, radius=300)
                draw_distance_lines(display_frame, all_targets)

                # =========================================
                # 声音/画面状态 → 传给攻击控制器
                # =========================================
                if audio_det:
                    audio_state = audio_det.get_state()
                    attacker.set_audio_state(audio_state.get("attack_hit", False))

                # 简单画面变化检测（用帧差判断角色是否在动）
                if hasattr(attacker, '_prev_frame_gray'):
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_y1 = max(0, SELF_CENTER_Y - 100)
                    roi_y2 = min(frame.shape[0], SELF_CENTER_Y + 100)
                    roi_x1 = max(0, SELF_CENTER_X - 100)
                    roi_x2 = min(frame.shape[1], SELF_CENTER_X + 100)
                    diff = cv2.absdiff(
                        attacker._prev_frame_gray[roi_y1:roi_y2, roi_x1:roi_x2],
                        curr_gray[roi_y1:roi_y2, roi_x1:roi_x2]
                    )
                    is_moving = diff.mean() > 3.0
                    attacker.set_visual_state(is_moving)
                    attacker._prev_frame_gray = curr_gray
                else:
                    attacker._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if loop_count == 0: print("[DEBUG] 3b-攻击/巡逻开始"); sys.stdout.flush()
                # =========================================
                # 攻击 or 巡逻
                # =========================================
                in_range.sort(key=lambda t: _dist_to_self(t))

                # 更新攻击控制器
                atk_info = attacker.update(in_range)

                # 画 9 方位攻击点绿线（始终显示，方便调试）
                atk_ox, atk_oy = attacker.atk_origin_x, attacker.atk_origin_y
                for dir_name, ax, ay in attacker.atk_directions:
                    if dir_name != "CENTER":
                        cv2.line(display_frame, (atk_ox, atk_oy),
                                 (ax, ay), (0, 200, 0), 1)
                        cv2.circle(display_frame, (ax, ay), 4, (0, 255, 0), -1)

                if atk_info["state"] != "IDLE":
                    # 攻击状态中 → 停止巡逻
                    patrol.on_target_found()

                    # 画锁定目标
                    if atk_info["locked"] and atk_info["target"] is not None:
                        lx, ly, lw, lh = atk_info["target"]
                        lcx, lcy = lx + lw // 2, ly + lh // 2
                        dist = _dist_to_self(atk_info["target"])

                        cv2.rectangle(display_frame, (lx - 5, ly - 5),
                                      (lx + lw + 5, ly + lh + 5), (255, 255, 255), 3)
                        cv2.drawMarker(display_frame, (lcx, lcy), (0, 0, 255),
                                       cv2.MARKER_CROSS, 30, 2)
                        cv2.line(display_frame, (SELF_CENTER_X, SELF_CENTER_Y),
                                 (lcx, lcy), (0, 255, 255), 1)

                        state_label = atk_info["state"]
                        dir_label = attacker.last_atk_dir or ""
                        label = f"{state_label} #{atk_info['count']} {dir_label} d={dist:.0f}"
                        cv2.putText(display_frame, label, (lx, ly - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                        cv2.putText(display_frame, label, (lx, ly - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # 攻击点击标记：Shift长按=紫色，普通左键=蓝色
                    if attacker.last_click_pos and \
                       time.time() - attacker.last_click_time < attacker.CLICK_SHOW_DURATION:
                        acx, acy = attacker.last_click_pos
                        if attacker._shift_held:
                            dot_color = (180, 0, 255)   # 紫色 (Shift+左键长按)
                        else:
                            dot_color = (255, 50, 50)   # 蓝色 (普通左键)
                        cv2.circle(display_frame, (acx, acy), 12, dot_color, -1)
                        cv2.circle(display_frame, (acx, acy), 12, (255, 255, 255), 2)
                        dir_text = attacker.last_atk_dir or "ATK"
                        cv2.putText(display_frame, dir_text, (acx + 15, acy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_color, 1)
                else:
                    # IDLE → 拾取 or 巡逻
                    attacker.on_target_lost()
                    # 只有角色实际移动了才清除误检标记
                    if getattr(patrol, '_moved_since_last_check', False):
                        attacker.clear_false_targets()
                        patrol._moved_since_last_check = False
                    patrol.clear_chase_target()

                    if loop_count == 0: print("[DEBUG] 3c-IDLE分支,准备拾取"); sys.stdout.flush()
                    # ---- 物品拾取（优先级高于巡逻）----
                    pick_active = False
                    if item_picker is not None and item_picker.enabled:
                        pick_result = item_picker.update(frame, game_hwnd, has_combat_target=False)
                        pick_active = pick_result["picking"]

                        # 拾取可视化 + 日志
                        detected_items = item_picker.detect_items(frame)
                        if detected_items and frame_count == 0:
                            dists = []
                            for ix, iy, iw, ih in detected_items:
                                d = ((ix+iw//2-SELF_CENTER_X)**2 + (iy+ih//2-SELF_CENTER_Y)**2)**0.5
                                dists.append(f"{d:.0f}px")
                            print(f"[PICK] 检测到 {len(detected_items)} 个物品 dist=[{', '.join(dists)}] state={pick_result['state']}")
                        for ix, iy, iw, ih in detected_items:
                            # 橙色框：物品
                            cv2.rectangle(display_frame, (ix, iy), (ix + iw, iy + ih), (0, 165, 255), 2)
                            cv2.putText(display_frame, "ITEM", (ix, iy - 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

                        if pick_result["target"]:
                            px, py = pick_result["target"]
                            # 拾取目标十字
                            cv2.drawMarker(display_frame, (px, py), (0, 165, 255),
                                           cv2.MARKER_CROSS, 20, 2)
                            cv2.line(display_frame, (SELF_CENTER_X, SELF_CENTER_Y),
                                     (px, py), (0, 165, 255), 2)
                            cv2.putText(display_frame, f"PICK [{pick_result['state']}]",
                                        (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 165, 255), 2)

                        # 左键点击红点标记
                        if item_picker.last_click_pos and \
                           time.time() - item_picker.last_click_time < item_picker.CLICK_SHOW_DURATION:
                            cx, cy = item_picker.last_click_pos
                            cv2.circle(display_frame, (cx, cy), 12, (0, 0, 255), -1)
                            cv2.circle(display_frame, (cx, cy), 12, (255, 255, 255), 2)
                            cv2.putText(display_frame, "CLICK", (cx + 15, cy + 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    if loop_count == 0: print("[DEBUG] 3d-拾取完成,准备巡逻"); sys.stdout.flush()
                    # ---- 巡逻（拾取不活跃时）----
                    if not pick_active:
                        # 把远处稳定怪物位置告诉巡逻器
                        out_range = [t for t in stable_targets if _dist_to_self(t) > 300]
                        monster_hints = []
                        for t in out_range:
                            cx, cy = _box_center(t)
                            monster_hints.append((cx, cy))
                        patrol.set_monster_hints(monster_hints)

                        patrol.on_target_lost()
                        patrol.update(frame, game_hwnd)
                    else:
                        # 拾取中 → 停止巡逻移动
                        patrol.on_target_found()  # 暂停巡逻

                if loop_count == 0: print("[DEBUG] 4-攻击/巡逻完成"); sys.stdout.flush()
                # A* 可视化
                if patrol.pathfinder is not None:
                    if patrol.state == "PATROL":
                        draw_pathfinder_overlay(display_frame, patrol.pathfinder)
                    else:
                        patrol.pathfinder._last_waypoints = None

                # =========================================
                # 显示信息
                # =========================================
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start_time = time.time()

                draw_exclude_zones(display_frame)
                draw_wall_overlay(display_frame, patrol.wall_mask)
                draw_patrol_info(display_frame, patrol.info)
                from config import GRID_VIZ_ENABLED
                if GRID_VIZ_ENABLED:
                    draw_grid_overlay(display_frame, patrol.info.get("grid_data"))
                draw_fps(display_frame, fps)
                draw_stats(display_frame, 0, target_count)

                # ---- 自动喝药 ----
                if potion_mgr is not None:
                    hp_r, mp_r, pot_action = potion_mgr.update(frame, game_hwnd)
                    # HP/MP 比例显示（左上角）
                    hp_color = (0, 255, 0) if hp_r >= 0.6 else (0, 165, 255) if hp_r >= 0.3 else (0, 0, 255)
                    mp_color = (255, 200, 0) if mp_r >= 0.3 else (0, 0, 255)
                    cv2.putText(display_frame, f"HP:{hp_r:.0%}", (10, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(display_frame, f"HP:{hp_r:.0%}", (10, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hp_color, 1)
                    cv2.putText(display_frame, f"MP:{mp_r:.0%}", (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(display_frame, f"MP:{mp_r:.0%}", (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mp_color, 1)
                    if pot_action:
                        cv2.putText(display_frame, f"DRINK {pot_action}!", (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                        cv2.putText(display_frame, f"DRINK {pot_action}!", (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 状态栏
                mode_tag = "YOLO" if use_yolo else "CV"
                status = (f"ATK:{'ON' if attacker.enabled else 'OFF'} "
                          f"PAT:{'ON' if patrol.enabled else 'OFF'} "
                          f"[{mode_tag}]")
                cv2.putText(display_frame, status, (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(display_frame, status, (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 音频检测显示
                if audio_det:
                    audio_state = audio_det.get_state()
                    if audio_state.get("has_fingerprint"):
                        if audio_state.get("attack_hit"):
                            cv2.putText(display_frame, "HIT!", (10, 130),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                            cv2.putText(display_frame, "HIT!", (10, 130),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        score = audio_state.get("match_score", 0)
                        cv2.putText(display_frame, f"AUDIO: {score:.2f}", (10, 155),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

                # HSV 调试
                if debug_hsv:
                    mx, my = mouse_pos
                    fh, fw = frame.shape[:2]
                    sh, sw = resize_for_display(display_frame).shape[:2]
                    ox = int(mx * fw / sw) if sw > 0 else 0
                    oy = int(my * fh / sh) if sh > 0 else 0
                    ox = max(0, min(fw - 1, ox))
                    oy = max(0, min(fh - 1, oy))

                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    h_val, s_val, v_val = hsv_frame[oy, ox]
                    b_val, g_val, r_val = frame[oy, ox]

                    info_text = f"({ox},{oy}) H:{h_val} S:{s_val} V:{v_val} | B:{b_val} G:{g_val} R:{r_val}"
                    cv2.putText(display_frame, info_text, (10, fh - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(display_frame, info_text, (10, fh - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.drawMarker(display_frame, (ox, oy), (0, 255, 255),
                                   cv2.MARKER_CROSS, 20, 1)

                last_display_frame = display_frame

                # 缩放显示
                if zoom_level[0] != 1.0:
                    fh, fw = display_frame.shape[:2]
                    # 缩放后尺寸
                    new_w = int(fw * zoom_level[0])
                    new_h = int(fh * zoom_level[0])
                    zoomed = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    # 裁剪显示区域（以缩放中心为基准）
                    cx = int(zoom_center[0] * zoom_level[0]) + pan_offset[0]
                    cy = int(zoom_center[1] * zoom_level[0]) + pan_offset[1]

                    # 显示窗口大小
                    disp_w = min(fw, 1920)
                    disp_h = min(fh, 1080)

                    x1 = max(0, cx - disp_w // 2)
                    y1 = max(0, cy - disp_h // 2)
                    x2 = min(new_w, x1 + disp_w)
                    y2 = min(new_h, y1 + disp_h)
                    x1 = max(0, x2 - disp_w)
                    y1 = max(0, y2 - disp_h)

                    show_frame = zoomed[y1:y2, x1:x2]

                    # 显示缩放信息
                    cv2.putText(show_frame, f"ZOOM: {zoom_level[0]:.1f}x", (10, show_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(show_frame, f"ZOOM: {zoom_level[0]:.1f}x", (10, show_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    show_frame = resize_for_display(display_frame)

                if loop_count == 0: print("[DEBUG] 5-准备显示"); sys.stdout.flush()
                cv2.imshow("Game Detector", show_frame)
                cv2.setMouseCallback("Game Detector", _mouse_cb)
                loop_count += 1
                if loop_count <= 3:
                    print(f"[DEBUG] 帧{loop_count} 显示完成"); sys.stdout.flush()

            # =========================================
            # 按键处理
            # =========================================
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print(f"[INFO] {'暂停' if paused else '继续'}")
            elif key == ord("d"):
                debug_hsv = not debug_hsv
                if debug_hsv:
                    cv2.setMouseCallback("Game Detector", _mouse_cb)
                print(f"[INFO] HSV调试: {'ON' if debug_hsv else 'OFF'}")
            elif key == ord("a"):
                attacker.enabled = not attacker.enabled
                print(f"[INFO] ATK: {'ON' if attacker.enabled else 'OFF'}")
            elif key == ord("r"):
                patrol.enabled = not patrol.enabled
                print(f"[INFO] PAT: {'ON' if patrol.enabled else 'OFF'}")
            elif key == ord("y"):
                if yolo_detector is not None:
                    use_yolo = not use_yolo
                    print(f"[INFO] 检测模式: {'YOLO' if use_yolo else '传统CV'}")
                else:
                    print("[INFO] YOLO 模型未加载，无法切换")
            elif key == ord("s"):
                if last_display_frame is not None:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(OUTPUT_DIR, f"debug_{ts}.png")
                    cv2.imwrite(path, last_display_frame)
                    print(f"[INFO] 已保存: {path}")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C")
    finally:
        if audio_det:
            audio_det.stop()
        capture.release()
        cv2.destroyAllWindows()
        print("[INFO] 结束")


if __name__ == "__main__":
    main()
