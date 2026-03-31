"""
visualizer.py — 调试画面绘制模块
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【职责】
  将所有调试信息绘制到 display_frame 上，不影响游戏逻辑。

【主要函数】
  draw_patrol_info(frame, patrol.info)
    → 右上角状态文字 + 角色中心方向箭头（青黄色）
    → 角色中心 → 点击目标紫色箭头（精确点击位置）
    → 底部路线进度条
  draw_attack_range(frame)    → 攻击范围圆圈
  draw_exclude_zones(frame)   → UI排除区域线
  draw_grid_overlay(frame, grid_data)  → 右上角小地图
  draw_yolo_all(frame, monsters, self_dets)  → YOLO检测框

【patrol.info 字段说明】
  state         当前状态 (IDLE/PATROL/STUCK/COMBAT)
  direction     当前移动方向 (UP/DOWN/LEFT/RIGHT 等8方向)
  click_pos     本次右键点击的屏幕坐标 (px, py)
  route_index   当前路线点序号
  route_total   路线总点数
  route_target  当前目标点世界坐标 (rx, ry)
  route_dist    与目标点的曼哈顿距离
"""

import cv2
import numpy as np
from config import (
    BOX_THICKNESS, FONT_SCALE,
    DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT,
    EXCLUDE_LEFT_RATIO, EXCLUDE_BOTTOM_RATIO,
    EXCLUDE_TOP_RATIO, EXCLUDE_MINIMAP_RIGHT_RATIO,
    EXCLUDE_MINIMAP_TOP_RATIO,
    SELF_CENTER_X, SELF_CENTER_Y,
)

COLOR_HP = (0, 0, 255)
COLOR_TARGET = (0, 255, 0)
COLOR_TEXT_BG = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hp_box(frame, box):
    """绘制血条检测框（红色）"""
    if box is None:
        return
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_HP, BOX_THICKNESS)


def draw_target_box_scan(frame, scan_info):
    """绘制怪物候选框（绿色）+ 来源标注"""
    tx, ty, tw, th = scan_info["target_box"]
    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), COLOR_TARGET, BOX_THICKNESS + 1)
    source = scan_info.get("source", "?")
    hx, hy, _, _ = scan_info["hp_box"]
    _put_label(frame, f"TARGET [{source}]", hx, hy - 5, COLOR_TARGET)


def draw_attack_range(frame, radius=300):
    """画攻击范围圈"""
    self_pt = (SELF_CENTER_X, SELF_CENTER_Y)
    cv2.circle(frame, self_pt, radius, (0, 180, 180), 1)
    cv2.circle(frame, self_pt, 80, (100, 100, 100), 1)  # 最小距离圈


def draw_distance_lines(frame, targets, locked_target=None):
    """
    画角色到所有目标的距离线
    最近的用红色粗线，其他用黑色细线
    """
    if not targets:
        return

    self_pt = (SELF_CENTER_X, SELF_CENTER_Y)
    cv2.circle(frame, self_pt, 5, (0, 255, 255), -1)

    # 计算每个目标的距离
    target_dists = []
    for t in targets:
        tx, ty, tw, th = t
        tcx, tcy = tx + tw // 2, ty + th // 2
        dist = ((tcx - SELF_CENTER_X) ** 2 + (tcy - SELF_CENTER_Y) ** 2) ** 0.5
        target_dists.append((t, tcx, tcy, dist))

    # 找最近的
    target_dists.sort(key=lambda x: x[3])
    nearest = target_dists[0] if target_dists else None

    # 先画其他目标（黑色细线）
    for t, tcx, tcy, dist in target_dists[1:]:
        cv2.line(frame, self_pt, (tcx, tcy), (50, 50, 50), 1)
        mid_x, mid_y = (SELF_CENTER_X + tcx) // 2, (SELF_CENTER_Y + tcy) // 2
        _put_label(frame, f"{dist:.0f}", mid_x, mid_y, (150, 150, 150), scale=0.35)

    # 再画最近的（红色粗线）
    if nearest:
        _, tcx, tcy, dist = nearest
        cv2.line(frame, self_pt, (tcx, tcy), (0, 0, 255), 2)
        mid_x, mid_y = (SELF_CENTER_X + tcx) // 2, (SELF_CENTER_Y + tcy) // 2
        _put_label(frame, f"{dist:.0f}px NEAREST", mid_x, mid_y, (0, 0, 255), scale=0.45)


def draw_exclude_zones(frame):
    """用黄色线绘制 UI 排除区域"""
    h, w = frame.shape[:2]
    color = (0, 200, 200)

    ex_left = int(w * EXCLUDE_LEFT_RATIO)
    ex_top = int(h * EXCLUDE_TOP_RATIO)
    ex_bottom = int(h * (1.0 - EXCLUDE_BOTTOM_RATIO))
    ex_minimap_x = int(w * (1.0 - EXCLUDE_MINIMAP_RIGHT_RATIO))
    ex_minimap_y = int(h * EXCLUDE_MINIMAP_TOP_RATIO)

    cv2.line(frame, (ex_left, 0), (ex_left, h), color, 1)
    cv2.line(frame, (0, ex_top), (w, ex_top), color, 1)
    cv2.line(frame, (0, ex_bottom), (w, ex_bottom), color, 1)
    cv2.rectangle(frame, (ex_minimap_x, 0), (w, ex_minimap_y), color, 1)

    # 画血球检测范围（青色圆圈）
    try:
        from config import POTION_ORB_CENTER_X, POTION_ORB_CENTER_Y, POTION_ORB_RADIUS
        cx = POTION_ORB_CENTER_X
        cy = POTION_ORB_CENTER_Y
        r = POTION_ORB_RADIUS
        # 整圆
        cv2.circle(frame, (cx, cy), r, (255, 255, 0), 2)
        # 中线分隔 HP/MP
        cv2.line(frame, (cx, cy - r), (cx, cy + r), (255, 255, 0), 1)
        # 标签
        cv2.putText(frame, "HP", (cx - r - 5, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, "MP", (cx + 5, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
    except ImportError:
        pass


def draw_yolo_detection(frame, det, is_monster=True):
    """
    绘制 YOLO 检测结果
    怪物：绿色框 + 类名 + 置信度
    自己：青色框
    """
    x, y, w, h = det["bbox"]
    conf = det["confidence"]
    cls_name = det["class"]

    if is_monster:
        color = (0, 255, 0)  # 绿色
        label = f"{cls_name} {conf:.0%}"
    else:
        color = (255, 255, 0)  # 青色
        label = f"SELF {conf:.0%}"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # 背景 + 前景文字
    cv2.putText(frame, label, (x, y - 5), FONT, 0.45, (0, 0, 0), 3)
    cv2.putText(frame, label, (x, y - 5), FONT, 0.45, color, 1)


def draw_yolo_all(frame, monsters, self_dets):
    """绘制所有 YOLO 检测结果"""
    for det in self_dets:
        draw_yolo_detection(frame, det, is_monster=False)
    for det in monsters:
        draw_yolo_detection(frame, det, is_monster=True)


def draw_wall_overlay(frame, wall_mask):
    """半透明红色墙壁覆盖（旧版，用 patrol wall_mask）"""
    if wall_mask is None:
        return
    overlay = frame.copy()
    overlay[wall_mask > 0] = (0, 0, 180)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


def draw_patrol_info(frame, patrol_info):
    """巡逻状态 + 路线目标标注 + 进度条"""
    state      = patrol_info.get("state", "IDLE")
    direction  = patrol_info.get("direction", "")
    click_pos  = patrol_info.get("click_pos")
    route_idx  = patrol_info.get("route_index", 0)
    route_tot  = patrol_info.get("route_total", 0)
    route_tgt  = patrol_info.get("route_target")   # (rx, ry) 世界坐标
    route_dist = patrol_info.get("route_dist", 0)

    state_colors = {
        "IDLE":   (150, 150, 150),
        "PATROL": (255, 200, 0),
        "STUCK":  (0, 0, 255),
        "COMBAT": (0, 255, 0),
    }
    color = state_colors.get(state, (200, 200, 200))
    fh, fw = frame.shape[:2]

    # ── 状态文字（右上角）──────────────────────────────
    text = f"STATE: {state}"
    if state == "PATROL":
        text += f" [{direction}]"
    _put_label(frame, text, fw - 260, 25, color, scale=0.6)

    # ── 行进方向指示（角色中心发出的粗箭头）──────────────
    if direction and state in ("PATROL", "STUCK"):
        try:
            from patrol_controller import DIRECTIONS
            ddx, ddy = DIRECTIONS[direction]
            arrow_len = 60
            ax = SELF_CENTER_X + int(ddx * arrow_len)
            ay = SELF_CENTER_Y + int(ddy * arrow_len)
            move_color = (0, 220, 255)  # 青黄色
            # 黑色描边（增加可读性）
            cv2.arrowedLine(frame, (SELF_CENTER_X, SELF_CENTER_Y),
                            (ax, ay), (0, 0, 0), 6, tipLength=0.35)
            # 实际颜色
            cv2.arrowedLine(frame, (SELF_CENTER_X, SELF_CENTER_Y),
                            (ax, ay), move_color, 3, tipLength=0.35)
            # 方向文字
            tx = max(10, min(fw - 80, ax + 6))
            ty = max(15, min(fh - 10, ay - 6))
            cv2.putText(frame, direction, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(frame, direction, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, move_color, 1)
        except Exception:
            pass

    # ── 路线目标箭头 + 标注（巡逻/卡住时）──────────────
    if click_pos is not None and state in ("PATROL", "STUCK"):
        cx, cy = click_pos
        purple = (200, 50, 255)

        # 箭头：角色中心 → 点击目标
        cv2.arrowedLine(frame, (SELF_CENTER_X, SELF_CENTER_Y),
                        (cx, cy), purple, 2, tipLength=0.2)

        # 终点大圆圈
        cv2.circle(frame, (cx, cy), 10, purple, 2)
        cv2.circle(frame, (cx, cy),  4, purple, -1)

        # 目标世界坐标 + 距离标签
        if route_tgt is not None:
            rx, ry = route_tgt
            lx = max(10, min(fw - 160, cx + 14))
            ly = max(20, min(fh - 10,  cy - 10))
            label = f"#{route_idx} ({rx},{ry}) d:{route_dist}"
            # 黑色描边 + 彩色字
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, purple, 1)

    # ── 路线进度条（底部）──────────────────────────────
    if route_tot > 0 and state in ("PATROL", "STUCK", "IDLE"):
        bar_x, bar_y = 10, fh - 18
        bar_w, bar_h = fw - 20, 10
        pct = route_idx / route_tot

        # 背景
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # 填充
        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), (255, 200, 0), -1)
        # 边框
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)
        # 文字
        prog_text = f"ROUTE {route_idx}/{route_tot}  {pct*100:.0f}%"
        cv2.putText(frame, prog_text, (bar_x + 4, bar_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(frame, prog_text, (bar_x + 4, bar_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_fps(frame, fps):
    _put_label(frame, f"FPS: {fps:.1f}", 10, 25, (0, 255, 255), scale=0.7)


def draw_stats(frame, motion_count, target_count):
    _put_label(frame, f"Targets: {target_count}", 10, 50, (200, 200, 200), scale=0.5)


def _put_label(frame, text, x, y, color, scale=None):
    if scale is None:
        scale = FONT_SCALE
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    cv2.rectangle(frame, (x, y - th - bl), (x + tw, y + bl), COLOR_TEXT_BG, cv2.FILLED)
    cv2.putText(frame, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def draw_grid_overlay(frame, grid_data, size=150):
    """
    左下角画小地图（300×300 → 150×150），显示覆盖情况。
    grid_data 来自 GridNavigator.get_viz_data()
    """
    if grid_data is None:
        return

    visited = grid_data.get("visited", set())
    walls = grid_data.get("walls", set())
    map_size = grid_data.get("map_size", 300)
    world_pos = grid_data.get("world_pos", (0, 0))
    waypoint = grid_data.get("waypoint")
    coverage = grid_data.get("coverage", 0)
    ocr_rate = grid_data.get("ocr_rate", 0)

    # 创建小地图（300→150，每2格1像素）
    scale = size / map_size  # 0.5
    minimap = np.zeros((size, size, 3), dtype=np.uint8)
    minimap[:] = (30, 30, 30)  # 深灰底色 = 未探索

    # 画已走过的格子（绿色）
    for (x, y) in visited:
        px = int(x * scale)
        py = int(y * scale)
        if 0 <= px < size and 0 <= py < size:
            minimap[py, px] = (0, 180, 0)

    # 画墙壁（红色）
    for (x, y) in walls:
        px = int(x * scale)
        py = int(y * scale)
        if 0 <= px < size and 0 <= py < size:
            minimap[py, px] = (0, 0, 150)

    # 画角色位置（白色大点）
    cx = int(world_pos[0] * scale)
    cy = int(world_pos[1] * scale)
    if 0 <= cx < size and 0 <= cy < size:
        cv2.circle(minimap, (cx, cy), 3, (255, 255, 255), -1)

    # 画航点（黄色 x）
    if waypoint:
        wpx = int(waypoint[0] * scale)
        wpy = int(waypoint[1] * scale)
        if 0 <= wpx < size and 0 <= wpy < size:
            cv2.drawMarker(minimap, (wpx, wpy), (0, 255, 255),
                            cv2.MARKER_CROSS, 5, 1)

    # 加边框
    cv2.rectangle(minimap, (0, 0), (size - 1, size - 1), (100, 100, 100), 1)

    # 放到主画面右上角
    fh, fw = frame.shape[:2]
    x_off = fw - size - 10
    y_off = 30

    if y_off + size < fh and x_off > 0:
        frame[y_off:y_off + size, x_off:x_off + size] = minimap

    # 覆盖率文字
    cov_text = f"Cover:{coverage:.1%} OCR:{ocr_rate:.0%} ({world_pos[0]},{world_pos[1]})"
    cv2.putText(frame, cov_text, (x_off, y_off + size + 15),
                FONT, 0.35, (0, 255, 255), 1, cv2.LINE_AA)


def resize_for_display(frame):
    h, w = frame.shape[:2]
    if w <= DISPLAY_MAX_WIDTH and h <= DISPLAY_MAX_HEIGHT:
        return frame
    scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
