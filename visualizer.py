"""
visualizer.py - 调试可视化模块 (ver04 精简版)
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


def draw_pathfinder_overlay(frame, pathfinder):
    """
    绘制 A* 寻路的完整可视化：
    1. 墙壁红色覆盖
    2. 路径黄色线
    3. 拐点红色标记 + 编号
    """
    if pathfinder is None:
        return

    # ---- 1. 墙壁覆盖 ----
    grid = pathfinder._grid
    if grid is not None:
        step = pathfinder.grid_step
        overlay = frame.copy()
        grid_h, grid_w = grid.shape

        for gy in range(grid_h):
            for gx in range(grid_w):
                if grid[gy, gx] == 0:  # 墙
                    px = gx * step
                    py = gy * step
                    cv2.rectangle(overlay, (px, py), (px + step, py + step),
                                  (0, 0, 150), cv2.FILLED)

        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # ---- 2. 路径线 + 拐点 ----
    waypoints = pathfinder._last_waypoints
    if waypoints and len(waypoints) > 0:
        from config import SELF_CENTER_X, SELF_CENTER_Y

        # 完整路线：角色 → 拐点1 → 拐点2 → ... → 目标
        pts = [(SELF_CENTER_X, SELF_CENTER_Y)] + list(waypoints)

        # 画路线（黄色粗线）
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 255, 255), 3)

        # 画拐点（红色圆 + 编号）
        for idx, (wx, wy) in enumerate(waypoints):
            # 红色圆圈
            cv2.circle(frame, (wx, wy), 8, (0, 0, 255), 2)
            cv2.circle(frame, (wx, wy), 3, (0, 0, 255), -1)
            # 编号
            label = f"W{idx + 1}"
            cv2.putText(frame, label, (wx + 10, wy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, label, (wx + 10, wy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 最终目标标记（绿色大圆）
        last = waypoints[-1]
        cv2.circle(frame, last, 12, (0, 255, 0), 2)
        cv2.putText(frame, "GOAL", (last[0] + 15, last[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def draw_patrol_info(frame, patrol_info):
    """巡逻状态 + 地形雷达"""
    state = patrol_info.get("state", "IDLE")
    direction = patrol_info.get("direction", "")
    click_pos = patrol_info.get("click_pos")
    terrain = patrol_info.get("terrain")

    state_colors = {
        "IDLE": (150, 150, 150),
        "PATROL": (255, 200, 0),
        "STUCK": (0, 0, 255),
        "COMBAT": (0, 255, 0),
    }
    color = state_colors.get(state, (200, 200, 200))

    fh, fw = frame.shape[:2]
    text = f"STATE: {state}"
    if state == "PATROL":
        text += f" [{direction}]"
    _put_label(frame, text, fw - 250, 25, color, scale=0.6)

    try:
        from patrol_controller import DIRECTIONS, PATROL_DARK_THRESHOLD
    except ImportError:
        return

    if terrain and state in ("PATROL", "IDLE", "STUCK"):
        for d_name, score in terrain.items():
            dx, dy = DIRECTIONS[d_name]
            line_len = int(25 * min(score / 150.0, 1.0)) + 5
            ex = SELF_CENTER_X + int(dx * line_len)
            ey = SELF_CENTER_Y + int(dy * line_len)
            if score < PATROL_DARK_THRESHOLD:
                lc = (0, 0, 180)
            else:
                g = int(min(score / 150.0, 1.0) * 255)
                lc = (0, g, 0)
            cv2.line(frame, (SELF_CENTER_X, SELF_CENTER_Y), (ex, ey), lc, 2)

    if click_pos is not None and state in ("PATROL", "STUCK"):
        cx, cy = click_pos
        purple = (180, 0, 255)  # 紫色
        cv2.arrowedLine(frame, (SELF_CENTER_X, SELF_CENTER_Y), (cx, cy), purple, 2, tipLength=0.25)
        cv2.circle(frame, (cx, cy), 8, purple, -1)  # 实心紫色圆点

    # 已走格子重心：黄色五角星
    centroid_screen = patrol_info.get("visited_centroid_screen")
    if centroid_screen is not None:
        import numpy as np
        cx, cy = centroid_screen
        # 限制在屏幕范围内
        cx = max(20, min(fw - 20, cx))
        cy = max(20, min(fh - 20, cy))
        # 画五角星
        r_out, r_in = 15, 6
        pts = []
        for i in range(10):
            angle = np.radians(-90 + i * 36)
            r = r_out if i % 2 == 0 else r_in
            pts.append((int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))))
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(frame, [pts_arr], (0, 255, 255))  # 黄色填充
        cv2.polylines(frame, [pts_arr], True, (0, 180, 180), 2)  # 边框
        _put_label(frame, "CENTROID", cx + 18, cy - 5, (0, 255, 255), scale=0.4)


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
