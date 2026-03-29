"""
test_pathfind.py - A* 寻路测试工具
截一张游戏画面，鼠标点击目标位置，实时显示寻路结果。

操作：
  鼠标左键点击 = 设置目标点，显示 A* 路径
  r = 重新截屏
  q = 退出
"""

import cv2
import numpy as np
import heapq
import time
from screen_capture import ScreenCapture, list_and_select_window
from config import CAPTURE_WINDOW_TITLE, SELF_CENTER_X, SELF_CENTER_Y, PATROL_DARK_THRESHOLD

# A* 寻路网格参数
GRID_STEP = 16       # 网格步长（像素），越小越精细但越慢
DARK_THRESH = 15     # 亮度低于此值 = 不可走

# 8 方向移动
MOVES = [(-1, -1), (0, -1), (1, -1),
         (-1, 0),           (1, 0),
         (-1, 1),  (0, 1),  (1, 1)]
MOVE_COSTS = [1.414, 1.0, 1.414,
              1.0,        1.0,
              1.414, 1.0, 1.414]


def frame_to_grid(frame, step=GRID_STEP, threshold=DARK_THRESH, expand=3):
    """
    将画面转换为可走性网格。
    策略：bright < threshold 的格子为深渊核心，向外扩展 expand 格作为墙壁。

    Args:
        expand: 深渊向外扩展的格数（深渊边缘 = 墙壁）

    Returns:
        grid: 2D numpy array, 1=可走, 0=墙
        (grid_h, grid_w): 网格尺寸
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    grid_h = h // step
    grid_w = w // step

    # 先按亮度标记深渊核心
    raw_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for gy in range(grid_h):
        for gx in range(grid_w):
            py = gy * step
            px = gx * step
            region = gray[py:py + step, px:px + step]
            brightness = region.mean()
            raw_grid[gy, gx] = 1 if brightness >= threshold else 0

    # 深渊区域膨胀 expand 格 → 边缘也变成墙
    if expand > 0:
        # 0=深渊 → 膨胀后更多0 → 反转操作
        wall_mask = (raw_grid == 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (expand * 2 + 1, expand * 2 + 1))
        expanded_wall = cv2.dilate(wall_mask, kernel, iterations=1)
        grid = (expanded_wall == 0).astype(np.uint8)
    else:
        grid = raw_grid

    return grid, (grid_h, grid_w)


def astar(grid, start, goal):
    """
    A* 寻路。
    start, goal: (gx, gy) 网格坐标
    Returns: 路径列表 [(gx, gy), ...] 或 None
    """
    grid_h, grid_w = grid.shape
    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < grid_w and 0 <= sy < grid_h):
        return None
    if not (0 <= gx < grid_w and 0 <= gy < grid_h):
        return None
    if grid[sy, sx] == 0 or grid[gy, gx] == 0:
        return None

    # 启发函数：对角距离
    def heuristic(x, y):
        dx = abs(x - gx)
        dy = abs(y - gy)
        return max(dx, dy) + 0.414 * min(dx, dy)

    open_set = []
    heapq.heappush(open_set, (heuristic(sx, sy), 0, sx, sy))
    came_from = {}
    g_score = {(sx, sy): 0}
    visited = set()

    while open_set:
        _, cost, cx, cy = heapq.heappop(open_set)

        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))

        if cx == gx and cy == gy:
            # 回溯路径
            path = [(gx, gy)]
            while (path[-1][0], path[-1][1]) != (sx, sy):
                path.append(came_from[(path[-1][0], path[-1][1])])
            path.reverse()
            return path

        for i, (dx, dy) in enumerate(MOVES):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h and grid[ny, nx] == 1:
                new_cost = cost + MOVE_COSTS[i]
                if (nx, ny) not in g_score or new_cost < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = new_cost
                    f = new_cost + heuristic(nx, ny)
                    heapq.heappush(open_set, (f, new_cost, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)

    return None  # 无路径


def simplify_path(path, min_angle=15):
    """
    简化路径：去掉直线上的中间点，只保留拐点。
    """
    if not path or len(path) <= 2:
        return path

    import math
    simplified = [path[0]]

    for i in range(1, len(path) - 1):
        px, py = path[i - 1]
        cx, cy = path[i]
        nx, ny = path[i + 1]

        # 计算方向变化
        a1 = math.atan2(cy - py, cx - px)
        a2 = math.atan2(ny - cy, nx - cx)
        angle_diff = abs(math.degrees(a2 - a1))
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff >= min_angle:
            simplified.append(path[i])

    simplified.append(path[-1])
    return simplified


def draw_path(frame, path, step, color=(0, 255, 255), thickness=2):
    """在画面上画寻路路径"""
    if not path or len(path) < 2:
        return

    for i in range(len(path) - 1):
        x1 = path[i][0] * step + step // 2
        y1 = path[i][1] * step + step // 2
        x2 = path[i + 1][0] * step + step // 2
        y2 = path[i + 1][1] * step + step // 2
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

    # 画拐点
    for px, py in path:
        x = px * step + step // 2
        y = py * step + step // 2
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def draw_grid_overlay(frame, grid, step, alpha=0.3):
    """半透明显示可走/不可走区域"""
    overlay = frame.copy()
    grid_h, grid_w = grid.shape

    for gy in range(grid_h):
        for gx in range(grid_w):
            if grid[gy, gx] == 0:  # 墙
                px = gx * step
                py = gy * step
                cv2.rectangle(overlay, (px, py), (px + step, py + step),
                              (0, 0, 150), cv2.FILLED)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# 鼠标状态
mouse_target = [None]
mouse_pos = [0, 0]


def mouse_cb(event, x, y, flags, param):
    scale = param.get("scale", 1.0)
    ox = int(x / scale)
    oy = int(y / scale)
    mouse_pos[0] = ox
    mouse_pos[1] = oy

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_target[0] = (ox, oy)


def main():
    print("=" * 50)
    print("  A* 寻路测试工具")
    print("  左键点击 = 设置目标，显示路径")
    print("  r = 重新截屏, q = 退出")
    print("=" * 50)

    game_window = list_and_select_window(title_hint=CAPTURE_WINDOW_TITLE)
    if game_window is None:
        return

    capture = ScreenCapture(window_obj=game_window)

    frame = capture.grab()
    if frame is None:
        print("截屏失败")
        return

    # 构建网格
    print(f"画面: {frame.shape[1]}x{frame.shape[0]}")
    print(f"网格步长: {GRID_STEP}px")

    t0 = time.time()
    grid, (grid_h, grid_w) = frame_to_grid(frame)
    print(f"网格大小: {grid_w}x{grid_h} ({time.time()-t0:.3f}s)")
    walkable = np.count_nonzero(grid)
    total = grid_h * grid_w
    print(f"可走格子: {walkable}/{total} ({walkable/total*100:.0f}%)")

    # 角色网格坐标
    start_gx = SELF_CENTER_X // GRID_STEP
    start_gy = SELF_CENTER_Y // GRID_STEP
    print(f"角色位置: ({SELF_CENTER_X},{SELF_CENTER_Y}) → 网格({start_gx},{start_gy})")

    # 显示
    fh, fw = frame.shape[:2]
    scale = min(1280 / fw, 720 / fh, 1.0)
    dw, dh = int(fw * scale), int(fh * scale)

    win_name = "A* Pathfinding Test"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_cb, {"scale": scale})

    last_path = None
    last_simplified = None
    show_grid = True
    current_thresh = 10       # 深渊亮度阈值（低于此为纯黑深渊）
    current_expand = 3        # 深渊向外扩展格数

    print(f"\n快捷键:")
    print(f"  +/- 调阈值(深渊亮度)")
    print(f"  [/] 调扩展范围(深渊边缘宽度)")
    print(f"  g=网格显示, r=重截, q=退出")
    print(f"当前: 阈值={current_thresh}, 扩展={current_expand}格\n")

    while True:
        display = frame.copy()

        # 画网格障碍
        if show_grid:
            draw_grid_overlay(display, grid, GRID_STEP, alpha=0.2)

        # 画角色位置
        cv2.circle(display, (SELF_CENTER_X, SELF_CENTER_Y), 8, (0, 255, 255), 2)
        cv2.putText(display, "SELF", (SELF_CENTER_X - 15, SELF_CENTER_Y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 如果有目标点，运行 A*
        if mouse_target[0] is not None:
            tx, ty = mouse_target[0]
            tgx = tx // GRID_STEP
            tgy = ty // GRID_STEP

            # 画目标
            cv2.circle(display, (tx, ty), 8, (0, 0, 255), 2)
            cv2.putText(display, "TARGET", (tx - 20, ty - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # A* 寻路
            t0 = time.time()
            path = astar(grid, (start_gx, start_gy), (tgx, tgy))
            elapsed = time.time() - t0

            if path:
                # 画完整路径（细线）
                draw_path(display, path, GRID_STEP, color=(100, 100, 100), thickness=1)

                # 简化路径（粗线 + 拐点）
                simplified = simplify_path(path)
                draw_path(display, simplified, GRID_STEP, color=(0, 255, 255), thickness=3)

                info = f"Path: {len(path)} steps -> {len(simplified)} waypoints ({elapsed*1000:.1f}ms)"
                last_path = path
                last_simplified = simplified
            else:
                info = f"No path found! ({elapsed*1000:.1f}ms)"

            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 鼠标位置信息 + 亮度
        mx, my = mouse_pos
        mgx, mgy = mx // GRID_STEP, my // GRID_STEP
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if 0 <= my < fh and 0 <= mx < fw:
            brightness = int(gray_frame[my, mx])
        else:
            brightness = 0
        if 0 <= mgy < grid_h and 0 <= mgx < grid_w:
            cell = "WALK" if grid[mgy, mgx] == 1 else "WALL"
            info_text = f"({mx},{my}) bright={brightness} {cell} | thresh={current_thresh}(+/-) expand={current_expand}([/])"
            # 黑色描边 + 红色粗体
            cv2.putText(display, info_text, (10, fh - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(display, info_text, (10, fh - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示
        show = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_AREA)
        cv2.imshow(win_name, show)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 重新截屏
            frame = capture.grab()
            if frame is not None:
                grid, _ = frame_to_grid(frame)
                mouse_target[0] = None
                print("重新截屏完成")
        elif key == ord('g'):
            show_grid = not show_grid
            print(f"网格显示: {'ON' if show_grid else 'OFF'}")
        elif key in (ord('+'), ord('=')):
            current_thresh += 2
            grid, _ = frame_to_grid(frame, GRID_STEP, current_thresh, current_expand)
            mouse_target[0] = None
            walkable = np.count_nonzero(grid)
            print(f"阈值: {current_thresh}, 扩展: {current_expand} (可走: {walkable}/{total} {walkable*100//total}%)")
        elif key in (ord('-'), ord('_')):
            current_thresh = max(2, current_thresh - 2)
            grid, _ = frame_to_grid(frame, GRID_STEP, current_thresh, current_expand)
            mouse_target[0] = None
            walkable = np.count_nonzero(grid)
            print(f"阈值: {current_thresh}, 扩展: {current_expand} (可走: {walkable}/{total} {walkable*100//total}%)")
        elif key == ord(']'):
            current_expand += 1
            grid, _ = frame_to_grid(frame, GRID_STEP, current_thresh, current_expand)
            mouse_target[0] = None
            walkable = np.count_nonzero(grid)
            print(f"阈值: {current_thresh}, 扩展: {current_expand} (可走: {walkable}/{total} {walkable*100//total}%)")
        elif key == ord('['):
            current_expand = max(0, current_expand - 1)
            grid, _ = frame_to_grid(frame, GRID_STEP, current_thresh, current_expand)
            mouse_target[0] = None
            walkable = np.count_nonzero(grid)
            print(f"阈值: {current_thresh}, 扩展: {current_expand} (可走: {walkable}/{total} {walkable*100//total}%)")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
