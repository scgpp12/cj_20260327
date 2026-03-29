# -*- coding: utf-8 -*-
"""BFS 调试：模拟 grid_navigator 的 BFS 查找"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from grid_navigator import GridNavigator

def main():
    nav = GridNavigator()

    # 模拟 OCR 读到坐标 (261, 230)
    test_positions = [(261, 230), (263, 232), (265, 234)]

    print("=== 初始状态 ===")
    print(f"visited: {len(nav.grid.visited)}, walls: {len(nav.grid.walls)}")

    # 标记几个位置为已走
    for x, y in test_positions:
        nav.grid.mark_visited(x, y)
    print(f"标记 {len(test_positions)} 个位置后: visited={len(nav.grid.visited)}")

    # 测试 BFS
    nav.world_x, nav.world_y = 261, 230
    target = nav.planner.find_next_target(261, 230)
    print(f"BFS 结果: {target}")

    # 模拟 A* 失败标墙
    print("\n=== 模拟 A* 失败标墙 ===")
    for direction in ["RIGHT", "DOWN_RIGHT", "UP_RIGHT"]:
        nav.on_direction_failed(direction)
    print(f"标墙后: walls={len(nav.grid.walls)}")

    # 再测试 BFS
    target = nav.planner.find_next_target(261, 230)
    print(f"BFS 结果: {target}")

    # 更多标墙
    print("\n=== 大量标墙 ===")
    for direction in ["LEFT", "UP", "DOWN", "UP_LEFT", "DOWN_LEFT"]:
        nav.on_direction_failed(direction)
    print(f"标墙后: walls={len(nav.grid.walls)}")

    target = nav.planner.find_next_target(261, 230)
    print(f"BFS 结果: {target}")

    # 检查周围8格是否都被墙堵了
    print("\n=== 检查 (261,230) 周围 ===")
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            nx, ny = 261+dx, 230+dy
            v = "V" if nav.grid.is_visited(nx, ny) else "."
            w = "W" if not nav.grid.is_walkable(nx, ny) else "."
            if v != "." or w != ".":
                print(f"  ({nx},{ny}): visited={v} wall={w}")

if __name__ == "__main__":
    main()
