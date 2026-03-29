# Game Motion HP Detector - 技术逻辑说明书

## 项目概述
这是一个完整的游戏自动化辅助系统，用于 2D/2.5D 传奇类游戏。系统实现了怪物检测、自动攻击、自主巡逻、物品拾取、自动喝药等功能。

---

## 1. 系统架构

```
main.py (主循环 ~12 FPS)
├── 截屏 → screen_capture.py
├── 检测管线（三选一）：
│   ├── 红球检测 → redball_detector.py （当前启用）
│   ├── YOLO 检测 → yolo_detector.py （已关闭）
│   └── 传统 CV 血条 → hp_detector.py （备选）
├── 稳定性过滤 → target_tracker.py （连续3帧确认）
├── 战斗系统 → action_controller.py
├── 巡逻系统
│   ├── 导航 → patrol_controller.py
│   ├── A* 寻路 → pathfinder.py
│   └── 网格覆盖 → grid_navigator.py
├── 物品拾取 → item_picker.py
├── 自动喝药 → potion_manager.py
├── 音频检测 → audio_detector.py
└── 可视化 → visualizer.py
```

---

## 2. 优先级体系

```
攻击范围内的怪(0~300px) > 拾取物品(350px) > 朝怪物方向巡逻 > 自由巡逻
```

---

## 3. 每帧主循环流程

```
第1步: 截屏
  mss 截取游戏窗口 → frame (1936x1040 BGR)

第2步: 检测怪物
  红球检测: HSV 红色阈值 + 圆形度过滤
  输出: 所有检测到的怪物框 [(x,y,w,h), ...]

第3步: 稳定性过滤 (target_tracker.py)
  当前帧检测结果 → 和上一帧匹配（位置距离<60px=同一目标）
  连续3帧检测到 → 标记为"稳定目标"（可攻击）
  1-2帧 → 灰色虚框显示，不攻击
  消失2帧以上 → 删除

第4步: 距离过滤
  只保留 0~300px 范围内的稳定目标 → in_range

第5步: 攻击 or 巡逻
  if in_range 有目标:
    → 攻击器接管
    → 巡逻暂停
  else:
    → 检查物品拾取
    → 如果没有物品 → 巡逻器接管

第6步: 自动喝药
  每帧检测血球颜色
  HP < 60% → 按键2
  MP < 30% → 按键3

第7步: 显示
  叠加所有框体、路径、状态信息 → cv2.imshow
```

---

## 4. 攻击控制器 (action_controller.py)

### 状态机

```
IDLE（空闲）
  │ 发现范围内怪物
  ▼
BURST（连击锁敌）
  │ 快速左键 2 下（间隔 0.1 秒）
  │ 每一下追踪目标最新位置
  ▼
WAITING（等待自动攻击）
  │
  ├─ 有声音 + 画面在动 → 继续等待（角色在自动攻击）
  │
  ├─ 有更近的怪（锁定>80px 且新怪距离<50%）→ 切换目标 → BURST
  │   └─ 切换冷却 3 秒，防止来回切
  │
  ├─ 2 秒无声音，目标仍在 → 重新连击 → BURST
  │
  ├─ 目标消失 1 秒 → 怪死了 → IDLE
  │
  └─ 10 秒硬超时 → 强制重新连击 → BURST
```

### 关键参数
| 参数 | 值 | 说明 |
|------|-----|------|
| BURST_CLICKS | 2 | 连击次数 |
| BURST_INTERVAL | 0.1s | 每下间隔 |
| WAIT_CHECK_INTERVAL | 2.0s | 无声音多久重新连击 |
| WAIT_ABSOLUTE_MAX | 10.0s | 绝对超时 |
| TARGET_GONE_CONFIRM | 1.0s | 目标消失多久确认死亡 |
| SWITCH_COOLDOWN | 3.0s | 切换目标冷却 |

### 攻击方式
- PostMessage 发送 Ctrl+左键（WM_KEYDOWN Ctrl → WM_LBUTTONDOWN → WM_LBUTTONUP → WM_KEYUP Ctrl）
- 点击坐标 = 怪物检测框中心
- 每下连击都获取目标最新位置（怪物在移动）

---

## 5. 巡逻控制器 (patrol_controller.py)

### 状态机

```
IDLE（等待）
  │ 1秒无怪物
  ▼
PATROL（巡逻中）
  │
  ├─ 画面1.5秒不变 → STUCK（撞墙）
  │   ├─ 朝前方盲点3下左键（打隐形怪）
  │   ├─ A* 重新规划
  │   └─ 换方向 → 回到 PATROL
  │
  ├─ 发现怪物 → COMBAT（攻击器接管）
  │
  └─ 每2.5秒执行一次移动
      ├─ A*MONSTER: 有远处怪物 → A* 规划朝怪物方向的路径
      ├─ A*PATROL: 无远处怪物 → A* 规划网格覆盖方向
      └─ FALLBACK: A* 失败 → 选最亮方向直线走
```

### 方向选择评分

```
8个方向打分:
  亮度分 (35-50%) — 地面亮=可走，暗=墙壁
  新鲜度 (15-30%) — 没走过的方向优先
  防回头 (20%)    — 最近走过的方向及反方向扣分
  怪物加分 (30%)  — 远处有怪的方向加分（仅有怪物时）
```

### 移动方式
- 右键长按 = 跑步（PostMessage WM_RBUTTONDOWN，每150ms重发保持长按）
- 发现怪物 → 释放右键停步

---

## 6. 红球检测 (redball_detector.py)

### 算法

```
1. BGR → HSV
2. 双范围红色掩码:
   H=0-10 (红色低段) OR H=170-180 (红色高段)
   S ≥ 150, V ≥ 50
3. 形态学闭合 → 连接碎片
4. 找轮廓
5. 过滤:
   面积: 300-15000 px²
   圆形度: ≥ 0.5
6. 排除 UI 区域（背包、底栏、小地图）
7. 排除角色血球位置
```

### 输出
```python
{
    "box": (x, y, w, h),       # 边界框
    "center": (cx, cy),         # 中心点
    "circularity": 0.75,        # 圆形度
    "dist": 123.4,              # 距离角色中心
}
```

---

## 7. 物品拾取 (item_picker.py)

### 检测方式
HSV 紫色检测（物品模型已替换为紫色球体）:
- H=110-160, S≥100, V≥50
- 圆形度 ≥ 0.3
- 面积 150-5000 px²

### 状态机

```
IDLE（扫描物品）
  │ 检测到紫色球体
  ▼
WALKING（走向物品）
  │ 画面变化 → 重新检测物品位置 → 点击新坐标
  │ 5秒超时 → 跳过该物品
  ▼
物品消失 → 拾取成功 → IDLE
```

### 跳过列表
- 捡不起来的物品记住 30 秒
- 50px 半径内不再尝试

---

## 8. 目标追踪器 (target_tracker.py)

### 算法

```
每帧:
  1. 当前检测结果 ↔ 已追踪目标列表
  2. 贪心匹配（位置距离 < 60px = 同一个）
  3. 匹配成功 → frames_seen += 1
  4. 连续 3 帧 → 标记为 stable（可攻击）
  5. 消失 → frames_lost += 1
  6. 消失 > 2 帧 → 删除
  7. 只返回 stable 目标给攻击系统
```

### 效果
- 1-2 帧的闪烁检测不会触发攻击
- 短暂遮挡（1-2帧）不会丢失目标
- 防止帧间跳动导致来回切换

---

## 9. A* 寻路 (pathfinder.py)

### 算法

```
1. 当前帧灰度图 → 16px 网格
2. 亮度 < 24 = 深渊核心
3. 深渊向外扩展 8 格 = 墙壁安全距离
4. A* 搜索: 角色中心 → 目标点
   - 8方向移动（对角线成本1.414，直线1.0）
   - Chebyshev 距离启发式
   - 最多搜索 10000 个格子
5. 路径简化 → 拐点列表
6. 返回 [(px1,py1), (px2,py2), ...] 画面坐标
```

### 可视化
- 半透明红色 = 墙壁/不可走区域
- 黄色线 + 橙色圆点 = A* 路径 + 拐点
- 绿色圆点 = 目标点 (GOAL)

---

## 10. 自动喝药 (potion_manager.py)

### 检测方式

```
血球位置: (92, 916) 半径 51
左半圆 = HP（红色）
右半圆 = MP（蓝色）

判断方法: RGB 通道比较
  红色像素: R > G×1.3 且 R > B×1.3
  蓝色像素: B > R×1.3 且 B > G×1.3

比例 = 有色像素 / 总像素 × 校准系数(1/0.83)
```

### 触发条件
| 条件 | 按键 | 冷却 |
|------|------|------|
| HP < 60% | 数字键 2 | 1.5 秒 |
| MP < 30% | 数字键 3 | 1.5 秒 |

### 防刷屏
- 连续 5 次喝药无效（HP/MP 没涨 5%）→ 暂停 30 秒

---

## 11. 音频检测 (audio_detector.py)

### 用途
检测攻击命中的声音 → 告诉攻击控制器"角色在自动攻击中"

### 工作原理

```
1. 预录攻击音效指纹（assets/attack_fingerprint.npz）
2. 后台线程持续录制游戏音频（WASAPI Loopback）
3. 滑动窗口 FFT 频谱特征提取
4. Pearson 相关系数匹配
5. 相关度 ≥ 0.65 = 检测到攻击声
```

### 设备
- 自动选择第一个 WASAPI Loopback 设备
- 当前: JBL Pulse 4 蓝牙音箱

---

## 12. 网格导航 (grid_navigator.py)

### 用途
扫地机器人模式，确保走遍所有区域

### 组件

```
GridMap: 动态网格 {(gx,gy): 状态}
  状态: UNKNOWN / OPEN / VISITED / WALL

PositionTracker: 相位相关估算世界坐标
  cv2.phaseCorrelate(prev, curr) → (dx, dy)
  累加 → 估算角色在世界中的位置

CoveragePlanner: BFS 前沿搜索
  从当前位置 BFS → 找最近的 UNKNOWN 格子
  返回方向 → patrol 朝那个方向走
```

---

## 13. 所有操作方式（PostMessage）

### 攻击
```python
PostMessage(hwnd, WM_KEYDOWN, VK_CONTROL, 0)    # 按下 Ctrl
PostMessage(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)  # 左键按下
PostMessage(hwnd, WM_LBUTTONUP, 0, lparam)       # 左键松开
PostMessage(hwnd, WM_KEYUP, VK_CONTROL, 0)       # 松开 Ctrl
```

### 移动（跑步）
```python
PostMessage(hwnd, WM_RBUTTONDOWN, MK_RBUTTON, lparam)  # 右键按下（持续）
# 每 150ms 重发一次保持长按
PostMessage(hwnd, WM_RBUTTONUP, 0, lparam)  # 右键松开（停步）
```

### 喝药
```python
PostMessage(hwnd, WM_KEYDOWN, VK_KEY, 0)   # 按下数字键
PostMessage(hwnd, WM_KEYUP, VK_KEY, 0)     # 松开
```

### 拾取
```python
PostMessage(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)  # 左键点击物品
PostMessage(hwnd, WM_LBUTTONUP, 0, lparam)
```

---

## 14. 快捷键

| 按键 | 功能 |
|------|------|
| q | 退出 |
| p | 暂停/继续 |
| a | 攻击开关 |
| r | 巡逻开关 |
| i | 拾取开关 |
| y | 切换 YOLO/传统CV |
| d | HSV 调试模式 |
| s | 保存截图 |
| 滚轮 | 画面缩放 |
| 中键拖拽 | 画面平移 |

---

## 15. 已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| 巡逻撞墙循环 | 部分修复 | A* 重新规划，但有时仍循环 |
| 红球误检墙壁纹理 | 已修复 | 圆形度≥0.5 + 饱和度≥150 |
| 物品拾取走不到位 | 测试中 | 持续点击模式 |
| 音频指纹格式兼容 | 已修复 | 支持新旧格式 |
| 怪物太近检测不到 | 已修复 | 取消 self 排除 |

---

## 16. 配置文件 (config.py) 关键参数

```python
# 角色位置
SELF_CENTER_X = 965
SELF_CENTER_Y = 454

# 红球检测
REDBALL_ENABLED = True
REDBALL_MIN_AREA = 300
REDBALL_MAX_AREA = 15000
REDBALL_MIN_CIRCULARITY = 0.5

# 攻击
AUTO_ATTACK_ENABLED = True
AUTO_ATTACK_COOLDOWN = 1.5

# 巡逻
PATROL_ENABLED = True
PATROL_IDLE_TIMEOUT = 1.0
PATROL_MOVE_INTERVAL = 2.5
PATROL_CLICK_DISTANCE = 200
PATROL_STUCK_TIMEOUT = 1.5
PATROL_DARK_THRESHOLD = 24

# 喝药
POTION_ENABLED = True
POTION_HP_THRESHOLD = 0.60
POTION_MP_THRESHOLD = 0.30

# 拾取
PICK_ENABLED = True
PICK_RANGE = 350
PICK_ARRIVE_DIST = 5

# A* 寻路
ASTAR_GRID_STEP = 16
ASTAR_DARK_THRESH = 24
ASTAR_WALL_EXPAND = 8
```
