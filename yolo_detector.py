"""
yolo_detector.py - 本地 YOLOv8 怪物检测
使用 ultralytics 推理，替代传统 CV 血条检测
"""

import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    """
    本地 YOLO 怪物检测器

    返回格式与 hp_detector 兼容:
    [
        {
            "class": "OXDemo",          # 类名
            "class_id": 1,              # 类 ID
            "confidence": 0.85,         # 置信度
            "bbox": (x, y, w, h),       # 边界框（左上角 + 宽高）
            "center": (cx, cy),         # 中心点
            "target_box": (x, y, w, h), # 兼容攻击系统的目标框
            "hp_box": (x, y, w, h),     # 兼容可视化的 hp_box（用 bbox 替代）
            "source": "yolo",           # 检测来源标识
        },
        ...
    ]
    """

    def __init__(self, model_path, confidence=0.25, iou_threshold=0.45, device=None):
        """
        Args:
            model_path: .pt 模型文件路径
            confidence: 最低置信度阈值
            iou_threshold: NMS 的 IoU 阈值
            device: 推理设备 (None=自动, 'cpu', '0'=GPU0)
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        print(f"[YOLO] 加载模型: {model_path}")
        self.model = YOLO(model_path)

        # 预热一次推理
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, conf=0.5)

        self.class_names = self.model.names  # {0: 'OX03', 1: 'OXDemo', ...}
        print(f"[YOLO] 类别: {self.class_names}")
        print(f"[YOLO] 置信度: {self.confidence}, IoU: {self.iou_threshold}")

    def detect(self, frame):
        """
        对一帧图像进行 YOLO 推理

        Args:
            frame: BGR 图像 (numpy array)

        Returns:
            list of dict: 检测结果列表
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []

        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            # xyxy 格式 → xywh 格式
            coords = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = self.class_names.get(cls_id, str(cls_id))

            w = x2 - x1
            h = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2

            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": conf,
                "bbox": (x1, y1, w, h),
                "center": (cx, cy),
                "target_box": (x1, y1, w, h),
                "hp_box": (x1, y1, w, h),
                "source": "yolo",
            })

        return detections

    def detect_monsters(self, frame, monster_classes=None):
        """
        只返回怪物（过滤掉 self 等非怪物类）

        Args:
            frame: BGR 图像
            monster_classes: 怪物类名列表，默认 ['OXDemo']

        Returns:
            list of dict: 怪物检测结果
        """
        if monster_classes is None:
            monster_classes = ["OXDemo"]

        all_dets = self.detect(frame)
        monsters = [d for d in all_dets if d["class"] in monster_classes]
        # self 模型已取消，不需要过滤自身
        return monsters

    def detect_self(self, frame, self_classes=None):
        """
        只返回自身角色检测

        Args:
            frame: BGR 图像
            self_classes: 角色类名列表，默认 ['OX03']

        Returns:
            list of dict: 角色检测结果
        """
        if self_classes is None:
            self_classes = ["OX03"]

        all_dets = self.detect(frame)
        return [d for d in all_dets if d["class"] in self_classes]
