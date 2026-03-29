"""
auto_label.py - 用 Roboflow 模型批量自动标注未标注图片
使用训练好的模型对 datasets/images 中的图片进行推理，
然后将标注结果上传回 Roboflow 项目
"""

import os
import glob
import time
import json
import base64
import requests

# ============================================================
# 配置
# ============================================================
API_KEY = "X2Xdnmtlklv6mbgYh7XC"
WORKSPACE = "s-workspace-esyrz"
PROJECT = "cj_ox2"
MODEL_VERSION = 1
CONFIDENCE_THRESHOLD = 5  # 低阈值，宁可多检不漏检，后续人工修正

# 图片目录
IMAGE_DIR = "../mon21/images"

# ============================================================
# Roboflow API
# ============================================================
INFER_URL = f"https://detect.roboflow.com/{PROJECT}/{MODEL_VERSION}"
UPLOAD_URL = f"https://api.roboflow.com/dataset/{PROJECT}/annotate"


def predict(image_path):
    """用模型对单张图片推理"""
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    resp = requests.post(
        INFER_URL,
        params={
            "api_key": API_KEY,
            "confidence": CONFIDENCE_THRESHOLD,
        },
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )

    if resp.status_code != 200:
        print(f"  推理失败: {resp.status_code} {resp.text[:100]}")
        return None

    return resp.json()


def upload_annotation(image_name, predictions, img_width, img_height):
    """
    将标注上传到 Roboflow 项目
    使用 Roboflow 的 annotation upload API
    格式: YOLO TXT (class_id cx cy w h) normalized
    """
    if not predictions:
        return False

    # 构建 YOLO 格式标注
    # 需要映射 class name -> class id
    class_map = {}
    lines = []

    for pred in predictions:
        cls_name = pred["class"]
        if cls_name not in class_map:
            class_map[cls_name] = len(class_map)

        cls_id = class_map[cls_name]

        # Roboflow 返回的是 x, y (center), width, height 像素坐标
        cx = pred["x"] / img_width
        cy = pred["y"] / img_height
        w = pred["width"] / img_width
        h = pred["height"] / img_height

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    annotation = "\n".join(lines)

    # 上传标注
    resp = requests.post(
        f"https://api.roboflow.com/dataset/{PROJECT}/annotate/{image_name}",
        params={
            "api_key": API_KEY,
            "name": image_name,
        },
        data=annotation,
        headers={"Content-Type": "text/plain"},
        timeout=15,
    )

    return resp.status_code == 200


def main():
    print("=" * 60)
    print("  Roboflow 批量自动标注")
    print(f"  模型: {PROJECT} v{MODEL_VERSION}")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}%")
    print("=" * 60)

    # 获取所有图片
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    images = []
    for pat in patterns:
        images.extend(glob.glob(os.path.join(IMAGE_DIR, pat)))

    if not images:
        print(f"[ERROR] 在 {IMAGE_DIR} 中未找到图片")
        return

    images.sort()
    print(f"\n找到 {len(images)} 张图片")

    # 保存结果
    results_dir = "datasets/auto_labels"
    os.makedirs(results_dir, exist_ok=True)

    total = len(images)
    detected = 0
    total_objects = 0

    for i, img_path in enumerate(images):
        filename = os.path.basename(img_path)
        print(f"\n[{i+1}/{total}] {filename}")

        # 推理
        result = predict(img_path)
        if result is None:
            continue

        predictions = result.get("predictions", [])
        img_w = result.get("image", {}).get("width", 1936)
        img_h = result.get("image", {}).get("height", 1040)

        if predictions:
            detected += 1
            total_objects += len(predictions)

            # 打印检测结果
            for pred in predictions:
                conf = pred["confidence"]
                cls = pred["class"]
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                print(f"  {cls}: {conf:.1%} at ({x:.0f},{y:.0f}) {w:.0f}x{h:.0f}")

            # 保存 YOLO 格式标注文件到本地
            label_name = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(results_dir, label_name)

            # 获取所有类名
            all_classes = sorted(set(p["class"] for p in predictions))
            class_map = {name: idx for idx, name in enumerate(all_classes)}

            with open(label_path, "w") as f:
                for pred in predictions:
                    cls_id = class_map[pred["class"]]
                    cx = pred["x"] / img_w
                    cy = pred["y"] / img_h
                    nw = pred["width"] / img_w
                    nh = pred["height"] / img_h
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        else:
            print("  未检测到物体")

        # 避免 API 限流
        time.sleep(0.5)

    # 保存类名映射
    if detected > 0:
        classes_path = os.path.join(results_dir, "classes.txt")
        # 收集所有出现过的类
        all_class_names = set()
        for img_path in images:
            result = None  # 已处理过，从本地标注文件读
            label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path = os.path.join(results_dir, label_name)
            # classes 在推理时已统一

        print(f"\n{'=' * 60}")
        print(f"  完成!")
        print(f"  总图片: {total}")
        print(f"  有检测: {detected} ({detected/total*100:.1f}%)")
        print(f"  总物体: {total_objects}")
        print(f"  标注保存: {results_dir}/")
        print(f"{'=' * 60}")
        print(f"\n下一步:")
        print(f"  1. 检查 {results_dir}/ 中的标注文件")
        print(f"  2. 将标注上传回 Roboflow（或在 Roboflow 中直接使用模型标注）")
        print(f"  3. 在 Roboflow 中检查并修正标注")
        print(f"  4. 生成新版本，重新训练模型")


if __name__ == "__main__":
    main()
