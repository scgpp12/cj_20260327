"""
upload_labels.py - 推理 + 正确类名映射 + 上传到 Roboflow
一步到位：对每张图推理，生成正确的 YOLO 标注，上传图片+标注
"""

import os
import time
import base64
import requests
from roboflow import Roboflow

# ============================================================
API_KEY = "X2Xdnmtlklv6mbgYh7XC"
WORKSPACE = "s-workspace-esyrz"
PROJECT = "cj_ox2"
MODEL_VERSION = 1
CONFIDENCE = 5  # 低阈值，宁多勿漏

IMG_DIR = "../mon21/images"
LABEL_DIR = "datasets/auto_labels_v2"

# 全局类名（按字母排序）
CLASSES = ["OxDemonGeneral", "OxDemonGuard", "OxDemonKing",
           "OxDemonPriest", "OxDemonWarrior", "OxMage", "self"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
# ============================================================


def predict(img_path):
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        f"https://detect.roboflow.com/{PROJECT}/{MODEL_VERSION}",
        params={"api_key": API_KEY, "confidence": CONFIDENCE},
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()
    return None


def make_yolo_label(predictions, img_w, img_h):
    """生成 YOLO 格式标注（全局统一 class ID）"""
    lines = []
    for p in predictions:
        cls_name = p["class"]
        if cls_name not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[cls_name]
        cx = p["x"] / img_w
        cy = p["y"] / img_h
        w = p["width"] / img_w
        h = p["height"] / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def main():
    os.makedirs(LABEL_DIR, exist_ok=True)

    # 写 classes.txt
    with open(os.path.join(LABEL_DIR, "classes.txt"), "w") as f:
        for c in CLASSES:
            f.write(c + "\n")

    # 连接 Roboflow
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
    print(f"总图片: {len(images)}")

    uploaded = 0
    skipped = 0

    for i, img_name in enumerate(images):
        img_path = os.path.join(IMG_DIR, img_name)
        label_name = img_name.replace(".png", ".txt")
        label_path = os.path.join(LABEL_DIR, label_name)

        print(f"[{i+1}/{len(images)}] {img_name}", end=" ")

        # 推理
        result = predict(img_path)
        if result is None:
            print("推理失败")
            skipped += 1
            continue

        preds = result.get("predictions", [])
        img_w = result.get("image", {}).get("width", 1936)
        img_h = result.get("image", {}).get("height", 1040)

        if not preds:
            print("无检测")
            skipped += 1
            continue

        # 生成标注
        label_text = make_yolo_label(preds, img_w, img_h)
        with open(label_path, "w") as f:
            f.write(label_text)

        # 上传到 Roboflow
        try:
            project.upload(
                image_path=img_path,
                annotation_path=label_path,
                split="train",
            )
            uploaded += 1
            print(f"-> {len(preds)} 检测, 已上传")
        except Exception as e:
            print(f"上传失败: {e}")
            skipped += 1

        time.sleep(0.3)  # 避免限流

    print(f"\n完成! 上传: {uploaded}, 跳过: {skipped}")


if __name__ == "__main__":
    main()
