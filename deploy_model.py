#!/usr/bin/env python3
"""
Script chạy sau khi YOLOv5 training hoàn tất:
- Copy best.pt vào reference_src/model/LP_detector.pt
- Verify model hoạt động với vài ảnh test
"""
import os
import shutil
import torch

TRAINED_BEST = "/Users/thongvuong/plate_ml/runs/train/LP_detector_retrain/weights/best.pt"
TARGET_MODEL  = "/Users/thongvuong/plate_ml/reference_src/model/LP_detector.pt"
TEST_IMAGES   = "/Users/thongvuong/plate_ml/ml_core/data/LP_detection/images/val"

def deploy_model():
    if not os.path.exists(TRAINED_BEST):
        print(f"❌ Không tìm thấy model tại: {TRAINED_BEST}")
        print("   Training có thể chưa xong hoặc lỗi.")
        return False

    size_mb = os.path.getsize(TRAINED_BEST) / 1e6
    print(f"✅ Tìm thấy model: {TRAINED_BEST} ({size_mb:.1f} MB)")

    # Backup model cũ
    backup = TARGET_MODEL.replace(".pt", "_backup.pt")
    if os.path.exists(TARGET_MODEL):
        shutil.copy2(TARGET_MODEL, backup)
        print(f"📦 Backup model cũ → {backup}")

    # Copy model mới
    shutil.copy2(TRAINED_BEST, TARGET_MODEL)
    print(f"🚀 Đã deploy model mới → {TARGET_MODEL}")

    return True

def verify_model():
    print("\n🔍 Kiểm tra model mới...")
    try:
        import sys
        sys.path.insert(0, "/Users/thongvuong/plate_ml")
        model = torch.hub.load(
            "ultralytics/yolov5", "custom",
            path=TARGET_MODEL, trust_repo=True, verbose=False
        )
        model.conf = 0.4

        import cv2
        val_dir = TEST_IMAGES
        images = [f for f in os.listdir(val_dir) if f.endswith((".jpg", ".jpeg", ".png"))][:10]

        detected = 0
        for img_name in images:
            img_path = os.path.join(val_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(img_rgb, size=320)
            plates = results.pandas().xyxy[0]
            if len(plates) > 0:
                detected += 1
                conf = plates["confidence"].max()
                print(f"  ✅ {img_name}: {len(plates)} biển, conf={conf:.3f}")
            else:
                print(f"  ⚠️  {img_name}: không detect được")

        print(f"\nTỉ lệ detect: {detected}/{len(images)} ảnh ({detected/len(images)*100:.0f}%)")

    except Exception as e:
        print(f"❌ Lỗi verify: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("POST-TRAINING DEPLOYMENT SCRIPT")
    print("=" * 60)
    if deploy_model():
        verify_model()
    print("\nDone!")
