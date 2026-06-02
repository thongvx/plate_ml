# ==============================================================================
# TẬP LỆNH HUẤN LUYỆN YOLO CHO BÀI TẬP LỚN ML (LPR)
# ==============================================================================
# Bộ dữ liệu bạn đã gửi từ Google Drive:
# - Link 1: Dữ liệu ảnh xe chứa biển số & nhãn YOLO (khung biển số [class_id, x, y, w, h]).
# - Link 2: Dữ liệu ảnh biển số đã cắt & nhãn YOLO của từng chữ số (OCR).

import os
import zipfile
# Thư viện học sâu YOLO thế hệ mới cực kỳ phổ biến
try:
    from ultralytics import YOLO
except ImportError:
    print("Vui lòng cài đặt ultralytics bằng lệnh: pip install ultralytics")
    exit()

def train_plate_detector(dataset_dir="data/plate_dataset"):
    """
    Bước 1: Huấn luyện mô hình Phát hiện vị trí biển số (Plate Detector)
    Dữ liệu đầu vào: Ảnh chụp toàn cảnh xe. Nhãn: Vị trí khung biển số xe.
    """
    print("--- KHỞI ĐỘNG HUẤN LUYỆN PHÁT HIỆN BIỂN SỐ XE (YOLOv8) ---")
    
    # 1. Tạo file cấu hình dataset.yaml cho YOLO
    yaml_content = f"""
    path: {os.path.abspath(dataset_dir)} # Thư mục gốc của tập dữ liệu
    train: images/train # Ảnh huấn luyện
    val: images/val     # Ảnh xác thực (validation)

    # Lớp cần phát hiện (chỉ có 1 lớp là biển số xe)
    names:
      0: license_plate
    """
    
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
        
    # 2. Khởi tạo mô hình YOLOv8n (YOLOv8 Nano - siêu nhẹ, phù hợp chạy CPU/GPU Mac)
    model = YOLO("yolov8n.yaml") # Tạo mô hình mới từ kiến trúc yaml
    
    # Tải trọng số pre-trained trên tập COCO để chuyển giao học tập (Transfer Learning) giúp hội tụ nhanh
    model = YOLO("yolov8n.pt") 
    
    # 3. Tiến hành huấn luyện
    # - epochs: số chu kỳ huấn luyện (chạy thử nghiệm nên để 50 - 100)
    # - imgsz: kích thước ảnh đầu vào (640 là tiêu chuẩn)
    # - batch: kích thước batch size (8 hoặc 16)
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu", # Nếu chạy trên Mac M1/M2/M3 bạn có thể đổi thành 'mps' để dùng GPU Apple
        name="vietnamese_lpr_detector"
    )
    print("Huấn luyện thành công! Trọng số tốt nhất được lưu tại: runs/detect/vietnamese_lpr_detector/weights/best.pt")
    
    # 4. Xuất mô hình sang định dạng ONNX siêu nhẹ để nhúng trực tiếp vào OpenCV (cv2.dnn)
    # Định dạng ONNX giúp app Streamlit load trực tiếp siêu nhanh mà không cần thư viện PyTorch nặng nề!
    onnx_path = model.export(format="onnx")
    print(f"Đã xuất file ONNX thành công: {onnx_path}")

def train_digit_recognizer(dataset_dir="data/ocr_dataset"):
    """
    Bước 2: Huấn luyện mô hình Phân đoạn & Nhận diện Ký tự bằng YOLO (OCR)
    Dữ liệu đầu vào: Ảnh biển số đã cắt. Nhãn: Vị trí và nhãn của từng chữ cái/chữ số [0-9, A-Z].
    """
    print("\n--- KHỞI ĐỘNG HUẤN LUYỆN NHẬN DIỆN KÝ TỰ BIỂN SỐ - OCR (YOLOv8) ---")
    
    yaml_content = f"""
    path: {os.path.abspath(dataset_dir)}
    train: images/train
    val: images/val

    # Danh sách 36 ký tự biển số xe Việt Nam
    names:
      0: '0'
      1: '1'
      2: '2'
      3: '3'
      4: '4'
      5: '5'
      6: '6'
      7: '7'
      8: '8'
      9: '9'
      10: 'A'
      11: 'B'
      12: 'C'
      13: 'D'
      14: 'E'
      15: 'F'
      16: 'G'
      17: 'H'
      18: 'I'
      19: 'J'
      20: 'K'
      21: 'L'
      22: 'M'
      23: 'N'
      24: 'O'
      25: 'P'
      26: 'Q'
      27: 'R'
      28: 'S'
      29: 'T'
      30: 'U'
      31: 'V'
      32: 'W'
      33: 'X'
      34: 'Y'
      35: 'Z'
    """
    
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
        
    # Huấn luyện mô hình OCR chuyên phát hiện chữ số
    model = YOLO("yolov8n.pt")
    model.train(
        data=yaml_path,
        epochs=80,
        imgsz=320, # Biển số xe nhỏ nên dùng size 320 là đủ sắc nét
        batch=32,
        device="cpu",
        name="vietnamese_lpr_ocr"
    )
    model.export(format="onnx")
    print("Huấn luyện OCR thành công!")

if __name__ == "__main__":
    # Hướng dẫn bạn cấu trúc thư mục dữ liệu trước khi chạy:
    # 
    # data/
    # ├── plate_dataset/
    # │   ├── images/
    # │   │   ├── train/ (chứa các file .jpg toàn cảnh xe)
    # │   │   └── val/
    # │   └── labels/
    # │       ├── train/ (chứa các file .txt nhãn tọa độ biển số)
    # │       └── val/
    # └── ocr_dataset/
    #     ├── images/
    #     │   ├── train/ (chứa các file .jpg ảnh biển số cắt nhỏ)
    #     │   └── val/
    #     └── labels/
    #         ├── train/ (chứa các file .txt nhãn tọa độ và class các chữ số)
    #         └── val/
    
    print("Vui lòng giải nén bộ dữ liệu của bạn từ Google Drive vào cấu trúc thư mục 'data/' ở trên.")
    print("Sau đó gỡ comment dòng dưới đây để tiến hành chạy huấn luyện thực tế:")
    # train_plate_detector("data/plate_dataset")
    # train_digit_recognizer("data/ocr_dataset")