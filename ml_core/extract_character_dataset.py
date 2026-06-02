import os
import cv2
import numpy as np
import shutil

# Thư mục chứa dữ liệu đầu ra và đầu vào
DOWNLOADS_DIR = "/Users/thongvuong/Downloads"
OCR_SOURCE_DIR = os.path.join(DOWNLOADS_DIR, "OCR")
OUTPUT_CHAR_DIR = "/Users/thongvuong/plate_ml/ml_core/data/characters"

CHAR_LIST = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]

def extract_characters():
    print("Initializing real character extraction from teacher's OCR dataset...")
    
    if not os.path.exists(OCR_SOURCE_DIR):
        print(f"ERROR: OCR dataset source not found at: {OCR_SOURCE_DIR}")
        print("Please download and unzip OCR.zip in your Downloads folder first!")
        return False
        
    # Tạo thư mục đầu ra
    shutil.rmtree(OUTPUT_CHAR_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_CHAR_DIR, exist_ok=True)
    for char in CHAR_LIST:
        os.makedirs(os.path.join(OUTPUT_CHAR_DIR, char), exist_ok=True)
        
    # Danh sách các thư mục train/val của ảnh và nhãn
    subsets = ["train", "val"]
    
    count_dict = {char: 0 for char in CHAR_LIST}
    total_cropped = 0
    
    for subset in subsets:
        img_dir = os.path.join(OCR_SOURCE_DIR, "images", subset)
        label_dir = os.path.join(OCR_SOURCE_DIR, "labels", subset)
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"Skipping missing subset folders: {subset}")
            continue
            
        # Duyệt qua các tệp nhãn .txt
        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        print(f"Processing subset '{subset}' containing {len(label_files)} plate images...")
        
        for lf in label_files:
            base_name = os.path.splitext(lf)[0]
            
            # Tìm ảnh tương ứng với tệp nhãn (có thể là .jpg, .png...)
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                test_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
                    
            if img_path is None:
                continue
                
            # Đọc ảnh biển số gốc
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h_img, w_img = img.shape[:2]
            
            # Đọc tệp nhãn YOLO
            label_path = os.path.join(label_dir, lf)
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Bỏ qua các class ID không hợp lệ ngoài khoảng 0-35
                if class_id < 0 or class_id >= len(CHAR_LIST):
                    continue
                    
                char_label = CHAR_LIST[class_id]
                
                # Chuyển đổi tọa độ YOLO [0, 1] sang pixel
                x_c = x_center * w_img
                y_c = y_center * h_img
                w_c = width * w_img
                h_c = height * h_img
                
                left = int(x_c - w_c/2.0)
                right = int(x_c + w_c/2.0)
                top = int(y_c - h_c/2.0)
                bottom = int(y_c + h_c/2.0)
                
                # Giới hạn trong biên ảnh
                left = max(0, left)
                right = min(w_img, right)
                top = max(0, top)
                bottom = min(h_img, bottom)
                
                if (right - left) < 4 or (bottom - top) < 4:
                    continue
                    
                # Cắt ký tự chữ/số
                char_crop = img[top:bottom, left:right]
                
                # Chuyển ảnh xám
                if len(char_crop.shape) == 3:
                    char_gray = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
                else:
                    char_gray = char_crop.copy()
                    
                # Binarize nhị phân thích nghi (adaptive thresholding)
                char_thresh = cv2.adaptiveThreshold(
                    char_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Padding nhỏ và Resize về chuẩn 28x28
                pad = int(max(char_thresh.shape) * 0.1)
                char_padded = cv2.copyMakeBorder(
                    char_thresh, pad, pad, pad, pad,
                    cv2.BORDER_CONSTANT, value=0
                )
                char_resized = cv2.resize(char_padded, (28, 28), interpolation=cv2.INTER_AREA)
                
                # Ghi tệp ký tự đã chuẩn hóa ra đĩa
                char_filename = f"{base_name}_{total_cropped}.png"
                char_out_path = os.path.join(OUTPUT_CHAR_DIR, char_label, char_filename)
                
                cv2.imwrite(char_out_path, char_resized)
                
                count_dict[char_label] += 1
                total_cropped += 1
                
    print(f"\nSUCCESS: Extracted and structured {total_cropped} character images!")
    print("Sample distribution per character class:")
    for char in CHAR_LIST:
        print(f"Class '{char}': {count_dict[char]} samples")
        
    return True

if __name__ == "__main__":
    extract_characters()
