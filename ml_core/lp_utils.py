"""
lp_utils.py - Tiện ích xử lý ảnh biển số xe
Tham khảo từ: https://github.com/trungdinh22/License-Plate-Recognition
  - CLAHE contrast enhancement
  - Deskew (straighten tilted plate) via HoughLines
  - Plate candidate scoring based on LP_detection dataset statistics
"""
import cv2
import numpy as np
import math
import os


# ========================
# CLAHE CONTRAST ENHANCEMENT
# ========================

def enhance_contrast(img):
    """
    Tăng độ tương phản ảnh dùng CLAHE trên kênh L của LAB color space.
    Tham khảo từ reference: changeContrast() trong utils_rotate.py
    
    Rất hiệu quả với ảnh chụp ban đêm, ngược sáng, hoặc biển số bị mờ.
    """
    if len(img.shape) == 2:
        # Ảnh xám - áp CLAHE trực tiếp
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    # Ảnh màu: chuyển sang LAB, tăng cường kênh L
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced


# ========================
# DESKEW - CĂN THẲNG BIỂN SỐ
# ========================

def compute_skew_angle(src_img, center_thres=0):
    """
    Tính góc nghiêng của biển số dùng HoughLinesP.
    Tham khảo từ reference: compute_skew() trong utils_rotate.py
    
    center_thres: 1 = bỏ qua các đường nằm gần đỉnh ảnh (lọc nhiễu)
    """
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    else:
        h, w = src_img.shape
    
    # Lọc median trước khi detect cạnh để giảm nhiễu
    blurred = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100, 
                      apertureSize=3, L2gradient=True)
    
    # Tìm các đoạn thẳng dài (đường viền biển số)
    min_line_len = max(10, w / 4.0)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, 
                             minLineLength=min_line_len, maxLineGap=h / 3.0)
    
    if lines is None:
        return 0.0
    
    # Tìm đường nằm ở vị trí cao nhất (gần mép trên) làm đường chuẩn
    min_y = 9999
    min_line_idx = 0
    for i, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            center_y = (y1 + y2) / 2.0
            if center_thres == 1 and center_y < 7:
                continue
            if center_y < min_y:
                min_y = center_y
                min_line_idx = i
    
    # Tính góc trung bình từ đường chuẩn
    angle_sum = 0.0
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_idx]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if abs(ang) <= math.radians(30):  # Loại bỏ góc cực đoan
            angle_sum += ang
            cnt += 1
    
    if cnt == 0:
        return 0.0
    
    return (angle_sum / cnt) * 180.0 / math.pi


def rotate_image(image, angle):
    """Xoay ảnh quanh tâm với góc cho trước."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def deskew(plate_img, use_clahe=False, center_thres=0):
    """
    Căn thẳng biển số nghiêng bằng HoughLines.
    Tham khảo: deskew() trong reference utils_rotate.py
    
    Args:
        plate_img: ảnh biển số BGR hoặc xám
        use_clahe: True = tăng tương phản trước khi detect góc
        center_thres: 1 = bỏ qua đường ở phần đỉnh ảnh
    
    Returns:
        Ảnh biển số đã xoay thẳng
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img
    
    src = enhance_contrast(plate_img) if use_clahe else plate_img
    angle = compute_skew_angle(src, center_thres)
    
    # Chỉ xoay nếu góc đáng kể (> 0.5 độ)
    if abs(angle) < 0.5:
        return plate_img
    
    return rotate_image(plate_img, angle)


def try_deskew_variants(plate_img):
    """
    Thử 4 biến thể deskew (2 CLAHE × 2 center_thres) như reference.
    Trả về danh sách 4 ảnh đã căn thẳng để OCR thử từng cái.
    """
    variants = []
    for use_clahe in [False, True]:
        for center_thres in [0, 1]:
            try:
                result = deskew(plate_img, use_clahe, center_thres)
                variants.append(result)
            except Exception:
                variants.append(plate_img)
    return variants


# ========================
# DETECT BIỂN SỐ BẰNG THỐNG KÊ TỪ LP_DETECTION DATASET
# ========================

def load_yolo_boxes(label_path, img_w, img_h):
    """
    Đọc file label YOLO và chuyển sang tọa độ pixel.
    Format: class cx cy w h (normalized 0-1)
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = map(float, parts)
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    
    return boxes


def compute_dataset_stats(dataset_path, max_samples=500):
    """
    Tính thống kê aspect ratio, y_ratio, w_ratio từ LP_detection dataset.
    Dùng để cải thiện scoring system khi detect biển số.
    
    Trả về dict: {'ar_mean', 'ar_std', 'y_ratio_mean', 'w_ratio_mean'}
    """
    images_dir = os.path.join(dataset_path, 'images', 'train')
    labels_dir = os.path.join(dataset_path, 'labels', 'train')
    
    if not os.path.exists(images_dir):
        return None
    
    ars = []
    y_ratios = []
    w_ratios = []
    
    img_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    np.random.shuffle(img_files)
    img_files = img_files[:max_samples]
    
    for img_file in img_files:
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h_img, w_img = img.shape[:2]
        boxes = load_yolo_boxes(label_path, w_img, h_img)
        
        for (x, y, w, h) in boxes:
            if h > 0 and w > 0:
                ars.append(w / float(h))
                y_ratios.append((y + h / 2.0) / h_img)
                w_ratios.append(w / float(w_img))
    
    if not ars:
        return None
    
    return {
        'ar_mean': float(np.mean(ars)),
        'ar_std': float(np.std(ars)),
        'y_ratio_mean': float(np.mean(y_ratios)),
        'y_ratio_std': float(np.std(y_ratios)),
        'w_ratio_mean': float(np.mean(w_ratios)),
        'w_ratio_std': float(np.std(w_ratios)),
        'ar_min': float(np.percentile(ars, 5)),
        'ar_max': float(np.percentile(ars, 95)),
        'samples': len(ars)
    }


# ========================
# PHÂN LOẠI 1-LINE vs 2-LINE (tham khảo helper.py của reference)
# ========================

def _linear_equation(x1, y1, x2, y2):
    """Tính hệ số đường thẳng qua 2 điểm: y = ax + b"""
    if x2 == x1:
        return 0, y1
    a = (y2 - y1) / float(x2 - x1)
    b = y1 - a * x1
    return a, b


def _point_on_line(x, y, x1, y1, x2, y2, tol=5):
    """Kiểm tra điểm (x,y) có nằm gần đường thẳng qua (x1,y1)-(x2,y2) không."""
    a, b = _linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return abs(y_pred - y) <= tol


def classify_plate_lines_by_centers(char_centers):
    """
    Phân loại biển số 1 hàng vs 2 hàng dựa trên tọa độ tâm các ký tự.
    Tham khảo: read_plate() trong reference helper.py
    
    Args:
        char_centers: list of (cx, cy) - tâm của từng ký tự
    
    Returns:
        "1" = biển 1 hàng (Rectangle)
        "2" = biển 2 hàng (Square)
    """
    if len(char_centers) < 3:
        return "1"
    
    # Tìm điểm trái nhất và phải nhất
    l_point = min(char_centers, key=lambda p: p[0])
    r_point = max(char_centers, key=lambda p: p[0])
    
    if l_point[0] == r_point[0]:
        return "2"
    
    # Kiểm tra xem tất cả các điểm có nằm trên đường thẳng nối l_point-r_point không
    for cx, cy in char_centers:
        on_line = _point_on_line(cx, cy, l_point[0], l_point[1], r_point[0], r_point[1], tol=8)
        if not on_line:
            return "2"
    
    return "1"


# ========================
# TIỆN ÍCH HIỂN THỊ DEBUG
# ========================

def draw_debug_overlay(img, plate_box, candidates=None, title="Detection"):
    """
    Vẽ kết quả detection lên ảnh với màu sắc rõ ràng.
    """
    result = img.copy()
    h_img, w_img = result.shape[:2]
    
    # Vẽ tất cả candidates (màu xám nhạt)
    if candidates:
        for cand in candidates[:10]:
            x, y, w, h = cand["box"]
            cv2.rectangle(result, (x, y), (x+w, y+h), (100, 100, 100), 1)
    
    # Vẽ biển số chọn được (màu xanh neon)
    x, y, w, h = plate_box
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 170), 3)
    
    # Vẽ tên debug
    cv2.putText(result, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 170), 2, cv2.LINE_AA)
    
    return result
