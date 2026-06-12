"""
segmentation.py - Phát hiện và phân đoạn biển số xe
Kết hợp:
  - Kỹ thuật xử lý ảnh OpenCV (Sobel, Canny, Morphology, Contours)
  - CLAHE contrast enhancement + Deskew từ reference (trungdinh22/License-Plate-Recognition)
  - Thông tin thống kê từ dataset LP_detection (YOLO labels)
"""
import cv2
import numpy as np
import os
import torch

# ========================
# HỖ TRỢ YOLO CỦA REFERENCE PROJECT
# ========================

_YOLO_LP_DETECT = None
_YOLO_LP_OCR = None
_YOLO_MODELS_LOADED = False

# Đường dẫn local YOLOv5 source (đã cache)
_YOLOV5_LOCAL = os.path.expanduser('~/.cache/torch/hub/ultralytics_yolov5_master')
# Đường dẫn chứa file model .pt — trong reference_src/model
_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reference_src', 'model'))

def _load_yolo_models():
    global _YOLO_LP_DETECT, _YOLO_LP_OCR, _YOLO_MODELS_LOADED
    if _YOLO_MODELS_LOADED:
        return _YOLO_LP_DETECT, _YOLO_LP_OCR

    try:
        detect_path = os.path.join(_MODEL_DIR, 'LP_detector.pt')
        ocr_path    = os.path.join(_MODEL_DIR, 'LP_ocr.pt')

        # Dùng local YOLOv5 cache (source='local') — giống hệt reference project
        _YOLO_LP_DETECT = torch.hub.load(
            _YOLOV5_LOCAL, 'custom',
            path=detect_path,
            source='local',
            force_reload=False,
            verbose=False
        )
        _YOLO_LP_DETECT.conf = 0.25  # ngưỡng thấp để không bỏ sót
        _YOLO_LP_DETECT.iou  = 0.45

        _YOLO_LP_OCR = torch.hub.load(
            _YOLOV5_LOCAL, 'custom',
            path=ocr_path,
            source='local',
            force_reload=False,
            verbose=False
        )
        _YOLO_LP_OCR.conf = 0.60  # y chang reference
        _YOLO_MODELS_LOADED = True
        print('✅ Đã load YOLOv5 LP_detector.pt & LP_ocr.pt từ local cache')
    except Exception as e:
        print(f'⚠️ Không thể tải YOLOv5 local: {e}. Dùng fallback OpenCV.')
        _YOLO_MODELS_LOADED = True

    return _YOLO_LP_DETECT, _YOLO_LP_OCR


# ========================
# HELPER: DỌC KÝ TỰ BIỂN SỐ BẰNG LP_OCR (theo reference)
# ========================

def _linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def _check_point_linear(x, y, x1, y1, x2, y2):
    import math
    a, b = _linear_equation(x1, y1, x2, y2)
    return math.isclose(a * x + b, y, abs_tol=3)

def _compute_skew(src_img, center_thres):
    import math
    h, w = src_img.shape[:2]
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, 30, 100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30,
                            minLineLength=w/1.5, maxLineGap=h/3.0)
    if lines is None:
        return 1
    min_line, min_line_pos = 100, 0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cp = [(x1+x2)/2, (y1+y2)/2]
            if center_thres == 1 and cp[1] < 7:
                continue
            if cp[1] < min_line:
                min_line, min_line_pos = cp[1], i
    angle, cnt = 0.0, 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if abs(ang) <= 30:
            angle += ang; cnt += 1
    return 0.0 if cnt == 0 else (angle/cnt)*180/math.pi

def _deskew(src_img, change_cons, center_thres):
    """Deskew ảnh biển số, y chang reference utils_rotate.deskew"""
    import math
    h, w = src_img.shape[:2]
    if change_cons == 1:
        # CLAHE trên kênh L
        lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        src_proc = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    else:
        src_proc = src_img
    angle = _compute_skew(src_proc, center_thres)
    center = tuple(np.array(src_img.shape[1::-1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(src_img, M, src_img.shape[1::-1], flags=cv2.INTER_LINEAR)

def _read_plate_yolo(yolo_ocr, plate_img):
    """
    Đọc ký tự biển số từ ảnh biển đã crop, dùng LP_ocr.pt.
    Trả về chuỗi biển số hoặc 'unknown'.
    Y chang helper.read_plate() của reference project.
    """
    results = yolo_ocr(plate_img)
    bb_list = results.pandas().xyxy[0].values.tolist()
    if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
        return 'unknown'

    center_list = []
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0] + bb[2]) / 2
        y_c = (bb[1] + bb[3]) / 2
        y_sum += y_c
        center_list.append([x_c, y_c, bb[-1]])  # bb[-1] = tên class (ký tự)

    # Phán loại 1 dòng / 2 dòng
    LP_type = '1'
    l_point = min(center_list, key=lambda p: p[0])
    r_point = max(center_list, key=lambda p: p[0])
    if l_point[0] != r_point[0]:
        for ct in center_list:
            if not _check_point_linear(ct[0], ct[1],
                                       l_point[0], l_point[1],
                                       r_point[0], r_point[1]):
                LP_type = '2'
                break

    y_mean = int(y_sum / len(bb_list))
    license_plate = ''
    if LP_type == '2':
        line_1 = [c for c in center_list if int(c[1]) <= y_mean]
        line_2 = [c for c in center_list if int(c[1]) >  y_mean]
        for l1 in sorted(line_1, key=lambda x: x[0]):
            license_plate += str(l1[2])
        license_plate += '-'
        for l2 in sorted(line_2, key=lambda x: x[0]):
            license_plate += str(l2[2])
    else:
        for l in sorted(center_list, key=lambda x: x[0]):
            license_plate += str(l[2])
    return license_plate


# ========================
# PHÂN LOẠI BIỂN SỐ
# ========================

def classify_plate_type(plate_img):
    """
    Phân loại kiểu dáng biển số:
    - Rectangle (1-line): aspect ratio > 2.2
    - Square (2-line): aspect ratio <= 2.2
    
    Kèm phát hiện màu biển số (White/Yellow/Blue/Green/Red).
    """
    h, w = plate_img.shape[:2]
    if h == 0 or w == 0:
        return "Rectangle (1-line)", 3.0, "White"
    
    aspect_ratio = w / float(h)
    plate_shape = "Rectangle (1-line)" if aspect_ratio > 2.2 else "Square (2-line)"
    plate_color = _detect_plate_color(plate_img)
    
    return plate_shape, aspect_ratio, plate_color


def _detect_plate_color(plate_img):
    """Phát hiện màu chính của nền biển số qua phân tích HSV."""
    if plate_img is None or plate_img.size == 0 or len(plate_img.shape) < 3:
        return "White"
    
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    total = plate_img.shape[0] * plate_img.shape[1]
    
    # Trắng: saturation thấp, value cao
    white_mask = (s_ch < 60) & (v_ch > 160)
    if np.sum(white_mask) / total > 0.25:
        return "White"
    
    masks = {
        "Yellow": cv2.inRange(hsv, np.array([15, 80, 80]), np.array([40, 255, 255])),
        "Blue":   cv2.inRange(hsv, np.array([100, 60, 60]), np.array([130, 255, 255])),
        "Green":  cv2.inRange(hsv, np.array([40, 60, 60]), np.array([90, 255, 255])),
        "Red1":   cv2.inRange(hsv, np.array([0, 60, 60]), np.array([10, 255, 255])),
        "Red2":   cv2.inRange(hsv, np.array([160, 60, 60]), np.array([180, 255, 255])),
    }
    
    ratios = {
        "Yellow": np.sum(masks["Yellow"] > 0) / total,
        "Blue": np.sum(masks["Blue"] > 0) / total,
        "Green": np.sum(masks["Green"] > 0) / total,
        "Red": (np.sum(masks["Red1"] > 0) + np.sum(masks["Red2"] > 0)) / total,
    }
    
    best = max(ratios, key=ratios.get)
    return best if ratios[best] > 0.08 else "White"


# ========================
# TIỀN XỬ LÝ ẢNH
# ========================

def _enhance_contrast(img):
    """CLAHE contrast enhancement (từ reference utils_rotate.changeContrast)."""
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced


def preprocess_image(img, blur_kernel=(5, 5)):
    """Chuyển xám + Bilateral Filter."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray, blurred


def threshold_image(gray_img):
    """
    Phân ngưỡng thông minh: so sánh Otsu và Adaptive, chọn cái cho nhiều
    contour ký tự hơn.
    """
    _, thresh_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 19, 9
    )
    
    h_plate, w_plate = gray_img.shape[:2]
    
    def count_valid(thresh):
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for c in cnts
                   if 0.20 < cv2.boundingRect(c)[3] / float(h_plate) < 0.95
                   and cv2.boundingRect(c)[2] > 3)
    
    return thresh_adapt if count_valid(thresh_adapt) >= count_valid(thresh_otsu) else thresh_otsu


# ========================
# DESKEW - CĂN THẲNG BIỂN SỐ (từ reference)
# ========================

def _compute_skew_angle(src_img, center_thres=0):
    """Tính góc nghiêng qua HoughLinesP."""
    import math
    h = src_img.shape[0]
    w = src_img.shape[1] if len(src_img.shape) > 1 else src_img.shape[0]
    
    blurred = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3, L2gradient=True)
    
    min_len = max(10, w / 4.0)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30,
                             minLineLength=min_len, maxLineGap=h / 3.0)
    if lines is None:
        return 0.0
    
    min_y = 9999
    min_idx = 0
    for i, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            cy = (y1 + y2) / 2.0
            if center_thres == 1 and cy < 7:
                continue
            if cy < min_y:
                min_y = cy
                min_idx = i
    
    angle_sum = 0.0
    cnt = 0
    for x1, y1, x2, y2 in lines[min_idx]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if abs(ang) <= math.radians(30):
            angle_sum += ang
            cnt += 1
    
    return (angle_sum / cnt) * 180.0 / math.pi if cnt > 0 else 0.0


def deskew_plate(plate_img, use_clahe=False, center_thres=0):
    """Căn thẳng biển số nghiêng (tham khảo reference deskew)."""
    import math
    if plate_img is None or plate_img.size == 0:
        return plate_img
    
    src = _enhance_contrast(plate_img) if use_clahe else plate_img
    angle = _compute_skew_angle(src, center_thres)
    
    if abs(angle) < 0.5:
        return plate_img
    
    h, w = plate_img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(plate_img, M, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ========================
# PHÁT HIỆN BIỂN SỐ - CẢI TIẾN VỚI THỐNG KÊ TỪ LP_DETECTION
# ========================

# Thống kê aspect ratio từ dataset LP_detection (tính trước)
# CarLongPlate: ~3.5-5.0, xemay (2-line): ~1.1-1.9, xe tải: ~3.0-4.5
_LP_STATS = {
    'ar_range_1line': (2.0, 6.5),   # biển 1 hàng
    'ar_range_2line': (1.0, 2.0),   # biển 2 hàng
    'y_ratio_mean': 0.60,           # biển thường ở 60% chiều cao ảnh
    'w_ratio_range': (0.05, 0.95),  # biển chiếm 5%-95% chiều rộng ảnh
}


def _score_candidate(cand, h_img, w_img, gray_img, thresh_img):
    """
    Chấm điểm ứng viên biển số dựa trên hình học và cấu trúc bên trong.
    - Aspect Ratio (AR) của biển ngang: 3.0 - 5.5, biển vuông: 1.1 - 1.8
    - Kích thước hợp lý (w_ratio: 0.05 - 0.6, h_ratio: 0.02 - 0.25)
    - Vị trí (x_center: 0.25 - 0.75, y_center: 0.40 - 0.88)
    - Mật độ ký tự bên trong: cực kỳ quan trọng! Cắt vùng ứng viên từ ảnh thresh,
      đếm số lượng contour ký tự hợp lệ bên trong.
    """
    x, y, w, h = cand["box"]
    ar = cand["aspect_ratio"]
    
    y_center = y + h / 2.0
    x_center = x + w / 2.0
    
    score = 0.0
    
    # 1. Aspect Ratio Score (Trọng số cực cao: 120 điểm)
    is_rect = 2.8 <= ar <= 5.8
    is_sq = 1.0 <= ar <= 2.0
    
    if is_rect:
        # Tối ưu quanh mức 4.0 - 4.8 cho biển ngang ô tô/xe máy cũ
        score += 120.0 * (1.0 - abs(ar - 4.4) / 1.6)
    elif is_sq:
        # Tối ưu quanh mức 1.2 - 1.5 cho biển vuông
        score += 100.0 * (1.0 - abs(ar - 1.35) / 0.45)
    else:
        score -= 200.0  # Phạt nặng aspect ratio dị dị
        
    # 2. Vị trí trong ảnh xe (Trọng số: 50 điểm)
    y_ratio = y_center / h_img
    x_ratio = x_center / w_img
    
    # Thưởng nếu nằm ở 38% - 88% chiều cao (vùng đặt biển số điển hình) và 20% - 80% chiều rộng
    if 0.38 <= y_ratio <= 0.88:
        score += 35.0
        # Càng gần trục dọc trung tâm càng tốt
        score += 15.0 * (1.0 - abs(x_ratio - 0.5) / 0.5)
    elif y_ratio < 0.28:
        score -= 150.0  # Phạt nặng logo xe trên capo
    else:
        score -= 50.0   # Quá sát viền
        
    # 3. Kích thước (Trọng số: 40 điểm)
    w_ratio = w / float(w_img)
    h_ratio = h / float(h_img)
    
    if 0.05 <= w_ratio <= 0.65 and 0.02 <= h_ratio <= 0.25:
        score += 40.0
    elif w_ratio > 0.75 or h_ratio > 0.35:
        score -= 180.0  # Phạt cực nặng ứng viên khổng lồ (bị nhóm cả cản trước)
    else:
        score -= 50.0   # Quá nhỏ (nhiễu)
        
    # 4. Mật độ ký tự bên trong (Trọng số cao nhất: 200 điểm!)
    cand_thresh = thresh_img[y:y+h, x:x+w]
    if cand_thresh.size > 0:
        # Đếm số lượng contour ký tự đứng bên trong ứng viên
        cnts, _ = cv2.findContours(cand_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_count = 0
        for c in cnts:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # Ký tự bên trong biển số phải thỏa mãn tỉ lệ hình học tương đối so với biển số
            ch_ratio = ch / float(h)
            cw_ratio = cw / float(w)
            char_ar = cw / float(ch)
            if (0.20 <= ch_ratio <= 0.90 and 
                0.01 <= cw_ratio <= 0.28 and 
                char_ar < 1.15 and 
                cw * ch > 12):
                char_count += 1
        
        # Thưởng lớn nếu số ký tự hợp lệ nằm trong khoảng [6, 10]
        if 6 <= char_count <= 10:
            score += 200.0
        elif 4 <= char_count <= 5:
            score += 80.0
        elif char_count >= 11:
            score -= 100.0  # Quá nhiều nhiễu
        else:
            score -= 120.0  # Quá ít ký tự, khả năng cao là logo hoặc đèn sương mù!
            
    return score


def _generate_clustering_candidates(gray_img, thresh_img, w_img, h_img):
    """
    Sinh các ứng viên biển số bằng cách tìm các contour giống ký tự
    và gom nhóm chúng lại nếu xếp gần nhau theo hàng ngang.
    """
    candidates = []
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 1. Lọc ra các contour đơn lẻ giống ký tự
    char_candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or w == 0:
            continue
        ar = w / float(h)
        h_ratio = h / float(h_img)
        w_ratio = w / float(w_img)
        
        if 0.015 < h_ratio < 0.12 and 0.005 < w_ratio < 0.08 and ar < 0.95 and w * h > 15:
            char_candidates.append((x, y, w, h, x + w/2.0, y + h/2.0))
            
    if len(char_candidates) < 3:
        return candidates
        
    # 2. Gom nhóm các ứng viên ký tự theo hàng ngang
    groups = []
    used = set()
    
    # Sắp xếp theo trục X để duyệt từ trái qua phải
    char_candidates = sorted(char_candidates, key=lambda item: item[0])
    
    for i in range(len(char_candidates)):
        if i in used:
            continue
        # Bắt đầu một nhóm mới
        group = [char_candidates[i]]
        used.add(i)
        
        avg_h = char_candidates[i][3]
        avg_cy = char_candidates[i][5]
        
        for j in range(i + 1, len(char_candidates)):
            if j in used:
                continue
            x2, y2, w2, h2, cx2, cy2 = char_candidates[j]
            if abs(cy2 - avg_cy) < avg_h * 0.7:
                rightmost_x = max(item[0] + item[2] for item in group)
                if x2 - rightmost_x < avg_h * 3.2:
                    group.append(char_candidates[j])
                    used.add(j)
                    avg_h = np.mean([item[3] for item in group])
                    avg_cy = np.mean([item[5] for item in group])
                    
        if len(group) >= 3:
            groups.append(group)
            
    # 3. Tạo bounding box bao quanh mỗi nhóm ký tự và tính toán ứng viên biển số
    for g in groups:
        xs = [item[0] for item in g]
        ys = [item[1] for item in g]
        ws = [item[2] for item in g]
        hs = [item[3] for item in g]
        
        min_x = min(xs)
        min_y = min(ys)
        max_x = max([x + w for x, w in zip(xs, ws)])
        max_y = max([y + h for y, h in zip(ys, hs)])
        
        box_w = max_x - min_x
        box_h = max_y - min_y
        
        pad_x = int(box_w * 0.08)
        pad_y = int(box_h * 0.18)
        
        rx = max(0, min_x - pad_x)
        ry = max(0, min_y - pad_y)
        rw = min(w_img - rx, box_w + 2 * pad_x)
        rh = min(h_img - ry, box_h + 2 * pad_y)
        
        if rw > 0 and rh > 0:
            cand_ar = rw / float(rh)
            candidates.append({
                "box": [rx, ry, rw, rh],
                "area": rw * rh,
                "aspect_ratio": cand_ar,
                "method": "clustering"
            })
            
    return candidates


def _apply_nms(candidates, overlap_thresh=0.45):
    """
    Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các ứng viên biển số
    bị trùng lặp đè nêm nhau, giữ lại ứng viên có score tốt nhất.
    """
    if not candidates:
        return []
        
    cands_sorted = sorted(candidates, key=lambda c: c["score"], reverse=True)
    keep = []
    
    while len(cands_sorted) > 0:
        best = cands_sorted.pop(0)
        keep.append(best)
        
        remaining = []
        for cand in cands_sorted:
            x1, y1, w1, h1 = best["box"]
            x2, y2, w2, h2 = cand["box"]
            
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            
            intersection = iw * ih
            union = (w1 * h1) + (w2 * h2) - intersection
            
            iou = intersection / float(union) if union > 0 else 0
            
            if iou < overlap_thresh:
                remaining.append(cand)
                
        cands_sorted = remaining
        
    return keep


def _extend_square_plate_if_needed(plate_box, car_img):
    """
    Tự động phát hiện và sửa lỗi khi YOLO hoặc OpenCV chỉ detect được hàng trên 
    của biển số vuông (2 dòng). Kiểm tra xem có hàng chữ nào ngay dưới không để tự động kéo dài box.
    """
    h_img, w_img = car_img.shape[:2]
    x, y, w, h = plate_box
    ar = w / float(h)
    
    # Biển vuông 2 hàng bị detect thiếu thường có AR nằm trong khoảng rộng từ 1.8 - 5.5.
    if 1.8 <= ar <= 5.5:
        # Bắt đầu tìm kiếm hơi thụt lên trên một chút (82% của h) để đảm bảo capture trọn vẹn
        # cả khi rìa của hàng dưới bị cắt lẹm một phần bởi box cũ
        search_y = y + int(h * 0.82)
        search_h = min(int(h * 1.5), h_img - search_y)
        
        if search_h > 5:
            search_region = car_img[search_y:search_y + search_h, x:x + w]
            gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
            
            # Kết hợp cả Adaptive và Otsu để nâng cao tối đa độ nhạy trong điều kiện thiếu sáng/ngược sáng
            thresh_adapt = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 7
            )
            _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            cnts_adapt, _ = cv2.findContours(thresh_adapt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_otsu, _ = cv2.findContours(thresh_otsu.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_ys_adapt = []
            for c in cnts_adapt:
                cx, cy, cw, ch = cv2.boundingRect(c)
                ch_ratio = ch / float(h)
                cw_ratio = cw / float(w)
                char_ar = cw / float(ch)
                if 0.12 <= ch_ratio <= 1.6 and cw_ratio < 0.40 and char_ar < 1.6:
                    valid_ys_adapt.append(cy + ch)
                    
            valid_ys_otsu = []
            for c in cnts_otsu:
                cx, cy, cw, ch = cv2.boundingRect(c)
                ch_ratio = ch / float(h)
                cw_ratio = cw / float(w)
                char_ar = cw / float(ch)
                if 0.12 <= ch_ratio <= 1.6 and cw_ratio < 0.40 and char_ar < 1.6:
                    valid_ys_otsu.append(cy + ch)
                    
            # Chọn danh sách nhận được nhiều nét ký tự đứng hơn
            valid_bottom_ys = valid_ys_adapt if len(valid_ys_adapt) >= len(valid_ys_otsu) else valid_ys_otsu
            
            # Nếu tìm thấy từ 2 ký tự trở lên xếp dưới, chắc chắn đây là biển vuông!
            if len(valid_bottom_ys) >= 2:
                max_bottom_in_search = max(valid_bottom_ys)
                abs_bottom_y = search_y + max_bottom_in_search
                
                # Tự động điều chỉnh kéo dài chính xác theo tọa độ Y của ký tự thấp nhất tìm thấy
                new_h = min(h_img - y, abs_bottom_y - y + int(h * 0.15))
                return [x, y, w, new_h]
                
    return plate_box


def _perspective_correct(plate_crop_raw):
    """
    Áp dụng Perspective Transform để nắn thẳng biển số bị lệch góc nhìn.
    Thuật toán:
      1. Tìm vùng sáng lớn nhất trong ảnh (vùng biển số trắng)
      2. Xấp xỉ contour → 4 góc → warpPerspective
      3. Fallback: dùng minAreaRect → boxPoints
    Cải tiến: Hạ thấp ngưỡng diện tích (10% thay vì 25%) để khắp phục biển nhỏ/lệch góc nặng.
    Trả về ảnh đã được nắn hoặc ảnh gốc nếu không tìm được 4 góc.
    """
    if plate_crop_raw is None or plate_crop_raw.size == 0:
        return plate_crop_raw

    h, w = plate_crop_raw.shape[:2]
    if h < 10 or w < 20:
        return plate_crop_raw

    # --- Bước 1: Tìm mask vùng biển ---
    # CLAHE + threshold để lấy vùng sáng của biển
    lab = cv2.cvtColor(plate_crop_raw, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_eq = clahe.apply(l_ch)

    # Otsu trên kênh L đã equalize
    _, mask = cv2.threshold(l_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology đóng để lấp lỗ hổng giữa chữ (tăng kích thước kernel một chút)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 6), max(3, h // 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # Morphology mở để loại nhiễu biên
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # --- Bước 2: Tìm contour lớn nhất ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop_raw

    # Hạ ngưỡng diện tích từ 25% xuống 10% để phục vụ biển lệch góc nặng hoặc nhỏ
    min_area = w * h * 0.10
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        valid = contours  # fallback: dùng tất cả
    largest = max(valid, key=cv2.contourArea)

    # --- Bước 3: Xấp xỉ polygon → cố gắng lấy 4 góc ---
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.04 * peri, True)

    def _order_pts(pts):
        """Sắp xếp 4 điểm: top-left, top-right, bottom-right, bottom-left."""
        pts = pts.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],   # top-left
            pts[np.argmin(diff)], # top-right
            pts[np.argmax(s)],   # bottom-right
            pts[np.argmax(diff)], # bottom-left
        ], dtype=np.float32)

    src_pts = None
    if len(approx) == 4:
        src_pts = _order_pts(approx)
    else:
        # Fallback: minAreaRect → boxPoints
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype(np.float32)
        src_pts = _order_pts(box)

    # --- Bước 4: Tính kích thước đích ---
    tl, tr, br, bl = src_pts
    width_top  = np.linalg.norm(tr - tl)
    width_bot  = np.linalg.norm(br - bl)
    height_l   = np.linalg.norm(bl - tl)
    height_r   = np.linalg.norm(br - tr)

    dst_w = int(max(width_top, width_bot))
    dst_h = int(max(height_l, height_r))

    if dst_w < 20 or dst_h < 8:
        return plate_crop_raw

    # Cho phép phóng to lên đến 2.5× để xử lý biển nghiêng lệch góc nặng
    dst_w = min(dst_w, int(w * 2.5))
    dst_h = min(dst_h, int(h * 2.5))

    dst_pts = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ], dtype=np.float32)

    # --- Bước 5: Warp Perspective ---
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(plate_crop_raw, M, (dst_w, dst_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

    # Chỉ giữ kết quả nếu aspect ratio kết quả hợp lý hơn ảnh gốc
    # (biển thực tế luôn có AR ≥ 1.2, relaxed từ 1.5 để phục vụ biển 2 dòng)
    warped_ar = dst_w / float(dst_h) if dst_h > 0 else 0
    orig_ar   = w / float(h) if h > 0 else 0
    if warped_ar < 1.0:  # kết quả tệ hơn → giữ nguyên
        return plate_crop_raw

    return warped


def detect_plate(car_img):
    """
    Phát hiện biển số xe bằng YOLOv5 LP_detector.pt — y chang reference project.
    Pipeline:
      1. Dùng LP_detector.pt detect vị trí biển (size=640)
      2. Crop biển + padding 15%
      3. Perspective correction (nắn biển lệch góc)
      4. Deskew 4 combo (change_cons, center_thres) y chang reference
      5. Nếu YOLO fail → fallback OpenCV
    """
    h_img, w_img = car_img.shape[:2]

    yolo_detect, yolo_ocr = _load_yolo_models()

    if yolo_detect is not None:
        try:
            plates = yolo_detect(car_img, size=640)
            list_plates = plates.pandas().xyxy[0].values.tolist()

            if len(list_plates) > 0:
                list_plates = sorted(list_plates, key=lambda p: p[4], reverse=True)
                plate = list_plates[0]
                conf = float(plate[4])

                x  = max(0, int(plate[0]))
                y  = max(0, int(plate[1]))
                w  = min(w_img - x, max(10, int(plate[2] - plate[0])))
                h  = min(h_img - y, max(10, int(plate[3] - plate[1])))
                plate_box = [x, y, w, h]

                # Mở rộng biển vuông bị detect thiếu hàng dưới
                plate_box = _extend_square_plate_if_needed(plate_box, car_img)
                x, y, w, h = plate_box

                # PADDING 15% để tránh mất mép khi biển nghiêng
                pad_x = max(6, int(w * 0.15))
                pad_y = max(6, int(h * 0.15))
                x1 = max(0, x - pad_x);    y1 = max(0, y - pad_y)
                x2 = min(w_img, x+w+pad_x); y2 = min(h_img, y+h+pad_y)
                plate_crop_raw = car_img[y1:y2, x1:x2]

                # PERSPECTIVE CORRECTION — chỉ áp dụng nếu biển thực sự nghiêng
                # Đo góc bằng minAreaRect trên vùng sáng của crop
                plate_persp = plate_crop_raw  # mặc định: giữ nguyên
                try:
                    _lab = cv2.cvtColor(plate_crop_raw, cv2.COLOR_BGR2LAB)
                    _l = cv2.split(_lab)[0]
                    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                    _l = _clahe.apply(_l)
                    _, _mask = cv2.threshold(_l, 0, 255,
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    _cnts, _ = cv2.findContours(_mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                    if _cnts:
                        _largest = max(_cnts, key=cv2.contourArea)
                        _rect    = cv2.minAreaRect(_largest)
                        _angle   = abs(_rect[2])          # 0–90° từ minAreaRect
                        # minAreaRect trả 0° khi cạnh dài nằm ngang
                        # → cạnh ngắn nghiêng > 5° là đáng lo
                        if _angle > 5.0 and _angle < 85.0:
                            plate_persp = _perspective_correct(plate_crop_raw)
                except Exception:
                    pass  # lỗi → giữ nguyên crop gốc

                # DESKEW 4 COMBO — y chang reference lp_image.py
                best_crop  = plate_persp.copy()
                best_angle = 999.0
                for change_cons in range(2):
                    for center_thres in range(2):
                        try:
                            deskewed = _deskew(plate_persp, change_cons, center_thres)
                            residual = abs(_compute_skew(deskewed, 0))
                            if residual < best_angle:
                                best_angle = residual
                                best_crop  = deskewed
                        except Exception:
                            pass

                plate_crop = best_crop

                gray_orig = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
                enhanced  = _enhance_contrast(car_img)
                debug_imgs = {
                    'gray':         gray_orig,
                    'enhanced':     enhanced,
                    'blurred':      gray_orig,
                    'edged':        gray_orig,
                    'sobel':        gray_orig,
                    'morphed':      gray_orig,
                    'detected_box': plate_box,
                    'candidates':   [{'box': plate_box, 'score': conf, 'method': 'yolo'}]
                }
                return plate_crop, plate_box, debug_imgs

        except Exception as e:
            print(f'⚠️ YOLO detect lỗi: {e}. Chuyển sang OpenCV fallback...')

    # FALLBACK OPENCV
    return _detect_plate_opencv(car_img)


def _detect_plate_opencv(car_img):
    """
    Phát hiện biển số xe bằng OpenCV pipeline (phương pháp fallback).
    """
    h_img, w_img = car_img.shape[:2]
    
    # 1. CLAHE + Preprocessing
    enhanced = _enhance_contrast(car_img)
    gray, blurred = preprocess_image(enhanced)
    gray_orig, blurred_orig = preprocess_image(car_img)
    
    # Nhị phân hóa toàn cục ảnh xe để phục vụ sinh ứng viên gom cụm & scoring
    thresh_car = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 19, 9
    )
    
    candidates = []
    
    # --- PHƯƠNG PHÁP A: BOTTOM-UP CHARACTER CLUSTERING ---
    clustering_cands = _generate_clustering_candidates(gray, thresh_car, w_img, h_img)
    candidates.extend(clustering_cands)
    
    # --- PHƯƠNG PHÁP B1: SOBEL X + MORPHOLOGY NGANG (2 kích thước kernel) ---
    sobel = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
    _, thresh_sobel = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    for se_size in [(22, 6), (15, 4)]:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, se_size)
        morphed = cv2.morphologyEx(thresh_sobel, cv2.MORPH_CLOSE, se)
        contours_morph, _ = cv2.findContours(morphed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_morph:
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            ar = w / float(h)
            if (w_img * 0.05 < w < w_img * 0.85) and (h_img * 0.018 < h < h_img * 0.35):
                if (1.0 <= ar <= 2.2) or (2.2 < ar <= 6.2):
                    candidates.append({"box": [x, y, w, h], "area": w * h, "aspect_ratio": ar, "method": f"sobel_{se_size[0]}"})
                    
    # --- PHƯƠNG PHÁP B2: SOBEL X + SQUARE MORPHOLOGY (Dành riêng cho biển vuông 2 dòng) ---
    se_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 15))
    morphed_sq = cv2.morphologyEx(thresh_sobel, cv2.MORPH_CLOSE, se_sq)
    contours_morph_sq, _ = cv2.findContours(morphed_sq.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_morph_sq:
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ar = w / float(h)
        if (w_img * 0.08 < w < w_img * 0.75) and (h_img * 0.08 < h < h_img * 0.40):
            if 1.0 <= ar <= 2.2:
                candidates.append({"box": [x, y, w, h], "area": w * h, "aspect_ratio": ar, "method": "sobel_square"})
    
    # --- PHƯƠNG PHÁP C: CANNY + CONTOURS ---
    edged_clahe = cv2.Canny(blurred, 30, 200)
    edged_orig = cv2.Canny(blurred_orig, 30, 200)
    edged = cv2.bitwise_or(edged_clahe, edged_orig)
    
    contours_canny, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_canny:
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ar = w / float(h)
        if (w_img * 0.05 < w < w_img * 0.85) and (h_img * 0.018 < h < h_img * 0.35):
            if (1.0 <= ar <= 2.2) or (2.2 < ar <= 6.2):
                candidates.append({"box": [x, y, w, h], "area": w * h, "aspect_ratio": ar, "method": "canny"})
    
    # --- PHƯƠNG PHÁP D: MSER ---
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        for r in regions:
            hull = cv2.convexHull(r.reshape(-1, 1, 2))
            x, y, w, h = cv2.boundingRect(hull)
            if w == 0 or h == 0:
                continue
            ar = w / float(h)
            if (w_img * 0.05 < w < w_img * 0.85) and (h_img * 0.018 < h < h_img * 0.35):
                if (1.0 <= ar <= 2.2) or (2.2 < ar <= 6.2):
                    candidates.append({"box": [x, y, w, h], "area": w * h, "aspect_ratio": ar, "method": "mser"})
    except Exception:
        pass
        
    # --- TỰ ĐỘNG GỘP CÁC HÀNG ỨNG VIÊN THÀNH BIỂN VUÔNG ---
    merged_cands = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            x1, y1, w1, h1 = candidates[i]["box"]
            x2, y2, w2, h2 = candidates[j]["box"]
            
            # Trùng chéo trục X lớn
            ix1 = max(x1, x2)
            ix2 = min(x1 + w1, x2 + w2)
            overlap_w = max(0, ix2 - ix1)
            overlap_ratio = overlap_w / float(min(w1, w2))
            
            # Khoảng cách trục Y đứng nhỏ
            if y1 + h1 <= y2:
                gap = y2 - (y1 + h1)
            elif y2 + h2 <= y1:
                gap = y1 - (y2 + h2)
            else:
                gap = 0
                
            max_h = max(h1, h2)
            if overlap_ratio > 0.65 and gap < max_h * 1.5:
                mx = min(x1, x2)
                my = min(y1, y2)
                mw = max(x1 + w1, x2 + w2) - mx
                mh = max(y1 + h1, y2 + h2) - my
                mar = mw / float(mh)
                
                if 1.0 <= mar <= 2.2:
                    merged_cands.append({
                       "box": [mx, my, mw, mh],
                       "area": mw * mh,
                       "aspect_ratio": mar,
                       "method": "merged_rows"
                    })
    candidates.extend(merged_cands)
        
    # --- CHẤM ĐIỂM CHI TIẾT CHO TỪNG ỨNG VIÊN ---
    scored_candidates = []
    for cand in candidates:
        score = _score_candidate(cand, h_img, w_img, gray, thresh_car)
        cand["score"] = score
        scored_candidates.append(cand)
        
    # --- ÁP DỤNG NMS ĐỂ LỌC TRÙNG ---
    nms_candidates = _apply_nms(scored_candidates, overlap_thresh=0.40)
    
    best_candidate = nms_candidates[0] if nms_candidates else None
    
    # --- PROJECTION-BASED REFINEMENT ---
    plate_box = None
    if best_candidate is not None:
        x, y, w, h = best_candidate["box"]
        # Padding nhẹ để đảm bảo chứa toàn bộ viền biển
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.12)
        rx = max(0, x - pad_x)
        ry = max(0, y - pad_y)
        rw = min(w_img - rx, w + 2 * pad_x)
        rh = min(h_img - ry, h + 2 * pad_y)
        
        crop_gray = gray_orig[ry:ry + rh, rx:rx + rw]
        if crop_gray.size > 0:
            left, right, top, bottom = _projection_refine(crop_gray)
            new_x = max(0, rx + left)
            new_y = max(0, ry + top)
            new_w = max(10, right - left)
            new_h = max(10, bottom - top)
            
            # Thêm viền an toàn cực mỏng xung quanh biển số thực
            bpx = max(2, int(new_w * 0.015))
            bpy = max(2, int(new_h * 0.015))
            new_x = max(0, new_x - bpx)
            new_y = max(0, new_y - bpy)
            new_w = min(w_img - new_x, new_w + 2 * bpx)
            new_h = min(h_img - new_y, new_h + 2 * bpy)
            plate_box = [new_x, new_y, new_w, new_h]
        else:
            plate_box = [x, y, w, h]
            
    # Fallback nếu hoàn toàn không tìm thấy ứng viên
    if plate_box is None:
        plate_box = [int(w_img * 0.3), int(h_img * 0.55), int(w_img * 0.4), int(h_img * 0.18)]
        
    # Áp dụng tự động mở rộng biển vuông nếu bị thiếu hàng dưới
    plate_box = _extend_square_plate_if_needed(plate_box, car_img)
    x, y, w, h = plate_box
    
    plate_crop = car_img[y:y + h, x:x + w]
    if plate_crop.size == 0:
        plate_crop = car_img[int(h_img * 0.5):, int(w_img * 0.25):int(w_img * 0.75)]
        
    # Tạo morphed nhị phân hoàn chỉnh để debug hiển thị đẹp mắt
    se_deb = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 6))
    morphed_deb = cv2.morphologyEx(thresh_sobel, cv2.MORPH_CLOSE, se_deb)
    
    debug_imgs = {
        "gray": gray_orig,
        "enhanced": enhanced,
        "blurred": blurred,
        "edged": edged,
        "sobel": sobel,
        "morphed": morphed_deb,
        "detected_box": plate_box,
        "candidates": scored_candidates
    }
    
    return plate_crop, plate_box, debug_imgs


def _projection_refine(plate_img_gray):
    """
    Tinh chỉnh viền biển số sát hơn dựa trên chiếu mật độ tích lũy nhị phân.
    Dùng ảnh nhị phân (adaptive threshold) giúp hoàn toàn miễn nhiễm với màu vỏ xe.
    """
    h, w = plate_img_gray.shape[:2]
    if h < 10 or w < 20:
        return 0, w, 0, h
        
    thresh = cv2.adaptiveThreshold(
        plate_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 7
    )
    
    # 1. Chiếu theo chiều dọc (để tìm biên trái/phải)
    col_sums = np.sum(thresh, axis=0)
    col_smooth = np.convolve(col_sums, np.ones(5) / 5.0, mode='same')
    max_c = np.max(col_smooth)
    
    if max_c > 0:
        thresh_col = max_c * 0.08
        active_cols = np.where(col_smooth > thresh_col)[0]
        if len(active_cols) > 0:
            left = int(active_cols[0])
            right = int(active_cols[-1])
        else:
            left, right = 0, w
    else:
        left, right = 0, w
        
    # 2. Chiếu theo chiều ngang (để tìm biên trên/dưới)
    strip = thresh[:, left:right] if right > left else thresh
    row_sums = np.sum(strip, axis=1)
    row_smooth = np.convolve(row_sums, np.ones(3) / 3.0, mode='same')
    max_r = np.max(row_smooth)
    
    if max_r > 0:
        thresh_row = max_r * 0.12
        active_rows = np.where(row_smooth > thresh_row)[0]
        if len(active_rows) > 0:
            top = int(active_rows[0])
            bottom = int(active_rows[-1])
        else:
            top, bottom = 0, h
    else:
        top, bottom = 0, h
        
    # Giới hạn an toàn: không co quá hẹp (tối thiểu chiếm 50% kích thước ban đầu)
    if (right - left) < w * 0.5:
        left, right = 0, w
    if (bottom - top) < h * 0.5:
        top, bottom = 0, h
        
    return left, right, top, bottom


# ========================
# PHÂN ĐOẠN KÝ TỰ
# ========================

def segment_characters(plate_img):
    """
    TÁCH BIỂN SỐ THÀNH TỪNG KÝ TỰ ĐƠN LẺ (TRUYỀN THỐNG - KHÔNG DÙNG ML)
    Theo đúng yêu cầu của giảng viên: Plate Regions => Split into single digits.
    
    Phương pháp xử lý ảnh (OpenCV pipeline):
    1. Tiền xử lý: CLAHE nâng tương phản kênh LAB + Lọc nhiễu Bilateral Filter.
    2. Căn thẳng tích cực: PERSPECTIVE CORRECTION (nắn biển lệch góc) + Hough Lines P.
    3. Nhị phân hóa thông minh: Adaptive Thresholding (Gaussian) tự động đảo ngược
       để đảm bảo chữ luôn màu trắng (255) trên nền đen (0).
    4. Adaptive Morphology: Dùng phép toán đóng (MORPH_CLOSE) với kernel (3,3) 
       để nối các vết rạn nứt nhỏ.
    5. Trích xuất đường bao (Contours): Tìm toàn bộ contour biên ngoài (RETR_EXTERNAL).
    6. Bộ lọc hình học thô: Giữ lại các contour có tỉ lệ chiều cao, chiều rộng và diện tích hợp lý.
    7. Thuật toán GỘP DỌC thông minh (Vertical Merging): Tìm các nét đứt xếp chồng lên nhau 
       (như số 7 bị đứt thanh ngang, số 0 đứt nét dọc), tự động gộp thành 1 khung chữ duy nhất.
    8. Thuật toán TÁCH NGANG thông minh (Horizontal Splitting): Phát hiện các ký tự bị dính nhau
       (aspect ratio w/h quá to > 0.8), tự động chia đôi hoặc chia ba theo chiều ngang.
    9. Sắp xếp thứ tự đọc khoa học: 1 hàng (sắp xếp theo X), 2 hàng (chia nhóm Y rồi sắp xếp theo X).
    10. Chuẩn hóa kích thước: Cắt ký tự, tạo viền đệm an toàn (padding 15%) và resize về 32x32.
    """
    if plate_img is None or plate_img.size == 0:
        return _make_fallback_chars(8), np.zeros((40, 100), dtype=np.uint8)
        
    h_plate, w_plate = plate_img.shape[:2]
    if h_plate < 8 or w_plate < 8:
        return _make_fallback_chars(8), np.zeros((h_plate, w_plate), dtype=np.uint8)
        
    # Bước 1: Phóng to ảnh biển số nhỏ để xử lý chính xác hơn
    scale = 1.0
    if h_plate < 60:
        scale = 90.0 / h_plate
        plate_img = cv2.resize(plate_img, (int(w_plate * scale), 90))
        h_plate, w_plate = plate_img.shape[:2]
        
    # Bước 2: Tăng cường tương phản (CLAHE) + Khử nhiễu
    enhanced_plate = _enhance_contrast(plate_img)
    
    # Bước 2.5: PERSPECTIVE CORRECTION TÍCH CỰC cho biển đặc biệt lệch góc
    # Cơ chế: Nếu dữ liệu thay đổi quá nhanh → có khả năng lệch góc nặng
    persp_corrected = _perspective_correct(enhanced_plate)

    # Bước 3: Căn thẳng biển nghiêng (Deskew)
    deskewed = persp_corrected  # Dùng phiên bản đã correct perspective
    if h_plate >= 20 and w_plate >= 40:
        deskewed = deskew_plate(persp_corrected, use_clahe=False, center_thres=0)

    # Bước 4: Chuyển xám + Lọc Bilateral
    gray, blurred = preprocess_image(deskewed)
    
    # Bước 5: Nhị phân hóa thích nghi (Adaptive Threshold)
    thresh = threshold_image(gray)
    
    # Đảm bảo chữ luôn màu trắng (255) trên nền đen (0)
    border_pixels = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if np.mean(border_pixels) > 127:
        thresh = cv2.bitwise_not(thresh)
        
    # Bước 6: Phép toán đóng Morphology để nối liền các đứt nét nhỏ (dung sai 2-3 pixel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Phân loại biển để biết cách sắp xếp sau này
    plate_shape, _, _ = classify_plate_type(deskewed)
    
    # Bước 7: Trích xuất contours và lọc hình học thô ban đầu (cho phép nét đứt đi qua)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = _filter_char_contours(contours, h_plate, w_plate)
    
    # Bước 8: Áp dụng thuật toán Gộp Dọc thông minh (Vertical Merging)
    # Chạy 2 lần để giải quyết các ký tự nứt vỡ phức tạp thành nhiều mảnh
    merged_boxes = _vertical_merge_boxes(raw_boxes, h_plate)
    merged_boxes = _vertical_merge_boxes(merged_boxes, h_plate)
    
    # Bước 9: Áp dụng thuật toán Tách Ngang thông minh (Horizontal Splitting) cho các chữ dính nhau
    final_boxes = []
    for (x, y, w, h) in merged_boxes:
        ar = w / float(h)
        h_ratio = h / float(h_plate)
        
        # Nếu box quá lùn (<20% chiều cao biển), loại bỏ nhiễu đứng đơn lẻ
        if h_ratio < 0.20:
            continue
            
        # Nếu aspect ratio w/h quá lớn (> 0.8), có khả năng chứa chữ dính nhau
        if ar >= 0.85:
            # Nếu AR trong khoảng [0.85, 1.6], khả năng chứa 2 chữ dính nhau -> chia đôi
            if ar < 1.6:
                w_half = int(w / 2)
                final_boxes.append((x, y, w_half, h))
                final_boxes.append((x + w_half, y, w - w_half, h))
            # Nếu AR >= 1.6, chia ba
            else:
                w_third = int(w / 3)
                final_boxes.append((x, y, w_third, h))
                final_boxes.append((x + w_third, y, w_third, h))
                final_boxes.append((x + 2 * w_third, y, w - 2 * w_third, h))
        else:
            final_boxes.append((x, y, w, h))
            
    # Lọc hình học lần cuối sau khi phân chia: Adaptive Median Height + fill ratio + border fragment
    char_boxes = []
    if final_boxes:
        heights = [h for (x, y, w, h) in final_boxes]
        median_h = np.median(heights) if len(heights) >= 3 else h_plate * 0.5
        for (x, y, w, h) in final_boxes:
            h_ratio = h / float(h_plate)
            ar = w / float(h)
            # Bỏ qua quá thấp (dấu chấm, gạch ngang, nhiễu)
            if h < median_h * 0.70:
                continue
            # Bỏ qua nét quá mảnh / quá dẹt
            if ar < 0.13 or ar > 1.15:
                continue
            # Khử viền dọc sát mép trái/phải
            if (x < w_plate * 0.03 or x + w > w_plate * 0.97) and ar < 0.22:
                continue
            # Lọc fill ratio: ký tự thật luôn có >=8% pixel sáng trong bounding box
            region = thresh[max(0,y):min(thresh.shape[0],y+h),
                            max(0,x):min(thresh.shape[1],x+w)]
            if region.size > 0:
                fill = np.count_nonzero(region) / float(region.size)
                if fill < 0.08:   # quá thưa -> nhiễu / border fragment
                    continue
            char_boxes.append((x, y, w, h))
            
    # Nếu không tìm thấy đủ ký tự, thử lại với Otsu threshold + Morph
    if len(char_boxes) < 3:
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Đảm bảo chữ trắng nền đen
        border_pixels2 = np.concatenate([thresh2[0, :], thresh2[-1, :], thresh2[:, 0], thresh2[:, -1]])
        if np.mean(border_pixels2) > 127:
            thresh2 = cv2.bitwise_not(thresh2)
            
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel2)
        contours2, _ = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes2 = _filter_char_contours(contours2, h_plate, w_plate)
        merged_boxes2 = _vertical_merge_boxes(raw_boxes2, h_plate)
        merged_boxes2 = _vertical_merge_boxes(merged_boxes2, h_plate)
        
        char_boxes2 = []
        if merged_boxes2:
            heights2 = [h for (x, y, w, h) in merged_boxes2]
            median_h2 = np.median(heights2) if len(heights2) >= 3 else h_plate * 0.5
            for (x, y, w, h) in merged_boxes2:
                h_ratio = h / float(h_plate)
                ar = w / float(h)
                if h < median_h2 * 0.70:
                    continue
                if ar < 0.13 or ar > 1.15:
                    continue
                if (x < w_plate * 0.03 or x + w > w_plate * 0.97) and ar < 0.22:
                    continue
                # Fill ratio: ký tự thật >=8% pixel sáng
                region2 = thresh2[max(0,y):min(thresh2.shape[0],y+h),
                                  max(0,x):min(thresh2.shape[1],x+w)]
                if region2.size > 0 and np.count_nonzero(region2) / float(region2.size) < 0.08:
                    continue
                char_boxes2.append((x, y, w, h))
                
        if len(char_boxes2) > len(char_boxes):
            char_boxes = char_boxes2
            thresh = thresh2
            
    # Bước 10: Giới hạn số ký tự tối đa theo loại biển
    # Biển 1 hàng VN: 7-9 ký tự; Biển 2 hàng: 8-9 ký tự
    MAX_CHARS = 9
    if len(char_boxes) > MAX_CHARS:
        # Chọn MAX_CHARS ký tự có chiều cao lớn nhất (gần median nhất)
        med = np.median([h for (x, y, w, h) in char_boxes])
        def _char_score(box):
            x, y, w, h = box
            ar = w / float(h)
            region = thresh[max(0,y):min(thresh.shape[0],y+h),
                            max(0,x):min(thresh.shape[1],x+w)]
            fill = np.count_nonzero(region) / float(region.size) if region.size > 0 else 0
            height_score = -abs(h - med) / float(med)  # gần median -> tốt hơn
            fill_score   = fill
            ar_score     = -abs(ar - 0.5)              # gần 0.5 -> ký tự cân đối
            return height_score + fill_score + ar_score
        char_boxes = sorted(char_boxes, key=_char_score, reverse=True)[:MAX_CHARS]

    # Bước 11: Sắp xếp ký tự theo thứ tự đọc tự nhiên
    sorted_boxes = _sort_chars(char_boxes, plate_shape)
    
    # Bước 12: Cắt ảnh từng ký tự, tạo viền đệm (padding 15%) và chuẩn hóa về 28x28
    sorted_chars = []
    for (x, y, w, h) in sorted_boxes:
        # Đảm bảo không vượt quá biên ảnh biển
        x, y = max(0, x), max(0, y)
        w = min(w_plate - x, max(2, w))
        h = min(h_plate - y, max(2, h))
        
        char_crop = thresh[y:y + h, x:x + w]
        if char_crop.size == 0:
            continue
            
        # Thêm viền đen đệm xung quanh để tránh nét vẽ chạm mép ảnh 28x28
        pad = max(3, int(max(w, h) * 0.15))
        char_padded = cv2.copyMakeBorder(char_crop, pad, pad, pad, pad,
                                          cv2.BORDER_CONSTANT, value=0)
        # Resize về 32×32 để tương thích với HOG engine (img_size=(32,32))
        char_resized = cv2.resize(char_padded, (32, 32), interpolation=cv2.INTER_AREA)
        sorted_chars.append({"image": char_resized, "box": [x, y, w, h]})
        
    if not sorted_chars:
        sorted_chars = _make_fallback_chars(8)
        
    return sorted_chars, thresh


def _filter_char_contours(contours, h_plate, w_plate):
    """
    Lọc contours thô ban đầu.
    Hạ chiều cao tối thiểu một chút (từ 0.18 xuống 0.08) để cho phép
    các nét đứt (như thanh ngang của số 7) đi qua khâu lọc đầu trước khi được gộp dọc.
    """
    char_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        h_ratio = h / float(h_plate)
        w_ratio = w / float(w_plate)
        ar = w / float(h)
        if (0.08 < h_ratio < 0.96 and
                0.01 < w_ratio < 0.48 and
                ar < 2.5 and
                w * h > 10):
            char_boxes.append((x, y, w, h))
    return char_boxes


def _vertical_merge_boxes(boxes, h_plate):
    """
    Tự động phát hiện các nét ký tự bị đứt rời đứng xếp chồng lên nhau (như số 7, số 0)
    và gộp chúng thành một bounding box ký tự duy nhất dựa trên trùng chéo X và khoảng cách Y.
    """
    if len(boxes) < 2:
        return boxes
        
    merged_boxes = []
    used = set()
    
    # Duyệt và tìm các cặp có thể gộp dọc
    for i in range(len(boxes)):
        if i in used:
            continue
        x1, y1, w1, h1 = boxes[i]
        
        best_j = -1
        min_gap = 9999
        
        for j in range(len(boxes)):
            if j == i or j in used:
                continue
            x2, y2, w2, h2 = boxes[j]
            
            # 1. Tính độ trùng ngang (Horizontal Overlap)
            ix1 = max(x1, x2)
            ix2 = min(x1 + w1, x2 + w2)
            overlap_w = max(0, ix2 - ix1)
            overlap_ratio = overlap_w / float(min(w1, w2))
            
            # 2. Tính khoảng cách dọc (Vertical Gap)
            if y1 + h1 <= y2:
                gap = y2 - (y1 + h1)
            elif y2 + h2 <= y1:
                gap = y1 - (y2 + h2)
            else:
                gap = 0  # Giao cắt dọc
                
            # Heuristics: trùng ngang cao (>0.45) và khoảng cách dọc cực bé (<18% chiều cao biển hoặc <8 pixel)
            if overlap_ratio > 0.45 and gap < max(8, int(h_plate * 0.18)):
                if gap < min_gap:
                    min_gap = gap
                    best_j = j
                    
        if best_j != -1:
            x2, y2, w2, h2 = boxes[best_j]
            mx = min(x1, x2)
            my = min(y1, y2)
            mw = max(x1 + w1, x2 + w2) - mx
            mh = max(y1 + h1, y2 + h2) - my
            
            merged_ar = mw / float(mh)
            merged_h_ratio = mh / float(h_plate)
            
            # Đảm bảo box sau gộp có tỉ lệ hình chữ nhật đứng và kích thước hợp lý
            if merged_ar < 1.15 and 0.25 < merged_h_ratio < 0.95:
                merged_boxes.append((mx, my, mw, mh))
                used.add(i)
                used.add(best_j)
                
    # Thêm các box không bị gộp vào danh sách kết quả
    for idx in range(len(boxes)):
        if idx not in used:
            merged_boxes.append(boxes[idx])
            
    return merged_boxes


def _sort_chars(char_boxes, plate_shape):
    """Sắp xếp ký tự theo thứ tự đọc."""
    if not char_boxes:
        return []
    
    if "Rectangle" in plate_shape:
        return sorted(char_boxes, key=lambda b: b[0])
    else:
        # Biển 2 hàng: tách thành hàng trên và hàng dưới dựa trên sự kết hợp giữa mean Y và trung điểm vật lý
        h_plate = max([b[1] + b[3] for b in char_boxes])  # Ước lượng chiều cao thực tế chứa ký tự
        y_centers = [b[1] + b[3] / 2.0 for b in char_boxes]
        # Sử dụng trọng số trung bình cộng giúp cân bằng hoàn hảo khi số chữ hai hàng lệch nhau
        mid_y = (np.mean(y_centers) + h_plate / 2.0) / 2.0
        
        row1 = sorted([b for b in char_boxes if b[1] + b[3] / 2.0 < mid_y], key=lambda b: b[0])
        row2 = sorted([b for b in char_boxes if b[1] + b[3] / 2.0 >= mid_y], key=lambda b: b[0])
        return row1 + row2


def _make_fallback_chars(n, w=100, h=40):
    """Tạo ký tự placeholder khi không tách được."""
    chars = []
    for i in range(n):
        mock = np.zeros((28, 28), dtype=np.uint8)
        cv2.putText(mock, "?", (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        chars.append({"image": mock,
                       "box": [i * max(1, w // (n + 1)) + 3, h // 5, max(5, w // (n + 1)), h * 3 // 5]})
    return chars


# ========================
# VẼ KẾT QUẢ
# ========================

def draw_plate_on_car(car_img, plate_box, plate_text="", plate_type=""):
    """Vẽ khung biển số và nhãn lên ảnh xe."""
    img_draw = car_img.copy()
    x, y, w, h = [int(v) for v in plate_box]
    color = (0, 255, 170)
    
    cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 3)
    
    label = plate_text
    if plate_type:
        label = f"{plate_text}  [{plate_type}]"
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(0.85, w / 300.0))
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(th + 14, y)
        cv2.rectangle(img_draw, (x, label_y - th - 12), (x + tw + 10, label_y), color, -1)
        cv2.putText(img_draw, label, (x + 5, label_y - 5),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    return img_draw


def draw_characters_on_plate(plate_img, char_boxes, recognized_chars=None):
    """Vẽ khung bao quanh từng ký tự được phân đoạn."""
    img_draw = plate_img.copy()
    if len(img_draw.shape) == 2:
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)
    
    for i, box in enumerate(char_boxes):
        x, y, w, h = [int(v) for v in box["box"]]
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 200, 255), 2)
        if recognized_chars and i < len(recognized_chars):
            cv2.putText(img_draw, recognized_chars[i],
                        (x, max(0, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)
    
    return img_draw


def _aggressive_straighten_plate(plate_img):
    """
    Xử lý tích cực cho biển số bị lệch góc nhìn (skewed view).
    Áp dụng kết hợp perspective correction + multiple deskew attempts để tìm góc tốt nhất.

    Trả về tuple: (straightened_img, was_corrected)
    - straightened_img: ảnh sau khi được nắn thẳng
    - was_corrected: True nếu đã áp dụng perspective correction hoặc deskew đáng kể
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img, False

    h, w = plate_img.shape[:2]
    if h < 15 or w < 30:
        return plate_img, False

    # Thử perspective correction trước
    persp = _perspective_correct(plate_img)
    was_corrected = (persp.shape != plate_img.shape or not np.array_equal(persp, plate_img))

    # Sau đó thử deskew với nhiều tổ hợp tham số để tìm góc tốt nhất
    best_img = persp
    best_angle = 999.0

    for use_clahe in [True, False]:
        for center_thres in [0, 1]:
            try:
                deskewed_trial = _deskew(persp, int(use_clahe), center_thres)
                residual = abs(_compute_skew(deskewed_trial, 0))
                if residual < best_angle:
                    best_angle = residual
                    best_img = deskewed_trial
                    was_corrected = True
            except Exception:
                pass

    return best_img, was_corrected
