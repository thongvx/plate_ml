import os
import cv2
import numpy as np

def create_sample_images(output_dir="/Users/thongvuong/plate_ml/sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Định nghĩa danh sách các ảnh mẫu cần tạo
    # Cấu trúc: [Tên file, Biển số xe, Màu nền biển (trắng/vàng), Loại xe (car/bike)]
    samples = [
        ("car_allowed_1.jpg", "30F-888.88", (255, 255, 255), "car"), # Trắng - Ô tô
        ("car_allowed_2.jpg", "29-A1 123.45", (255, 255, 255), "bike"), # Trắng - Xe máy
        ("car_blocked.jpg", "51A-999.99", (255, 255, 255), "car"), # Trắng - Ô tô
        ("car_allowed_3.jpg", "88A-777.77", (100, 230, 255), "car") # Vàng - Ô tô (biển số màu vàng xe kinh doanh)
    ]
    
    for filename, plate_text, bg_color, vehicle_type in samples:
        # Tạo ảnh nền canvas kích thước 300x500
        img = np.ones((300, 500, 3), dtype=np.uint8) * 40 # Nền xám đen
        
        # Vẽ một số họa tiết biểu diễn hậu cảnh/gara đường phố
        cv2.line(img, (0, 220), (500, 220), (80, 80, 80), 2)
        cv2.line(img, (100, 220), (0, 300), (80, 80, 80), 1)
        cv2.line(img, (400, 220), (500, 300), (80, 80, 80), 1)
        
        if vehicle_type == "car":
            # --- VẼ HÌNH Ô TÔ GIẢ LẬP ---
            # Thân xe (màu đỏ neon hoặc xanh dương)
            color_car = (200, 50, 50) if "blocked" in filename else (50, 150, 220)
            # Mái xe & kính chắn gió
            pts = np.array([[150, 90], [350, 90], [390, 140], [110, 140]], np.int32)
            cv2.fillPoly(img, [pts], (20, 20, 20)) # Kính đen
            
            # Thân xe chính
            cv2.rectangle(img, (90, 140), (410, 210), color_car, -1)
            # Bumper (Cản trước xe chứa biển số)
            cv2.rectangle(img, (100, 200), (400, 230), (30, 30, 30), -1)
            
            # Đèn pha hai bên
            cv2.circle(img, (120, 170), 18, (255, 255, 200), -1) # Pha trái
            cv2.circle(img, (380, 170), 18, (255, 255, 200), -1) # Pha phải
            cv2.circle(img, (120, 170), 18, (0, 200, 255), 2)
            cv2.circle(img, (380, 170), 18, (0, 200, 255), 2)
            
            # Bánh xe (lốp xe lộ một góc dưới)
            cv2.circle(img, (130, 235), 22, (10, 10, 10), -1)
            cv2.circle(img, (370, 235), 22, (10, 10, 10), -1)
            
            # Vẽ Biển Số Ô Tô nằm giữa cản xe (Rectangle / 1-line plate)
            # Kích thước biển số ~ 140 x 38
            px, py, pw, ph = 180, 205, 140, 38
            cv2.rectangle(img, (px, py), (px+pw, py+ph), bg_color, -1) # Nền biển số
            cv2.rectangle(img, (px, py), (px+pw, py+ph), (0, 0, 0), 2) # Viền đen
            
            # Render chữ số biển xe cực nét lên biển số
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            thickness = 2
            # Căn giữa chữ
            (tw, th), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
            tx = px + int((pw - tw) / 2)
            ty = py + int((ph + th) / 2)
            cv2.putText(img, plate_text, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
        else:
            # --- VẼ HÌNH XE MÁY GIẢ LẬP (BIỂN VUÔNG 2 HÀNG) ---
            # Thân xe máy (Mô phỏng mặt trước xe Honda SH / Vespa)
            cv2.rectangle(img, (200, 100), (300, 220), (120, 120, 140), -1) # Khung đầu xe
            # Kính chắn gió xe máy
            pts = np.array([[220, 60], [280, 60], [290, 100], [210, 100]], np.int32)
            cv2.fillPoly(img, [pts], (30, 30, 30))
            
            # Đèn xe máy
            cv2.rectangle(img, (230, 110), (270, 130), (255, 255, 230), -1)
            
            # Yếm xe phía dưới
            pts_yem = np.array([[170, 160], [330, 160], [280, 240], [220, 240]], np.int32)
            cv2.fillPoly(img, [pts_yem], (180, 30, 50)) # Yếm đỏ
            
            # Bánh xe trước
            cv2.rectangle(img, (238, 230), (262, 290), (15, 15, 15), -1)
            
            # Vẽ Biển Số Xe Máy nằm trên yếm xe (Square / 2-line plate)
            # Biển vuông Việt Nam có kích thước ~ 80 x 60
            px, py, pw, ph = 210, 165, 80, 60
            cv2.rectangle(img, (px, py), (px+pw, py+ph), bg_color, -1) # Nền
            cv2.rectangle(img, (px, py), (px+pw, py+ph), (0, 0, 0), 2) # Viền
            
            # Biển vuông có 2 hàng chữ:
            # Hàng 1: Mã tỉnh & Chữ cái (e.g. 29-A1)
            # Hàng 2: Số sê-ri (e.g. 123.45)
            parts = plate_text.split(" ")
            if len(parts) == 2:
                row1_text, row2_text = parts[0], parts[1]
            else:
                row1_text, row2_text = plate_text[:5], plate_text[5:]
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 2
            
            # Vẽ Hàng 1
            (w1, h1), _ = cv2.getTextSize(row1_text, font, font_scale, thickness)
            tx1 = px + int((pw - w1) / 2)
            ty1 = py + int((ph/2 - h1)/2) + h1 + 4
            cv2.putText(img, row1_text, (tx1, ty1), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
            # Vẽ Hàng 2
            (w2, h2), _ = cv2.getTextSize(row2_text, font, font_scale, thickness)
            tx2 = px + int((pw - w2) / 2)
            ty2 = py + int(ph/2) + int((ph/2 - h2)/2) + h2 - 2
            cv2.putText(img, row2_text, (tx2, ty2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
        # Ghi ảnh mẫu ra đĩa
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    create_sample_images()
    print("Successfully generated mock vehicle sample images!")
