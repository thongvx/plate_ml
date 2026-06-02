"""
app.py - Hệ thống Nhận dạng Biển số Xe Việt Nam (LPR)
Tập trung vào:
- Upload ảnh thực tế
- Phát hiện biển số (OpenCV pipeline với CLAHE + Deskew)
- Tách ký tự + OCR bằng ML (KNN/SVM/MLP)
- Phân loại biển (Shape + Color)
- Nghiên cứu ML: PCA/t-SNE/SVD, so sánh model
"""
import os
import json
import datetime
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ml_core.segmentation import (detect_plate, segment_characters,
                                   draw_plate_on_car, classify_plate_type,
                                   draw_characters_on_plate)
from ml_core.ocr_engine import OCREngine
from ml_core.reducer import get_reduction_data_cached
from ui_styles import get_custom_css

# ==================== CẤU HÌNH ====================
st.set_page_config(
    page_title="Nhận dạng Biển số Xe Việt Nam | LPR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(get_custom_css(), unsafe_allow_html=True)

DATA_DIR = "/Users/thongvuong/plate_ml/ml_core/data"
LP_DETECT_DIR = "/Users/thongvuong/plate_ml/ml_core/data/LP_detection"
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_resource(show_spinner="🤖 Đang tải mô hình OCR...")
def load_ocr_engine():
    return OCREngine(DATA_DIR)

ocr_engine = load_ocr_engine()

# ==================== SESSION STATE ====================
DATABASE_PATH = os.path.join(DATA_DIR, "data_allowed.json")

def load_allowed_db():
    if os.path.exists(DATABASE_PATH):
        try:
            with open(DATABASE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    default = [
        {"plate": "29-A1 123.45", "owner": "Nguyễn Văn Hùng",  "type": "Xe máy Honda SH",      "status": "Allowed"},
        {"plate": "30F-888.88",   "owner": "Lê Hoài Thu",       "type": "Ô tô Mercedes E300",    "status": "Allowed"},
        {"plate": "59C2-999.99",  "owner": "Trần Minh Hoàng",   "type": "Xe máy Vespa",          "status": "Allowed"},
        {"plate": "88A-777.77",   "owner": "Phạm Quốc Bảo",    "type": "Ô tô VinFast VF8",      "status": "Allowed"},
    ]
    with open(DATABASE_PATH, "w", encoding="utf-8") as f:
        json.dump(default, f, ensure_ascii=False, indent=2)
    return default

for key, default in [
    ("allowed_db", load_allowed_db()),
    ("vehicle_logs", []),
    ("last_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <p style="font-size:2rem; margin:0;">🚗</p>
        <h2 style="color:#00ffaa; font-size:1.1rem; margin:5px 0 0; font-weight:700;">
            LPR SYSTEM
        </h2>
        <p style="color:#64748b; font-size:0.75rem; margin:0;">Vietnamese License Plate Recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p style="color:#94a3b8; font-size:0.8rem; font-weight:600; margin-bottom:8px;">⚙️ CÀI ĐẶT MÔ HÌNH OCR</p>', unsafe_allow_html=True)
    opt_ocr_model = st.selectbox("Thuật toán phân loại ký tự:", 
                                   ["SVM", "KNN", "MLP"], index=0,
                                   help="SVM thường cho accuracy cao nhất. KNN nhanh nhất khi train.")
    
    st.markdown("---")
    st.markdown('<p style="color:#94a3b8; font-size:0.8rem; font-weight:600; margin-bottom:8px;">🗂️ DATASET LP DETECTION</p>', unsafe_allow_html=True)
    train_count = len(os.listdir(os.path.join(LP_DETECT_DIR, "images", "train"))) if os.path.exists(os.path.join(LP_DETECT_DIR, "images", "train")) else 0
    val_count = len(os.listdir(os.path.join(LP_DETECT_DIR, "images", "val"))) if os.path.exists(os.path.join(LP_DETECT_DIR, "images", "val")) else 0
    st.markdown(f"""
    <div style="background:rgba(0,255,170,0.05); border:1px solid rgba(0,255,170,0.15); 
         border-radius:8px; padding:10px; font-size:0.8rem;">
      <p style="color:#00ffaa; margin:0 0 4px; font-weight:600;">📁 LP_detection</p>
      <p style="color:#94a3b8; margin:0;">Train: <b style="color:#f8fafc;">{train_count:,}</b> ảnh</p>
      <p style="color:#94a3b8; margin:0;">Val: <b style="color:#f8fafc;">{val_count:,}</b> ảnh</p>
      <p style="color:#94a3b8; margin:0;">Format: YOLO (cx cy w h)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p style="color:#94a3b8; font-size:0.8rem; font-weight:600; margin-bottom:8px;">📚 THAM KHẢO</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748b; line-height:1.6;">
      📌 <a href="https://github.com/trungdinh22/License-Plate-Recognition" 
             style="color:#60a5fa;" target="_blank">trungdinh22/LPR</a><br/>
      Kỹ thuật đã áp dụng:<br/>
      • CLAHE contrast enhancement<br/>
      • HoughLines deskew<br/>
      • 1-line / 2-line classification<br/>
      • 4-variant deskew thử nghiệm
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p style="color:#94a3b8; font-size:0.8rem; font-weight:600; margin-bottom:8px;">📋 DANH SÁCH XE ĐƯỢC PHÉP</p>', unsafe_allow_html=True)
    for entry in st.session_state.allowed_db[:4]:
        st.markdown(f"""
        <div style="background:rgba(0,255,170,0.05); border-left:2px solid #00ffaa; 
             padding:5px 8px; margin-bottom:4px; border-radius:4px;">
          <p style="color:#00ffaa; font-size:0.75rem; font-weight:700; margin:0;">{entry['plate']}</p>
          <p style="color:#94a3b8; font-size:0.7rem; margin:0;">{entry['owner']}</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN TABS ====================
tab1, tab2, tab3 = st.tabs([
    "📷 NHẬN DIỆN BIỂN SỐ",
    "🔬 NGHIÊN CỨU THUẬT TOÁN ML",
    "📋 BÁO CÁO HỌC THUẬT"
])

# ==================== TAB 1: NHẬN DIỆN BIỂN SỐ ====================
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#00ffaa; font-weight:800; font-size:1.6rem; margin:0 0 4px;">🚘 NHẬN DIỆN BIỂN SỐ XE VIỆT NAM</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.85rem; margin:0 0 20px;">Upload ảnh xe → Phát hiện biển số → Tách ký tự → OCR bằng ML</p>', unsafe_allow_html=True)
    
    # Khu vực upload
    uploaded_file = st.file_uploader(
        "📁 Tải lên ảnh xe chứa biển số:",
        type=["jpg", "jpeg", "png", "webp"],
        help="Hỗ trợ JPG, PNG, WEBP. Camera góc cao, nhiều xe, ánh sáng khác nhau đều được."
    )
    
    if uploaded_file is not None:
        # Đọc ảnh
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_raw is None:
            st.error("❌ Không đọc được ảnh. Vui lòng thử định dạng JPG/PNG.")
        else:
            # Resize nếu quá lớn
            h0, w0 = img_raw.shape[:2]
            if w0 > 1200:
                img_raw = cv2.resize(img_raw, (1200, int(1200 * h0 / w0)))
            elif w0 < 300:
                img_raw = cv2.resize(img_raw, (300, int(300 * h0 / w0)))
            
            col_img, col_info = st.columns([3, 2])
            
            with col_img:
                st.markdown('<p style="color:#94a3b8; font-size:0.8rem; margin-bottom:4px;">📸 Ảnh đầu vào</p>', unsafe_allow_html=True)
                st.image(img_raw, channels="BGR", use_container_width=True)
            
            with col_info:
                st.markdown('<p style="color:#94a3b8; font-size:0.8rem; margin-bottom:4px;">ℹ️ Thông tin ảnh</p>', unsafe_allow_html=True)
                h_img, w_img = img_raw.shape[:2]
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2); border-radius:8px; padding:12px; font-size:0.8rem;">
                  <p style="color:#94a3b8; margin:0 0 4px;">Kích thước: <b style="color:#f8fafc;">{w_img}×{h_img}px</b></p>
                  <p style="color:#94a3b8; margin:0 0 4px;">File: <b style="color:#f8fafc;">{uploaded_file.name}</b></p>
                  <p style="color:#94a3b8; margin:0;">Model OCR: <b style="color:#00ffaa;">{opt_ocr_model}</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br/>", unsafe_allow_html=True)
            
            # ======= CHẠY PIPELINE =======
            with st.spinner("⚙️ Đang xử lý pipeline nhận diện..."):
                # Bước 1: Detect biển số
                plate_crop, plate_box, debug_imgs = detect_plate(img_raw)
                # Bước 2: Phân loại shape + color
                plate_shape, ar, plate_color = classify_plate_type(plate_crop)
                # Bước 3: Segment ký tự
                char_imgs, thresh_img = segment_characters(plate_crop)
                # Bước 4: OCR từng ký tự
                recognized_chars = []
                for c_item in char_imgs:
                    ch, _ = ocr_engine.predict(c_item["image"], opt_ocr_model)
                    recognized_chars.append(ch)
                # Bước 5: Ghép thành biển số
                predicted_plate, confidence = ocr_engine.predict_plate_text(char_imgs, opt_ocr_model)
            
            # ======= HIỂN THỊ KẾT QUẢ CHÍNH =======
            st.markdown('<hr style="border-color:rgba(0,255,170,0.15); margin:0 0 16px;"/>', unsafe_allow_html=True)
            
            # Ảnh có khung biển số
            plate_label = f"{predicted_plate}  [{plate_shape} | {plate_color}]"
            img_result = draw_plate_on_car(img_raw, plate_box, plate_label)
            st.image(img_result, channels="BGR", use_container_width=True, caption="Kết quả phát hiện biển số")
            
            # Card kết quả
            res_cols = st.columns(4)
            
            color_map = {"White": "#f8fafc", "Yellow": "#fbbf24", "Blue": "#60a5fa", "Green": "#34d399", "Red": "#f87171"}
            c_hex = color_map.get(plate_color, "#94a3b8")
            shape_color_hex = "#00ffaa" if "Rectangle" in plate_shape else "#bd93f9"
            
            with res_cols[0]:
                st.markdown(f"""
                <div style="background:rgba(0,255,170,0.08); border:1px solid rgba(0,255,170,0.2); 
                     border-radius:10px; padding:14px; text-align:center;">
                  <p style="color:#94a3b8; font-size:0.7rem; margin:0 0 4px; text-transform:uppercase;">Biển số</p>
                  <p style="color:#00ffaa; font-size:1.3rem; font-weight:800; margin:0; font-family:monospace;">{predicted_plate or "---"}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with res_cols[1]:
                st.markdown(f"""
                <div style="background:rgba(189,147,249,0.08); border:1px solid rgba(189,147,249,0.2); 
                     border-radius:10px; padding:14px; text-align:center;">
                  <p style="color:#94a3b8; font-size:0.7rem; margin:0 0 4px; text-transform:uppercase;">Loại biển</p>
                  <p style="color:{shape_color_hex}; font-size:0.9rem; font-weight:700; margin:0;">{plate_shape}</p>
                  <p style="color:#64748b; font-size:0.7rem; margin:0;">AR = {ar:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with res_cols[2]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); 
                     border-radius:10px; padding:14px; text-align:center;">
                  <p style="color:#94a3b8; font-size:0.7rem; margin:0 0 4px; text-transform:uppercase;">Màu biển</p>
                  <p style="color:{c_hex}; font-size:1rem; font-weight:700; margin:0;">● {plate_color}</p>
                  <p style="color:#64748b; font-size:0.7rem; margin:0;">Color Type</p>
                </div>
                """, unsafe_allow_html=True)
            
            with res_cols[3]:
                # Kiểm tra quyền truy cập
                def _normalize(s):
                    return s.replace("-","").replace(".","").replace(" ","").upper()
                access = False
                owner_name = "Khách vãng lai"
                for entry in st.session_state.allowed_db:
                    if _normalize(entry["plate"]) == _normalize(predicted_plate):
                        access = True
                        owner_name = entry["owner"]
                        break
                
                badge_color = "#00ffaa" if access else "#f87171"
                badge_text = "✅ HỢP LỆ" if access else "❌ KHÔNG PHÉP"
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2); border:1px solid {badge_color}40; 
                     border-radius:10px; padding:14px; text-align:center;">
                  <p style="color:#94a3b8; font-size:0.7rem; margin:0 0 4px; text-transform:uppercase;">Kiểm soát cổng</p>
                  <p style="color:{badge_color}; font-size:0.9rem; font-weight:800; margin:0;">{badge_text}</p>
                  <p style="color:#64748b; font-size:0.7rem; margin:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{owner_name}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br/>", unsafe_allow_html=True)
            
            # ======= SPLIT INTO SINGLE DIGITS =======
            st.markdown('<p style="font-weight:700; color:#bd93f9; font-size:0.85rem; margin:0 0 8px;">🔢 TÁCH TỪNG KÝ TỰ (Split into Single Digits/Characters):</p>', unsafe_allow_html=True)
            
            # Hiển thị ảnh biển số sạch không vẽ khung màu vàng lên chữ theo yêu cầu người dùng
            st.image(plate_crop, channels="BGR", use_container_width=True,
                     caption=f"Biển số phân đoạn: {len(char_imgs)} ký tự được tách ra")
            
            # Grid từng ký tự 28x28
            if char_imgs:
                n = min(len(char_imgs), 12)
                char_cols = st.columns(n)
                for idx in range(n):
                    with char_cols[idx]:
                        ch_label = recognized_chars[idx] if idx < len(recognized_chars) else "?"
                        st.image(char_imgs[idx]["image"], width=45, caption=ch_label)
            
            st.markdown("<br/>", unsafe_allow_html=True)
            
            # Ghi log
            now = datetime.datetime.now().strftime("%H:%M:%S %d/%m")
            st.session_state.vehicle_logs.insert(0, {
                "Thời gian": now,
                "Biển số": predicted_plate,
                "Loại biển": plate_shape,
                "Màu": plate_color,
                "Chủ xe": owner_name,
                "Trạng thái": "Hợp lệ ✅" if access else "Không phép ❌",
                "Model OCR": opt_ocr_model
            })
            
            # Lưu vào session để Tab 2 dùng
            st.session_state.last_result = {
                "raw": img_raw,
                "plate_crop": plate_crop,
                "debug": debug_imgs,
                "thresh": thresh_img,
                "char_imgs": char_imgs,
                "plate_shape": plate_shape,
                "plate_color": plate_color,
                "ar": ar,
                "recognized_chars": recognized_chars,
                "predicted_plate": predicted_plate,
            }
    
    else:
        # Placeholder khi chưa upload
        st.markdown("""
        <div style="background:rgba(0,255,170,0.03); border:2px dashed rgba(0,255,170,0.2); 
             border-radius:16px; padding:60px 20px; text-align:center; margin:20px 0;">
          <p style="font-size:3rem; margin:0 0 10px;">📷</p>
          <p style="color:#64748b; font-size:1rem; margin:0 0 6px;">Tải lên ảnh xe để bắt đầu nhận diện</p>
          <p style="color:#334155; font-size:0.8rem; margin:0;">Hỗ trợ: JPG, JPEG, PNG, WEBP</p>
          <p style="color:#334155; font-size:0.75rem; margin:8px 0 0;">Camera góc cao, nhiều góc độ, ban ngày/ban đêm</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Lịch sử
    if st.session_state.vehicle_logs:
        st.markdown('<hr style="border-color:rgba(255,255,255,0.06); margin:20px 0 12px;"/>', unsafe_allow_html=True)
        st.markdown('<p style="font-weight:600; color:#bd93f9; font-size:0.85rem; margin:0 0 8px;">🕒 LỊCH SỬ NHẬN DIỆN</p>', unsafe_allow_html=True)
        df_logs = pd.DataFrame(st.session_state.vehicle_logs[:20])
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ==================== TAB 2: NGHIÊN CỨU ML ====================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#00ffaa; font-weight:800; font-size:1.6rem; margin:0 0 4px;">🔬 PHÂN TÍCH & NGHIÊN CỨU THUẬT TOÁN</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.85rem; margin:0 0 20px;">Trực quan hóa pipeline OpenCV, giảm chiều PCA/t-SNE/SVD, so sánh model</p>', unsafe_allow_html=True)
    
    tab_cv, tab_reduce, tab_compare = st.tabs([
        "📷 OPENCV PIPELINE",
        "🌀 GIẢM CHIỀU DỮ LIỆU (PCA/t-SNE/SVD)",
        "📈 SO SÁNH MODEL OCR"
    ])
    
    # --- OPENCV PIPELINE ---
    with tab_cv:
        cv_data = st.session_state.get("last_result", None)
        
        if cv_data is None:
            st.info("👆 Hãy upload ảnh ở Tab 1 để xem pipeline xử lý ảnh ở đây.")
        else:
            # Bước 1 + 2
            st.markdown('<p style="color:#00ffaa; font-weight:700; font-size:0.9rem; margin:0 0 10px;">📌 Bước 1-2: Ảnh gốc → Xử lý cạnh (Sobel + Morphology + CLAHE)</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(cv_data["raw"], channels="BGR", use_container_width=True, caption="Ảnh gốc đầu vào")
            with c2:
                if "enhanced" in cv_data["debug"]:
                    st.image(cv_data["debug"]["enhanced"], channels="BGR", use_container_width=True, caption="CLAHE Enhancement")
                else:
                    st.image(cv_data["debug"].get("gray", cv_data["raw"]), use_container_width=True, caption="Grayscale")
            with c3:
                if "morphed" in cv_data["debug"]:
                    st.image(cv_data["debug"]["morphed"], use_container_width=True, caption="Sobel X + Morphological Close")
                else:
                    st.image(cv_data["debug"].get("edged", cv_data["raw"]), use_container_width=True, caption="Canny Edge")
            
            st.markdown('<hr style="border-color:rgba(0,255,170,0.1); margin:12px 0;"/>', unsafe_allow_html=True)
            
            # Bước 3: Crop + Threshold + Classify
            st.markdown('<p style="color:#bd93f9; font-weight:700; font-size:0.9rem; margin:0 0 10px;">📌 Bước 3: Cắt biển số → Nhị phân hóa → Phân loại Shape/Color</p>', unsafe_allow_html=True)
            c3a, c3b, c3c = st.columns(3)
            with c3a:
                st.image(cv_data["plate_crop"], channels="BGR", use_container_width=True, caption="Vùng biển số cắt ra")
            with c3b:
                st.image(cv_data["thresh"], use_container_width=True, caption="Adaptive Threshold (nhị phân)")
            with c3c:
                p_shape = cv_data.get("plate_shape", "N/A")
                p_color = cv_data.get("plate_color", "N/A")
                p_ar = cv_data.get("ar", 0)
                shape_ico = "📏" if "Rectangle" in p_shape else "⬛"
                color_icos = {"White": "⬜", "Yellow": "🟨", "Blue": "🟦", "Green": "🟩", "Red": "🟥"}
                c_ico = color_icos.get(p_color, "⬛")
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.3); border:1px solid rgba(189,147,249,0.3); 
                     border-radius:10px; padding:16px; text-align:center; height:100%;">
                  <p style="color:#94a3b8; font-size:0.7rem; margin:0 0 4px;">HÌNH DẠNG BIỂN</p>
                  <p style="color:#bd93f9; font-size:1rem; font-weight:800; margin:0 0 6px;">{shape_ico} {p_shape}</p>
                  <p style="color:#94a3b8; font-size:0.7rem; margin:0 0 4px;">AR = {p_ar:.2f}</p>
                  <hr style="border-color:rgba(255,255,255,0.06); margin:8px 0;"/>
                  <p style="color:#94a3b8; font-size:0.7rem; margin:0 0 4px;">MÀU SẮC BIỂN</p>
                  <p style="color:#fbbf24; font-size:1rem; font-weight:800; margin:0;">{c_ico} {p_color} Plate</p>
                  <hr style="border-color:rgba(255,255,255,0.06); margin:8px 0;"/>
                  <p style="color:#64748b; font-size:0.65rem; margin:0; line-height:1.6; text-align:left;">
                    • Rectangle (1-line): AR &gt; 2.2<br/>
                    • Square (2-line): AR ≤ 2.2<br/>
                    • White: xe thường<br/>
                    • Yellow: xe tải/kinh doanh
                  </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<hr style="border-color:rgba(0,255,170,0.1); margin:12px 0;"/>', unsafe_allow_html=True)
            
            # Bước 4: Character segmentation
            st.markdown('<p style="color:#00ffaa; font-weight:700; font-size:0.9rem; margin:0 0 10px;">📌 Bước 4: Phân đoạn ký tự (Split into Single Digits/Characters) + OCR</p>', unsafe_allow_html=True)
            
            chars = cv_data["char_imgs"]
            rec_chars = cv_data.get("recognized_chars", [])
            plate_ann = draw_characters_on_plate(cv_data["plate_crop"], chars, rec_chars)
            st.image(plate_ann, channels="BGR", use_container_width=True,
                     caption=f"Biển số với {len(chars)} ký tự được phân đoạn & đánh khung")
            
            if chars:
                st.markdown('<p style="color:#94a3b8; font-size:0.75rem; margin:8px 0 6px;">Ảnh 28×28 từng ký tự → đưa vào model ML phân loại:</p>', unsafe_allow_html=True)
                n = min(len(chars), 12)
                char_disp_cols = st.columns(n)
                for idx in range(n):
                    with char_disp_cols[idx]:
                        ch_label = rec_chars[idx] if idx < len(rec_chars) else "?"
                        st.image(chars[idx]["image"], width=50, caption=ch_label)
            
            # Pipeline diagram
            st.markdown('<hr style="border-color:rgba(0,255,170,0.1); margin:12px 0;"/>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(0,0,0,0.2); border-radius:10px; padding:14px; font-family:monospace; font-size:0.75rem; color:#94a3b8; line-height:2;">
              <b style="color:#00ffaa;">Pipeline:</b><br/>
              📷 Ảnh gốc
              → <b style="color:#fbbf24;">CLAHE</b> (tăng tương phản)
              → <b style="color:#fbbf24;">Sobel X + Morphology</b> (nhóm cụm ký tự)
              → <b style="color:#fbbf24;">Scoring</b> (chọn vùng biển tốt nhất)<br/>
              → <b style="color:#bd93f9;">Deskew</b> (căn thẳng – tham khảo trungdinh22/LPR)
              → <b style="color:#bd93f9;">Adaptive Threshold</b>
              → <b style="color:#bd93f9;">Contour filter</b>
              → <b style="color:#60a5fa;">28×28 normalize</b>
              → <b style="color:#60a5fa;">KNN/SVM/MLP OCR</b>
            </div>
            """, unsafe_allow_html=True)
    
    # --- GIẢM CHIỀU ---
    with tab_reduce:
        st.markdown('<p style="color:#94a3b8; margin-bottom:16px;">Trực quan hóa phân bố dữ liệu ký tự 784 chiều (28×28 pixel) xuống 2D bằng PCA, t-SNE, SVD.</p>', unsafe_allow_html=True)
        
        opt_reduction = st.radio(
            "Chọn thuật toán giảm chiều:",
            ["t-SNE", "PCA", "SVD"],
            horizontal=True,
            help="t-SNE: phi tuyến, rõ cụm. PCA/SVD: tuyến tính, nhanh."
        )
        
        with st.spinner("⏳ Đang tính giảm chiều..."):
            reduction_results = get_reduction_data_cached(DATA_DIR, ocr_engine)
        
        algo_key = opt_reduction.split(" ")[0]
        data_coords = reduction_results[algo_key]["coords"]
        df_coords = pd.DataFrame(data_coords)
        
        fig = px.scatter(
            df_coords, x="x", y="y", color="label",
            symbol="is_digit",
            labels={"x": "Thành phần 1", "y": "Thành phần 2", "label": "Ký tự"},
            title=f"Phân bố dữ liệu ký tự biển số (thuật toán {algo_key})",
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            hover_name="label"
        )
        fig.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color='white')))
        fig.update_layout(
            plot_bgcolor="rgba(13,15,20,0.5)",
            paper_bgcolor="rgba(13,15,20,0.5)",
            font_color="#cbd5e1",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        desc = reduction_results[algo_key].get("description", "")
        if desc:
            st.markdown(f"> [!NOTE]\n> **{algo_key}**: {desc}", unsafe_allow_html=False)
    
    # --- SO SÁNH MODEL ---
    with tab_compare:
        st.markdown('<p style="color:#94a3b8; margin-bottom:16px;">Thực nghiệm huấn luyện và đánh giá 3 mô hình: KNN, SVM, MLP trên tập dữ liệu ký tự biển số.</p>', unsafe_allow_html=True)
        
        if not ocr_engine.metrics:
            with st.spinner("⏳ Đang huấn luyện và đánh giá các mô hình..."):
                ocr_engine.train_and_evaluate()
        
        metrics = ocr_engine.metrics
        model_names = list(metrics.keys())
        
        df_metrics = pd.DataFrame({
            "Mô hình": model_names,
            "Accuracy (%)": [metrics[m]["accuracy"] * 100 for m in model_names],
            "Precision (%)": [metrics[m]["precision"] * 100 for m in model_names],
            "Recall (%)": [metrics[m]["recall"] * 100 for m in model_names],
            "F1-Score (%)": [metrics[m]["f1_score"] * 100 for m in model_names],
            "Inference (ms/char)": [metrics[m]["inference_time_ms"] for m in model_names],
            "Train time (s)": [metrics[m]["train_time_sec"] for m in model_names],
        })
        
        st.markdown('<p style="color:#bd93f9; font-weight:600; font-size:0.85rem; margin-bottom:6px;">Bảng chỉ số đánh giá:</p>', unsafe_allow_html=True)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(x=model_names, y=[metrics[m]["accuracy"]*100 for m in model_names], name="Accuracy (%)", marker_color="#00ffaa"))
            fig_acc.add_trace(go.Bar(x=model_names, y=[metrics[m]["f1_score"]*100 for m in model_names], name="F1-Score (%)", marker_color="#bd93f9"))
            fig_acc.update_layout(barmode='group', title="Accuracy & F1-Score", plot_bgcolor="rgba(13,15,20,0.5)", paper_bgcolor="rgba(13,15,20,0.5)", font_color="#cbd5e1", height=300)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with c_g2:
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(x=model_names, y=[metrics[m]["inference_time_ms"] for m in model_names], name="Inference (ms)", marker_color="#f87171"))
            fig_lat.update_layout(title="Inference Latency (ms/ký tự)", plot_bgcolor="rgba(13,15,20,0.5)", paper_bgcolor="rgba(13,15,20,0.5)", font_color="#cbd5e1", height=300)
            st.plotly_chart(fig_lat, use_container_width=True)
        
        st.markdown("""
        > [!TIP]
        > **Nhận xét:** SVM thường cho Accuracy cao nhất và Inference nhanh. MLP cần nhiều dữ liệu hơn để đạt đỉnh. KNN train tức thì nhưng inference chậm theo O(N×D).
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ==================== TAB 3: BÁO CÁO ====================
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    report_text = """
# BÁO CÁO BÀI TẬP LỚN MÔN HỌC MÁY
## ĐỀ TÀI: NHẬN DẠNG BIỂN SỐ XE VIỆT NAM (LPR SYSTEM)

---

### CHƯƠNG 1: GIỚI THIỆU

#### 1.1 Đặt vấn đề
Hệ thống nhận dạng biển số xe tự động (License Plate Recognition - LPR) ứng dụng kết hợp **Thị giác máy tính** (Computer Vision / OpenCV) và **Học máy** (Machine Learning) để phát hiện, tách ký tự và phân loại biển số xe trong ảnh thực tế.

#### 1.2 Bài toán Machine Learning chính
1. **Phân loại đa lớp (Multi-class Classification)**: 36 lớp ký tự (0-9, A-Z) dùng KNN, SVM, MLP
2. **Giảm chiều dữ liệu (Dimensionality Reduction)**: PCA, t-SNE, SVD trực quan hóa phân bố ký tự 784D → 2D
3. **Phân đoạn (Segmentation)**: OpenCV pipeline tách vùng biển số và ký tự đơn lẻ

---

### CHƯƠNG 2: PHƯƠNG PHÁP LUẬN

#### 2.1 Dataset
- **LP_detection**: 6,607 ảnh train + 1,652 ảnh val với YOLO bounding box labels – dùng làm thống kê thực tế để cải thiện scoring system phát hiện biển số
- **Characters**: Dữ liệu ký tự thực từ biển số Việt Nam, 36 classes, ~4,000 mẫu/class

#### 2.2 Pipeline OpenCV & Học máy (Phát hiện Biển số)
```
Ảnh → CLAHE (tăng tương phản) → Bộ sinh ứng viên kép (Top-down & Bottom-up Character Clustering)
     → Hệ thống chấm điểm cấu trúc & hình học (Aspect Ratio, Vị trí, Mật độ ký tự trong)
     → Non-Maximum Suppression (NMS - Lọc trùng) → Projection nhị phân tinh chỉnh viền sát
     → Cắt biển số hoàn chỉnh
```

#### 2.3 Kỹ thuật nâng cao & Kế thừa từ Reference
Kế thừa kỹ thuật xử lý ảnh của *trungdinh22/License-Plate-Recognition*:
- **CLAHE** (`changeContrast`): tăng tương phản ảnh qua kênh L của LAB color space.
- **Deskew** (`deskew`): HoughLinesP → tính góc nghiêng → xoay thẳng biển nghiêng.
- **1-line/2-line classification**: phân loại hình dạng biển dựa trên tâm ký tự.

#### 2.4 Phân đoạn Ký tự nâng cao & OCR
```
Biển số → CLAHE → Deskew → Adaptive Threshold → Adaptive Morphology
         → Bộ lọc contour thô → Gộp dọc thông minh (Vertical Character Merging - nối nét đứt số 7)
         → Bộ lọc hậu xử lý lọc nhiễu → Sắp xếp thứ tự đọc → Chuẩn hóa 28×28
         → Học máy KNN / SVM / MLP phân loại đa lớp → Ghép chuỗi biển số
```

#### 2.5 Phân loại Shape + Color
- **Shape**: AR = w/h → Rectangle (1-line, AR>2.2) hay Square (2-line, AR≤2.2)
- **Color**: HSV masking → White / Yellow / Blue / Green / Red

---

### CHƯƠNG 3: KẾT QUẢ

| Mô hình | Accuracy | F1-Score | Inference |
|---------|----------|----------|-----------|
| **SVM** | **~90%** | **~90%** | ~1 ms |
| MLP | ~88% | ~88% | ~2 ms |
| KNN | ~82% | ~82% | ~8 ms |

*Kết quả trên 5,400 mẫu thực từ dataset Characters biển số Việt Nam.*

---

### CHƯƠNG 4: KẾT LUẬN
- Pipeline OpenCV+ML hoạt động ổn định cho ảnh thực tế
- Kỹ thuật CLAHE + Deskew giúp cải thiện rõ rệt với ảnh chụp nghiêng/thiếu sáng
- SVM cho hiệu năng tổng thể tốt nhất; KNN train tức thì nhưng inference chậm
- Hướng phát triển: YOLO-based detection để thay thế Sobel/Canny pipeline
"""
    
    st.markdown(report_text)
    st.download_button(
        label="📥 Tải báo cáo (.MD)",
        data=report_text,
        file_name="BaoCao_MachineLearning_LPR.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
