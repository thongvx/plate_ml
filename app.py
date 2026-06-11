"""
app.py - Hệ thống Nhận dạng Biển số Xe Việt Nam (LPR)
Group 4 | MLE501.22 | MSA36HN
Pipeline: OpenCV detect/segment → HOG (skimage) → ANN (numpy) → biển số
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
from ml_core.alpr_engine import ALPREngine
from ui_styles import get_custom_css

# ─────────────────────────── CẤU HÌNH ─────────────────────────────────────────
st.set_page_config(
    page_title="LPR System | MLE501.22 Group 4",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)

DATA_DIR     = os.path.join(os.path.dirname(__file__), "ml_core", "data")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "ml_core", "results")
LP_DETECT_DIR = os.path.join(DATA_DIR, "LP_detection")
os.makedirs(DATA_DIR, exist_ok=True)


@st.cache_resource(show_spinner="🤖 Đang tải mô hình HOG+ANN...")
def load_engine() -> ALPREngine:
    return ALPREngine()


engine = load_engine()

# ─────────────────────────── DATABASE ──────────────────────────────────────────
DATABASE_PATH = os.path.join(DATA_DIR, "data_allowed.json")


def load_allowed_db():
    if os.path.exists(DATABASE_PATH):
        try:
            with open(DATABASE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    default = [
        {"plate": "29A-123.45", "owner": "Nguyễn Văn Hùng",  "type": "Xe máy Honda SH",   "status": "Allowed"},
        {"plate": "30F-888.88", "owner": "Lê Hoài Thu",       "type": "Ô tô Mercedes E300", "status": "Allowed"},
        {"plate": "51F-178.12","owner": "Trần Minh Hoàng",   "type": "Xe máy Vespa",        "status": "Allowed"},
        {"plate": "88A-777.77", "owner": "Phạm Quốc Bảo",    "type": "Ô tô VinFast VF8",    "status": "Allowed"},
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

# ─────────────────────────── SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px;">
        <p style="font-size:2.5rem; margin:0;">🚘</p>
        <h2 style="color:#00ffaa; font-size:1.1rem; margin:6px 0 2px; font-weight:800; letter-spacing:2px;">
            LPR SYSTEM
        </h2>
        <p style="color:#64748b; font-size:0.72rem; margin:0;">Vietnamese License Plate Recognition</p>
        <p style="color:#475569; font-size:0.68rem; margin:4px 0 0;">Group 4 · MLE501.22 · MSA36HN</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="color:#94a3b8; font-size:0.78rem; font-weight:700; margin-bottom:8px;">🧠 MÔ HÌNH NHẬN DẠNG</p>', unsafe_allow_html=True)
    info = engine.model_info
    st.markdown(f"""
    <div style="background:rgba(0,255,170,0.06); border:1px solid rgba(0,255,170,0.2);
         border-radius:10px; padding:12px; font-size:0.75rem; line-height:1.8;">
        <p style="color:#00ffaa; font-weight:700; margin:0 0 6px;">✅ HOG + ANN</p>
        <p style="color:#94a3b8; margin:0;">Feature: <b style="color:#f8fafc;">HOG 324-dim</b></p>
        <p style="color:#94a3b8; margin:0;">Img size: <b style="color:#f8fafc;">32×32 px</b></p>
        <p style="color:#94a3b8; margin:0;">Classes: <b style="color:#f8fafc;">{info['n_classes']} ký tự</b></p>
        <p style="color:#94a3b8; margin:0;">ANN: <b style="color:#f8fafc;">128→64→30</b></p>
        <p style="color:#94a3b8; margin:0;">Val acc: <b style="color:#00ffaa;">~95%</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="color:#94a3b8; font-size:0.78rem; font-weight:700; margin-bottom:6px;">📁 DATASET THỐNG KÊ</p>', unsafe_allow_html=True)
    tr = len(os.listdir(os.path.join(LP_DETECT_DIR, "images", "train"))) if os.path.exists(os.path.join(LP_DETECT_DIR, "images", "train")) else 0
    vl = len(os.listdir(os.path.join(LP_DETECT_DIR, "images", "val")))   if os.path.exists(os.path.join(LP_DETECT_DIR, "images", "val"))   else 0
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
         border-radius:8px; padding:10px; font-size:0.75rem;">
        <p style="color:#94a3b8; margin:0;">🏋️ Train: <b style="color:#f8fafc;">{tr:,} ảnh</b></p>
        <p style="color:#94a3b8; margin:0;">🧪 Val: <b style="color:#f8fafc;">{vl:,} ảnh</b></p>
        <p style="color:#94a3b8; margin:0;">📐 Format: YOLO bbox</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="color:#94a3b8; font-size:0.78rem; font-weight:700; margin-bottom:6px;">📋 XE ĐƯỢC PHÉP VÀO</p>', unsafe_allow_html=True)
    for entry in st.session_state.allowed_db[:5]:
        st.markdown(f"""
        <div style="border-left:2px solid #00ffaa; padding:4px 8px; margin-bottom:5px;
             background:rgba(0,255,170,0.04); border-radius:4px;">
            <p style="color:#00ffaa; font-size:0.73rem; font-weight:700; margin:0; font-family:monospace;">{entry['plate']}</p>
            <p style="color:#64748b; font-size:0.67rem; margin:0;">{entry['owner']}</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────── MAIN TABS ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📷  NHẬN DIỆN BIỂN SỐ",
    "🔬  PHÂN TÍCH ML & HOG",
    "📋  BÁO CÁO HỌC THUẬT",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — NHẬN DIỆN BIỂN SỐ
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#00ffaa; font-weight:800; font-size:1.5rem; margin:0 0 4px;">🚘 NHẬN DIỆN BIỂN SỐ XE VIỆT NAM</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.83rem; margin:0 0 20px;">Upload ảnh xe → OpenCV phát hiện biển → Tách ký tự → HOG + ANN nhận dạng</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "📁 Tải lên ảnh xe:",
        type=["jpg", "jpeg", "png", "webp"],
        help="Hỗ trợ JPG, PNG, WEBP."
    )

    if uploaded is not None:
        raw_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

        if img_raw is None:
            st.error("❌ Không đọc được ảnh.")
        else:
            h0, w0 = img_raw.shape[:2]
            if w0 > 1200:
                img_raw = cv2.resize(img_raw, (1200, int(1200 * h0 / w0)))

            col_img, col_info = st.columns([3, 2])
            with col_img:
                st.markdown('<p style="color:#94a3b8; font-size:0.78rem; margin-bottom:4px;">📸 Ảnh đầu vào</p>', unsafe_allow_html=True)
                st.image(img_raw, channels="BGR", use_container_width=True)
            with col_info:
                h_i, w_i = img_raw.shape[:2]
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2); border-radius:8px; padding:14px; font-size:0.78rem;">
                    <p style="color:#94a3b8; margin:0 0 4px;">Kích thước: <b style="color:#f8fafc;">{w_i}×{h_i} px</b></p>
                    <p style="color:#94a3b8; margin:0 0 4px;">File: <b style="color:#f8fafc;">{uploaded.name}</b></p>
                    <p style="color:#94a3b8; margin:0;">Model OCR: <b style="color:#00ffaa;">HOG + ANN</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            # ── PIPELINE ────────────────────────────────────────────────────
            with st.spinner("⚙️ Đang xử lý pipeline…"):
                plate_crop, plate_box, debug_imgs = detect_plate(img_raw)
                plate_shape, ar, plate_color = classify_plate_type(plate_crop)
                char_imgs, thresh_img = segment_characters(plate_crop)
                formatted_plate, mean_conf, per_char = engine.predict_plate(char_imgs)

            # ── KẾT QUẢ CHÍNH ───────────────────────────────────────────────
            st.markdown('<hr style="border-color:rgba(0,255,170,0.15); margin:4px 0 16px;"/>', unsafe_allow_html=True)

            plate_label = f"{formatted_plate}  [{plate_shape} | {plate_color}]"
            img_result = draw_plate_on_car(img_raw, plate_box, plate_label)
            st.image(img_result, channels="BGR", use_container_width=True, caption="Kết quả phát hiện biển số")

            # 4 cards
            c1, c2, c3, c4 = st.columns(4)
            color_map = {"White": "#f8fafc", "Yellow": "#fbbf24", "Blue": "#60a5fa", "Green": "#34d399", "Red": "#f87171"}
            c_hex = color_map.get(plate_color, "#94a3b8")
            shape_hex = "#00ffaa" if "Rectangle" in plate_shape else "#bd93f9"

            with c1:
                st.markdown(f"""
                <div style="background:rgba(0,255,170,0.08); border:1px solid rgba(0,255,170,0.2);
                     border-radius:10px; padding:14px; text-align:center;">
                    <p style="color:#94a3b8; font-size:0.68rem; margin:0 0 4px; text-transform:uppercase;">Biển số</p>
                    <p style="color:#00ffaa; font-size:1.25rem; font-weight:800; margin:0; font-family:monospace;">{formatted_plate or '---'}</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background:rgba(189,147,249,0.08); border:1px solid rgba(189,147,249,0.2);
                     border-radius:10px; padding:14px; text-align:center;">
                    <p style="color:#94a3b8; font-size:0.68rem; margin:0 0 4px; text-transform:uppercase;">Loại biển</p>
                    <p style="color:{shape_hex}; font-size:0.9rem; font-weight:700; margin:0;">{plate_shape}</p>
                    <p style="color:#475569; font-size:0.68rem; margin:0;">AR = {ar:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                conf_pct = int(mean_conf * 100)
                conf_hex = "#00ffaa" if conf_pct >= 80 else "#fbbf24" if conf_pct >= 60 else "#f87171"
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                     border-radius:10px; padding:14px; text-align:center;">
                    <p style="color:#94a3b8; font-size:0.68rem; margin:0 0 4px; text-transform:uppercase;">Độ tin cậy</p>
                    <p style="color:{conf_hex}; font-size:1.25rem; font-weight:800; margin:0;">{conf_pct}%</p>
                    <p style="color:#475569; font-size:0.68rem; margin:0;">softmax confidence</p>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                def _norm(s): return s.replace("-","").replace(".","").replace(" ","").upper()
                access, owner_name = False, "Khách vãng lai"
                for entry in st.session_state.allowed_db:
                    if _norm(entry["plate"]) == _norm(formatted_plate):
                        access, owner_name = True, entry["owner"]; break
                badge_hex = "#00ffaa" if access else "#f87171"
                badge_txt = "✅ HỢP LỆ" if access else "❌ KHÔNG PHÉP"
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2); border:1px solid {badge_hex}40;
                     border-radius:10px; padding:14px; text-align:center;">
                    <p style="color:#94a3b8; font-size:0.68rem; margin:0 0 4px; text-transform:uppercase;">Kiểm soát cổng</p>
                    <p style="color:{badge_hex}; font-size:0.9rem; font-weight:800; margin:0;">{badge_txt}</p>
                    <p style="color:#475569; font-size:0.68rem; margin:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{owner_name}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            # ── HIỂN THỊ TỪNG KÝ TỰ ────────────────────────────────────────
            st.markdown('<p style="font-weight:700; color:#bd93f9; font-size:0.83rem; margin:0 0 8px;">🔢 KÝ TỰ ĐÃ TÁCH & NHẬN DẠNG:</p>', unsafe_allow_html=True)
            st.image(plate_crop, channels="BGR", use_container_width=True,
                     caption=f"Biển số — {len(char_imgs)} ký tự được tách")

            if char_imgs:
                n = min(len(char_imgs), 12)
                char_cols = st.columns(n)
                for idx in range(n):
                    with char_cols[idx]:
                        ch_label = per_char[idx]["char"] if idx < len(per_char) else "?"
                        conf_c   = per_char[idx]["confidence"] if idx < len(per_char) else 0.0
                        st.image(char_imgs[idx]["image"], width=48, caption=f"{ch_label}\n{int(conf_c*100)}%")

            # ── LOG ─────────────────────────────────────────────────────────
            now = datetime.datetime.now().strftime("%H:%M:%S %d/%m")
            st.session_state.vehicle_logs.insert(0, {
                "Thời gian": now,
                "Biển số": formatted_plate,
                "Loại biển": plate_shape,
                "Màu": plate_color,
                "Chủ xe": owner_name,
                "Trạng thái": "Hợp lệ ✅" if access else "Không phép ❌",
                "Confidence": f"{int(mean_conf*100)}%",
            })
            st.session_state.last_result = {
                "raw": img_raw, "plate_crop": plate_crop, "debug": debug_imgs,
                "thresh": thresh_img, "char_imgs": char_imgs,
                "plate_shape": plate_shape, "plate_color": plate_color,
                "ar": ar, "per_char": per_char, "formatted_plate": formatted_plate,
            }
    else:
        st.markdown("""
        <div style="background:rgba(0,255,170,0.03); border:2px dashed rgba(0,255,170,0.2);
             border-radius:16px; padding:60px 20px; text-align:center; margin:20px 0;">
            <p style="font-size:3rem; margin:0 0 10px;">📷</p>
            <p style="color:#64748b; font-size:1rem; margin:0 0 6px;">Tải lên ảnh xe để bắt đầu nhận diện</p>
            <p style="color:#334155; font-size:0.78rem; margin:0;">Hỗ trợ: JPG, JPEG, PNG, WEBP</p>
            <p style="color:#334155; font-size:0.73rem; margin:8px 0 0;">Pipeline: OpenCV → HOG → ANN (Group 4 model, val acc ~95%)</p>
        </div>
        """, unsafe_allow_html=True)

    # Lịch sử
    if st.session_state.vehicle_logs:
        st.markdown('<hr style="border-color:rgba(255,255,255,0.06); margin:20px 0 12px;"/>', unsafe_allow_html=True)
        st.markdown('<p style="font-weight:600; color:#bd93f9; font-size:0.83rem; margin:0 0 8px;">🕒 LỊCH SỬ NHẬN DIỆN</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.vehicle_logs[:20]), use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PHÂN TÍCH ML & HOG
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#00ffaa; font-weight:800; font-size:1.5rem; margin:0 0 4px;">🔬 PHÂN TÍCH ML — HOG + ANN PIPELINE</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.83rem; margin:0 0 20px;">Trực quan hóa quá trình xử lý: OpenCV pipeline, HOG features, kiến trúc ANN và kết quả huấn luyện.</p>', unsafe_allow_html=True)

    sub1, sub2, sub3, sub4 = st.tabs([
        "📷 OpenCV Pipeline",
        "📊 HOG Features",
        "🧠 Kiến trúc ANN",
        "📈 Kết quả Training",
    ])

    # ── SUB-TAB 1: OPENCV PIPELINE ───────────────────────────────────────────
    with sub1:
        cv_data = st.session_state.get("last_result")
        if cv_data is None:
            st.info("👆 Hãy upload ảnh ở Tab 1 để xem pipeline xử lý ảnh ở đây.")
        else:
            st.markdown('<p style="color:#00ffaa; font-weight:700; font-size:0.88rem; margin:0 0 10px;">📌 Bước 1–2: Ảnh gốc → CLAHE → Sobel + Morphology</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.image(cv_data["raw"], channels="BGR", use_container_width=True, caption="Ảnh gốc")
            with c2:
                img2 = cv_data["debug"].get("enhanced", cv_data["debug"].get("gray", cv_data["raw"]))
                if img2.ndim == 2: st.image(img2, use_container_width=True, caption="CLAHE / Grayscale")
                else: st.image(img2, channels="BGR", use_container_width=True, caption="CLAHE")
            with c3:
                img3 = cv_data["debug"].get("morphed", cv_data["debug"].get("edged", cv_data["raw"]))
                if img3.ndim == 2: st.image(img3, use_container_width=True, caption="Sobel X + Morphology Close")
                else: st.image(img3, channels="BGR", use_container_width=True, caption="Edge Map")

            st.markdown('<hr style="border-color:rgba(0,255,170,0.1); margin:14px 0;"/>', unsafe_allow_html=True)
            st.markdown('<p style="color:#bd93f9; font-weight:700; font-size:0.88rem; margin:0 0 10px;">📌 Bước 3: Cắt biển số → Adaptive Threshold → Phân loại Shape/Color</p>', unsafe_allow_html=True)
            ca, cb, cc = st.columns(3)
            with ca: st.image(cv_data["plate_crop"], channels="BGR", use_container_width=True, caption="Vùng biển số")
            with cb: st.image(cv_data["thresh"], use_container_width=True, caption="Adaptive Threshold")
            with cc:
                psh = cv_data.get("plate_shape", "N/A")
                pcl = cv_data.get("plate_color", "N/A")
                par = cv_data.get("ar", 0)
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.3); border:1px solid rgba(189,147,249,0.3);
                     border-radius:10px; padding:16px; text-align:center;">
                    <p style="color:#bd93f9; font-size:1rem; font-weight:800; margin:0 0 4px;">{psh}</p>
                    <p style="color:#94a3b8; font-size:0.72rem; margin:0;">AR = {par:.2f}</p>
                    <hr style="border-color:rgba(255,255,255,0.06); margin:8px 0;"/>
                    <p style="color:#fbbf24; font-size:0.9rem; font-weight:700; margin:0;">● {pcl} Plate</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<hr style="border-color:rgba(0,255,170,0.1); margin:14px 0;"/>', unsafe_allow_html=True)
            st.markdown('<p style="color:#60a5fa; font-weight:700; font-size:0.88rem; margin:0 0 10px;">📌 Bước 4: Phân đoạn ký tự → HOG → ANN OCR</p>', unsafe_allow_html=True)
            chars = cv_data["char_imgs"]
            per_ch = cv_data.get("per_char", [])
            rec_chars = [p["char"] for p in per_ch]
            plate_ann = draw_characters_on_plate(cv_data["plate_crop"], chars, rec_chars)
            st.image(plate_ann, channels="BGR", use_container_width=True,
                     caption=f"{len(chars)} ký tự phân đoạn & nhận dạng")
            if chars:
                n = min(len(chars), 12)
                disp_cols = st.columns(n)
                for idx in range(n):
                    with disp_cols[idx]:
                        ch_l = per_ch[idx]["char"] if idx < len(per_ch) else "?"
                        conf_l = per_ch[idx]["confidence"] if idx < len(per_ch) else 0.0
                        st.image(chars[idx]["image"], width=50, caption=f"{ch_l} {int(conf_l*100)}%")

            st.markdown("""
            <div style="background:rgba(0,0,0,0.2); border-radius:10px; padding:14px;
                 font-family:monospace; font-size:0.73rem; color:#94a3b8; line-height:2; margin-top:16px;">
                <b style="color:#00ffaa;">Pipeline đầy đủ:</b><br/>
                📷 Ảnh gốc
                → <b style="color:#fbbf24;">CLAHE</b> (tăng tương phản)
                → <b style="color:#fbbf24;">Sobel X + Morphology</b>
                → <b style="color:#fbbf24;">Candidate scoring</b><br/>
                → <b style="color:#bd93f9;">Deskew (HoughLines)</b>
                → <b style="color:#bd93f9;">Adaptive Threshold</b>
                → <b style="color:#bd93f9;">Contour filter</b><br/>
                → <b style="color:#60a5fa;">Resize 32×32</b>
                → <b style="color:#60a5fa;">HOG (324 dims)</b>
                → <b style="color:#60a5fa;">StandardScaler</b>
                → <b style="color:#60a5fa;">ANN → Softmax → Class</b>
            </div>
            """, unsafe_allow_html=True)

    # ── SUB-TAB 2: HOG FEATURES ──────────────────────────────────────────────
    with sub2:
        cv_data = st.session_state.get("last_result")
        if cv_data is None:
            st.info("👆 Hãy upload ảnh ở Tab 1 trước.")
        else:
            chars = cv_data["char_imgs"]
            per_ch = cv_data.get("per_char", [])
            if not chars:
                st.warning("Không tách được ký tự từ ảnh này.")
            else:
                st.markdown("""
                <p style="color:#94a3b8; font-size:0.82rem; margin-bottom:16px;">
                HOG (Histogram of Oriented Gradients) trích xuất gradient hướng cục bộ từ ảnh,
                tạo ra feature vector 324 chiều bền vững với thay đổi ánh sáng và biến dạng nhỏ.
                </p>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:rgba(0,255,170,0.05); border:1px solid rgba(0,255,170,0.15);
                     border-radius:8px; padding:10px 14px; font-size:0.77rem; margin-bottom:16px;">
                    <b style="color:#00ffaa;">HOG params:</b>
                    orientations=<b style="color:#f8fafc;">9</b> |
                    pixels_per_cell=<b style="color:#f8fafc;">(8,8)</b> |
                    cells_per_block=<b style="color:#f8fafc;">(2,2)</b> |
                    block_norm=<b style="color:#f8fafc;">L2-Hys</b> |
                    output=<b style="color:#00ffaa;">324 dims</b>
                </div>
                """, unsafe_allow_html=True)

                hog_data = engine.get_hog_visualizations(chars[:min(len(chars), 10)])
                n = len(hog_data)

                # Hiển thị cặp original/HOG
                orig_cols = st.columns(n)
                hog_cols  = st.columns(n)
                for idx, item in enumerate(hog_data):
                    with orig_cols[idx]:
                        st.image(item["original"], width=55,
                                 caption=f"'{item['char']}' {int(item['confidence']*100)}%")
                    with hog_cols[idx]:
                        st.image(item["hog"], width=55, caption="HOG")

                st.markdown('<p style="color:#475569; font-size:0.72rem; margin-top:10px;">Hàng trên: ảnh ký tự gốc | Hàng dưới: HOG visualization (gradient directions)</p>', unsafe_allow_html=True)

                # HOG histogram chart cho ký tự đầu tiên
                if hog_data:
                    feat_vec = engine.extract_hog(chars[0]["image"])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=feat_vec[:100], mode="lines",
                        line=dict(color="#00ffaa", width=1.5),
                        name="HOG feature (first 100 dims)"
                    ))
                    fig.update_layout(
                        title=f"HOG Feature Vector — ký tự '{hog_data[0]['char']}' (100/324 dims đầu tiên)",
                        xaxis_title="Dimension", yaxis_title="Value",
                        plot_bgcolor="rgba(13,15,20,0.5)", paper_bgcolor="rgba(13,15,20,0.5)",
                        font_color="#cbd5e1", height=250,
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── SUB-TAB 3: KIẾN TRÚC ANN ─────────────────────────────────────────────
    with sub3:
        st.markdown("""
        <p style="color:#94a3b8; font-size:0.82rem; margin-bottom:16px;">
        Mô hình ANN (Artificial Neural Network) được huấn luyện trên Kaggle bởi đồng đội,
        nhận đầu vào là HOG feature vector 324 chiều và phân loại thành 30 lớp ký tự.
        </p>
        """, unsafe_allow_html=True)

        # Architecture diagram
        layers = [
            ("INPUT", "324", "HOG features\n(StandardScaler)", "#475569", "#94a3b8"),
            ("Dense", "128", "ReLU activation", "#1e3a5f", "#60a5fa"),
            ("Dropout", "0.2", "Regularization", "#2d1b4e", "#bd93f9"),
            ("Dense", "64",  "ReLU activation", "#1e3a5f", "#60a5fa"),
            ("Dropout", "0.2", "Regularization", "#2d1b4e", "#bd93f9"),
            ("Dense", "30",  "Softmax → class prob", "#0d3320", "#00ffaa"),
        ]

        layer_cols = st.columns(len(layers))
        for i, (lname, units, desc, bg, color) in enumerate(layers):
            with layer_cols[i]:
                st.markdown(f"""
                <div style="background:{bg}; border:1px solid {color}40;
                     border-radius:10px; padding:14px 8px; text-align:center; min-height:120px;">
                    <p style="color:{color}; font-size:0.72rem; font-weight:700; margin:0 0 4px; text-transform:uppercase;">{lname}</p>
                    <p style="color:#f8fafc; font-size:1.2rem; font-weight:800; margin:0 0 4px;">{units}</p>
                    <p style="color:#94a3b8; font-size:0.65rem; margin:0; line-height:1.4;">{desc}</p>
                </div>
                {"<p style='text-align:center; color:#475569; font-size:1rem; margin:4px 0;'>→</p>" if i < len(layers)-1 else ""}
                """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Thống kê model
        mc1, mc2, mc3, mc4 = st.columns(4)
        stats = [
            ("Total params", "~42K", "#00ffaa"),
            ("Input dims", "324", "#60a5fa"),
            ("Output classes", "30", "#bd93f9"),
            ("Val Accuracy", "~95%", "#fbbf24"),
        ]
        for col, (label, val, color) in zip([mc1, mc2, mc3, mc4], stats):
            with col:
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2); border:1px solid rgba(255,255,255,0.08);
                     border-radius:8px; padding:12px; text-align:center;">
                    <p style="color:#64748b; font-size:0.68rem; margin:0 0 4px; text-transform:uppercase;">{label}</p>
                    <p style="color:{color}; font-size:1.1rem; font-weight:800; margin:0;">{val}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # 30 classes
        st.markdown('<p style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:8px;">30 lớp ký tự nhận dạng:</p>', unsafe_allow_html=True)
        classes = engine.classes
        # Nhóm: số 0-9 và chữ A-Z
        digits  = [c for c in classes if c.isdigit()]
        letters = [c for c in classes if c.isalpha()]
        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.2); border-radius:8px; padding:12px; font-family:monospace; font-size:0.82rem;">
            <span style="color:#60a5fa; font-weight:700;">Số ({len(digits)}): </span>
            <span style="color:#f8fafc;">{' '.join(digits)}</span><br/>
            <span style="color:#bd93f9; font-weight:700;">Chữ ({len(letters)}): </span>
            <span style="color:#f8fafc;">{' '.join(letters)}</span><br/>
            <span style="color:#475569; font-size:0.7rem;">
            * Không có: I, J, O, Q, R, W (dễ nhầm lẫn trên biển số VN)
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Training config
        st.markdown('<p style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:8px;">⚙️ Cấu hình huấn luyện:</p>', unsafe_allow_html=True)
        config_df = pd.DataFrame([
            {"Tham số": "Optimizer", "Giá trị": "Adam (lr=0.001)"},
            {"Tham số": "Loss", "Giá trị": "Categorical Cross-Entropy"},
            {"Tham số": "Metrics", "Giá trị": "Accuracy, F1"},
            {"Tham số": "Epochs", "Giá trị": "15"},
            {"Tham số": "Batch size", "Giá trị": "Auto"},
            {"Tham số": "Dropout", "Giá trị": "0.2 (2 lớp)"},
            {"Tham số": "Platform", "Giá trị": "Kaggle GPU"},
        ])
        st.dataframe(config_df, use_container_width=True, hide_index=True)

    # ── SUB-TAB 4: KẾT QUẢ TRAINING ─────────────────────────────────────────
    with sub4:
        st.markdown('<p style="color:#94a3b8; font-size:0.82rem; margin-bottom:16px;">Kết quả huấn luyện từ notebook Kaggle của đồng đội (Group 4 · MLE501.22 · MSA36HN).</p>', unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            chart_train = os.path.join(RESULTS_DIR, "chart_training_history.png")
            if os.path.exists(chart_train):
                st.image(chart_train, use_container_width=True,
                         caption="Training & Validation Accuracy/Loss qua 15 epochs")
        with r2:
            chart_dist = os.path.join(RESULTS_DIR, "chart_class_distribution.png")
            if os.path.exists(chart_dist):
                st.image(chart_dist, use_container_width=True,
                         caption="Phân bố mẫu dữ liệu theo lớp ký tự")

        st.markdown('<hr style="border-color:rgba(255,255,255,0.06); margin:16px 0;"/>', unsafe_allow_html=True)

        chart_conf = os.path.join(RESULTS_DIR, "chart_char_predictions.png")
        if os.path.exists(chart_conf):
            st.image(chart_conf, use_container_width=True,
                     caption="Kết quả dự đoán mẫu: xanh = đúng, đỏ = sai")

        chart_seq = os.path.join(RESULTS_DIR, "chart_sequence_test.png")
        if os.path.exists(chart_seq):
            st.image(chart_seq, use_container_width=True,
                     caption="Kiểm tra chuỗi ký tự: Direct ANN nhận dạng đúng hoàn toàn")

        st.markdown('<hr style="border-color:rgba(255,255,255,0.06); margin:16px 0;"/>', unsafe_allow_html=True)

        st.markdown('<p style="color:#bd93f9; font-weight:700; font-size:0.85rem; margin-bottom:10px;">📊 Kiểm tra trên biển số thực tế (Direct ANN vs Proxy ANN):</p>', unsafe_allow_html=True)
        r3, r4 = st.columns(2)
        with r3:
            chart_acc = os.path.join(RESULTS_DIR, "chart_plate_accuracy.png")
            if os.path.exists(chart_acc):
                st.image(chart_acc, use_container_width=True,
                         caption="Per-Plate Character Accuracy: Direct ANN trung bình 80.8%")
        with r4:
            chart_t1 = os.path.join(RESULTS_DIR, "chart_test_result_1.png")
            if os.path.exists(chart_t1):
                st.image(chart_t1, use_container_width=True,
                         caption="Ví dụ: biển 29A00003 — Direct ANN nhận dạng 100%")

        chart_pca = os.path.join(RESULTS_DIR, "chart_pca_tsne.png")
        if os.path.exists(chart_pca):
            st.image(chart_pca, use_container_width=True,
                     caption="PCA & t-SNE visualization của HOG feature space — 30 lớp tách biệt rõ ràng")

        # Bảng tóm tắt
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:8px;">📋 Tóm tắt kết quả:</p>', unsafe_allow_html=True)
        summary_df = pd.DataFrame([
            {"Chỉ số": "Validation Accuracy", "Giá trị": "~95%", "Ghi chú": "Trên tập val 20%"},
            {"Chỉ số": "Direct ANN (per-plate)", "Giá trị": "80.8% avg", "Ghi chú": "Kiểm tra trên 4 biển thực"},
            {"Chỉ số": "Proxy ANN (per-plate)", "Giá trị": "61.6% avg", "Ghi chú": "Dùng template thay thế"},
            {"Chỉ số": "Training time/epoch", "Giá trị": "~0.28s", "Ghi chú": "Kaggle GPU"},
            {"Chỉ số": "Model size", "Giá trị": "656 KB", "Ghi chú": "alpr_model.h5"},
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — BÁO CÁO HỌC THUẬT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    report_text = """# BÁO CÁO BÀI TẬP LỚN MÔN HỌC MÁY
## ĐỀ TÀI: NHẬN DẠNG BIỂN SỐ XE VIỆT NAM — HOG + ANN PIPELINE
**Nhóm 4 · MLE501.22 · MSA36HN**

---

### CHƯƠNG 1: GIỚI THIỆU

#### 1.1 Đặt vấn đề
Hệ thống nhận dạng biển số xe tự động (License Plate Recognition - LPR) kết hợp
**Thị giác máy tính** (Computer Vision / OpenCV) và **Học máy** (Machine Learning)
để phát hiện, tách ký tự và phân loại biển số xe trong ảnh thực tế.

#### 1.2 Các bài toán Machine Learning trong hệ thống
1. **Feature Extraction (HOG)**: Trích xuất gradient hướng cục bộ từ ảnh ký tự 32×32 → vector 324 chiều
2. **Phân loại đa lớp (ANN)**: 30 lớp ký tự (0-9, A-Z bỏ I J O Q R W) bằng mạng Dense 128→64→30
3. **Giảm chiều trực quan hóa**: PCA, t-SNE trên HOG feature space xác nhận sự phân tách tốt của các lớp
4. **Phân đoạn (Segmentation)**: OpenCV pipeline phát hiện và tách vùng biển số + ký tự đơn lẻ

---

### CHƯƠNG 2: PHƯƠNG PHÁP LUẬN

#### 2.1 Dataset
- **LP_detection**: 6,607 ảnh train + 1,652 ảnh val với YOLO bounding box labels
- **Characters**: ~1,715 ảnh ký tự thực tế từ biển số Việt Nam, 30 classes

#### 2.2 HOG Feature Extraction
```
Ảnh ký tự (bất kỳ kích thước)
  → Grayscale + resize 32×32
  → HOG (orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), L2-Hys)
  → Vector 324 chiều
  → StandardScaler normalize
  → Đưa vào ANN
```

#### 2.3 Kiến trúc ANN
```
Input(324) → Dense(128, ReLU) → Dropout(0.2)
           → Dense(64, ReLU)  → Dropout(0.2)
           → Dense(30, Softmax) → class probabilities
```
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Cross-Entropy
- Epochs: 15 | Platform: Kaggle GPU

#### 2.4 OpenCV Detection Pipeline
```
Ảnh → CLAHE → Sobel X + Morphology → Candidate scoring → Deskew (HoughLines)
     → Adaptive Threshold → Contour filter → Resize 32×32 → HOG → ANN
```

---

### CHƯƠNG 3: KẾT QUẢ

| Chỉ số | Giá trị |
|--------|---------|
| **Validation Accuracy** | **~95%** |
| Direct ANN per-plate (avg) | 80.8% |
| Proxy ANN per-plate (avg) | 61.6% |
| Model size | 656 KB |
| Training time/epoch | ~0.28s |

**Direct ANN** (dùng ảnh thực) vượt trội hơn **Proxy ANN** (dùng template) ~19% trên 4 biển test.

---

### CHƯƠNG 4: KẾT LUẬN
- HOG + ANN cho accuracy ~95% trên validation set, vượt KNN/SVM pixel-based cũ
- Pipeline OpenCV+HOG+ANN nhẹ (~656KB), không phụ thuộc GPU lúc inference
- Hướng phát triển: YOLO-based detection, tăng dataset, thử CNN thay Dense layers
"""

    st.markdown(report_text)
    st.download_button(
        label="📥 Tải báo cáo (.MD)",
        data=report_text,
        file_name="BaoCao_Group4_MLE501_HOG_ANN.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
