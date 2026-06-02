import streamlit as st

def get_custom_css():
    """
    CSS tùy chỉnh phong cách Glassmorphism và Neon cao cấp dành cho Streamlit
    """
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Thiết lập font và màu nền tổng thể */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d0f14;
        color: #e2e8f0;
    }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(18, 22, 33, 0.85) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Thiết kế thẻ Card Glassmorphic */
    .glass-card {
        background: rgba(22, 28, 45, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 255, 170, 0.3);
        transform: translateY(-2px);
    }
    
    /* Neon Texts & Badges */
    .neon-text-teal {
        color: #00ffaa;
        text-shadow: 0 0 10px rgba(0, 255, 170, 0.5);
        font-weight: 800;
    }
    
    .neon-text-purple {
        color: #bd93f9;
        text-shadow: 0 0 10px rgba(189, 147, 249, 0.5);
        font-weight: 800;
    }
    
    .neon-text-red {
        color: #ff5555;
        text-shadow: 0 0 10px rgba(255, 85, 85, 0.5);
        font-weight: 800;
    }
    
    .badge-allowed {
        background: rgba(0, 255, 170, 0.15);
        color: #00ffaa;
        border: 1px solid rgba(0, 255, 170, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        text-shadow: 0 0 5px rgba(0, 255, 170, 0.3);
    }
    
    .badge-blocked {
        background: rgba(255, 85, 85, 0.15);
        color: #ff5555;
        border: 1px solid rgba(255, 85, 85, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        text-shadow: 0 0 5px rgba(255, 85, 85, 0.3);
    }

    .badge-unknown {
        background: rgba(255, 184, 108, 0.15);
        color: #ffb86c;
        border: 1px solid rgba(255, 184, 108, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        text-shadow: 0 0 5px rgba(255, 184, 108, 0.3);
    }
    
    /* Thiết kế cho biển số xe */
    .plate-render-box {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Consolas', monospace;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        border: 3px solid #334155;
        border-radius: 8px;
        padding: 10px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5), inset 0 0 8px rgba(0,0,0,0.1);
        display: inline-block;
        letter-spacing: 2px;
        margin: 10px 0;
    }
    
    /* Thiết kế tiêu đề hoành tráng */
    .app-header {
        text-align: center;
        padding: 30px 0;
        margin-bottom: 20px;
        background: radial-gradient(circle, rgba(0,255,170,0.08) 0%, rgba(13,15,20,0) 70%);
    }
    .app-header h1 {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
        margin-bottom: 0px !important;
        background: linear-gradient(135deg, #00ffaa 0%, #bd93f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Hiệu ứng quét dòng quét camera (Scanner line animation) */
    @keyframes scanner {
        0% { top: 0%; opacity: 0.8; }
        50% { top: 100%; opacity: 0.8; }
        100% { top: 0%; opacity: 0.8; }
    }
    .camera-container {
        position: relative;
        overflow: hidden;
        border-radius: 12px;
        border: 2px solid rgba(0, 255, 170, 0.2);
        box-shadow: 0 0 20px rgba(0, 255, 170, 0.1);
    }
    .camera-scanner-line {
        position: absolute;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, rgba(0,255,170,0) 0%, rgba(0,255,170,0.8) 50%, rgba(0,255,170,0) 100%);
        box-shadow: 0 0 8px #00ffaa;
        animation: scanner 4s infinite linear;
        pointer-events: none;
        z-index: 10;
    }
    
    /* Nút bấm nâng cấp */
    .stButton > button {
        background: linear-gradient(135deg, rgba(22, 28, 45, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%) !important;
        color: #00ffaa !important;
        border: 1px solid rgba(0, 255, 170, 0.4) !important;
        border-radius: 8px !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 0 10px rgba(0, 255, 170, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ffaa 0%, #bd93f9 100%) !important;
        color: #0d0f14 !important;
        border-color: #00ffaa !important;
        box-shadow: 0 0 15px rgba(0, 255, 170, 0.4) !important;
        transform: scale(1.02);
    }
    
    /* Tùy chỉnh Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(22, 28, 45, 0.5);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 255, 170, 0.12) !important;
        color: #00ffaa !important;
        border: 1px solid rgba(0, 255, 170, 0.3) !important;
    }
    
    /* Chỉnh kiểu dáng cho các bảng biểu hiển thị dữ liệu */
    .dataframe {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }
    </style>
    """

def get_svg_gate(state="Closed"):
    """
    Tạo đoạn mã HTML/SVG động để mô phỏng rào chắn (Barrier Gate)
    Các trạng thái: "Closed", "Opening", "Open", "Closing"
    """
    # Góc xoay của rào chắn tương ứng với trạng thái
    # Closed: nằm ngang (0 độ)
    # Open: dựng đứng (-90 độ)
    # Opening: đang mở lên (-45 độ với hiệu ứng CSS transition)
    # Closing: đang hạ xuống (-45 độ)
    
    angle = 0
    transition_css = "transition: transform 1.5s cubic-bezier(0.25, 1, 0.5, 1);"
    led_color = "#ff3333" # Đỏ mặc định
    led_shadow = "rgba(255, 51, 51, 0.8)"
    state_text = "CỔNG ĐÓNG"
    text_color = "#ff5555"
    
    if state == "Open":
        angle = -90
        led_color = "#00ffaa"
        led_shadow = "rgba(0, 255, 170, 0.8)"
        state_text = "CỔNG MỞ"
        text_color = "#00ffaa"
    elif state == "Opening":
        angle = -90 # Để hoạt họa CSS chạy từ 0 tới -90
        led_color = "#ffcc00"
        led_shadow = "rgba(255, 204, 0, 0.8)"
        state_text = "ĐANG MỞ..."
        text_color = "#ffcc00"
    elif state == "Closing":
        angle = 0 # Để hoạt họa CSS chạy từ -90 tới 0
        led_color = "#ffcc00"
        led_shadow = "rgba(255, 204, 0, 0.8)"
        state_text = "ĐANG ĐÓNG..."
        text_color = "#ffcc00"

    svg_code = f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 15px; text-align: center; box-shadow: inset 0 0 20px rgba(0,0,0,0.3);">
        <div style="font-size: 14px; font-weight: bold; color: #94a3b8; margin-bottom: 10px; letter-spacing: 1px;">TRẠNG THÁI RÀO CHẮN</div>
        
        <svg width="100%" height="160" viewBox="0 0 350 160" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Nền bầu trời đêm tối -->
            <rect width="350" height="160" rx="8" fill="#090d16" />
            <line x1="0" y1="130" x2="350" y2="130" stroke="#1e293b" stroke-width="4" />
            
            <!-- Vạch kẻ đường vàng đen cảnh báo rào chắn -->
            <path d="M 0,130 L 350,130" stroke="#ffcc00" stroke-width="4" stroke-dasharray="15,10" />
            
            <!-- Đèn giao thông/LED chỉ thị trạng thái -->
            <circle cx="50" cy="40" r="16" fill="#1e293b" stroke="#334155" stroke-width="2" />
            <circle cx="50" cy="40" r="10" fill="{led_color}" style="box-shadow: 0 0 20px {led_shadow}; filter: drop-shadow(0px 0px 6px {led_color});" />
            
            <!-- Bốt bảo vệ / Trụ đỡ rào chắn -->
            <rect x="250" y="70" width="40" height="60" rx="4" fill="#334155" stroke="#475569" stroke-width="2" />
            <rect x="260" y="80" width="20" height="15" fill="#1e293b" />
            <!-- Khớp xoay rào chắn -->
            <circle cx="270" cy="80" r="8" fill="#94a3b8" stroke="#cbd5e1" stroke-width="2" />
            
            <!-- Thanh chắn (Barrier Arm) di chuyển xoay -->
            <!-- Tâm xoay nằm ở trụ đỡ (x=270, y=80) -->
            <g style="{transition_css} transform-origin: 270px 80px; transform: rotate({angle}deg);">
                <!-- Thanh chắn dài -->
                <rect x="50" y="76" width="220" height="8" rx="2" fill="#ff5555" />
                <!-- Các sọc trắng phản quang trên thanh chắn -->
                <rect x="70" y="76" width="25" height="8" fill="#ffffff" />
                <rect x="120" y="76" width="25" height="8" fill="#ffffff" />
                <rect x="170" y="76" width="25" height="8" fill="#ffffff" />
                <rect x="220" y="76" width="25" height="8" fill="#ffffff" />
                <!-- Đèn LED nhấp nháy đỏ trên thanh rào chắn -->
                <circle cx="60" cy="80" r="2.5" fill="#ff3333" />
                <circle cx="110" cy="80" r="2.5" fill="#ff3333" />
                <circle cx="160" cy="80" r="2.5" fill="#ff3333" />
                <circle cx="210" cy="80" r="2.5" fill="#ff3333" />
            </g>
        </svg>
        
        <div style="margin-top: 10px; font-size: 22px; font-weight: 800; color: {text_color}; text-shadow: 0 0 10px {led_shadow}; letter-spacing: 2px;">
            {state_text}
        </div>
    </div>
    """
    return svg_code

def get_tts_speech_component(plate_text):
    """
    Mã HTML + JS nhúng để gọi Web Speech API phát âm thanh tiếng Việt.
    Sẽ kích hoạt mỗi khi có biển số mới được phát hiện đi qua cổng rào.
    """
    if not plate_text:
        return ""
        
    text_speech = f"Xin mời xe biển số {plate_text} đi qua!"
    
    # Tạo component HTML ẩn chạy mã JavaScript SpeechSynthesis
    html_code = f"""
    <html>
    <head>
    <script>
    function speakText() {{
        if ('speechSynthesis' in window) {{
            // Hủy các giọng nói đang chờ phát
            window.speechSynthesis.cancel();
            
            var msg = new SpeechSynthesisUtterance();
            msg.text = "{text_speech}";
            msg.lang = "vi-VN"; // Thiết lập ngôn ngữ tiếng Việt
            msg.volume = 1.0;
            msg.rate = 0.9; // Nói chậm lại một chút cho tự nhiên và dễ nghe
            msg.pitch = 1.0;
            
            // Tìm giọng nói tiếng Việt chuẩn trong hệ thống nếu có
            var voices = window.speechSynthesis.getVoices();
            var viVoice = voices.find(function(voice) {{
                return voice.lang === 'vi-VN' || voice.lang.includes('vi');
            }});
            if (viVoice) {{
                msg.voice = viVoice;
            }}
            
            window.speechSynthesis.speak(msg);
        }}
    }}
    // Gọi phát âm thanh ngay khi load
    window.onload = function() {{
        // Web Speech API cần tương tác người dùng hoặc thỉnh thoảng cần delay ngắn để tải voices
        setTimeout(speakText, 100);
    }};
    </script>
    </head>
    <body style="background:transparent; margin:0; padding:0; overflow:hidden; width:1px; height:1px;">
    </body>
    </html>
    """
    return html_code
