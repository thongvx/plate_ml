import os
import pickle
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import cv2

# Danh sách 35 ký tự biển số xe Việt Nam có trong CNN_letter_Dataset:
# 0-9 (10 số) + A-Z trừ 'O' (25 chữ cái) = 35 lớp
# Lưu ý: CNN_letter_Dataset KHÔNG có class 'O' do dễ nhầm với số '0'
CHAR_LIST = (
    [str(i) for i in range(10)] +
    [c for c in 'ABCDEFGHIJKLMNPQRSTUVWXYZ']  # không có 'O'
)
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHAR_LIST)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_LIST)}

class OCREngine:
    def __init__(self, models_dir="/Users/thongvuong/plate_ml/ml_core/data"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {
            "KNN": None,
            "SVM": None,
            "MLP": None
        }
        self.metrics = {}
        
        # Load các mô hình đã huấn luyện nếu có sẵn
        self.load_models()
        
    def load_models(self):
        """Load các mô hình pickle từ đĩa nếu tồn tại"""
        loaded = True
        for m_name in ["KNN", "SVM", "MLP"]:
            path = os.path.join(self.models_dir, f"{m_name.lower()}_model.pkl")
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.models[m_name] = pickle.load(f)
                except Exception:
                    loaded = False
            else:
                loaded = False
                
        metrics_path = os.path.join(self.models_dir, "model_comparison.pkl")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'rb') as f:
                    self.metrics = pickle.load(f)
            except Exception:
                pass
                
        return loaded

    def save_models(self):
        """Lưu các mô hình pickle xuống đĩa"""
        for m_name, model in self.models.items():
            if model is not None:
                path = os.path.join(self.models_dir, f"{m_name.lower()}_model.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
                    
        metrics_path = os.path.join(self.models_dir, "model_comparison.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)

    def generate_synthetic_dataset(self, samples_per_char=50):
        """
        Tạo tập dữ liệu chữ cái giả lập chất lượng cao bằng cách vẽ các chữ cái 
        từ các font chữ khác nhau, kết hợp xoay, biến dạng và thêm nhiễu pixel.
        Điều này đảm bảo HS có thể chạy thử nghiệm huấn luyện ML THẬT ngay lập tức!
        """
        X = []
        y = []
        
        # Các kiểu font chữ mô phỏng trong OpenCV
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX
        ]
        
        for char_idx, char in enumerate(CHAR_LIST):
            for _ in range(samples_per_char):
                # Tạo ảnh nền đen kích thước 28x28
                img = np.zeros((28, 28), dtype=np.uint8)
                
                # Chọn font và tham số ngẫu nhiên
                font = np.random.choice(fonts)
                scale = np.random.uniform(0.65, 0.8)
                thick = np.random.randint(1, 3)
                
                # Tính toán kích thước chữ để căn giữa
                (w, h), baseline = cv2.getTextSize(char, font, scale, thick)
                x_pos = int((28 - w) / 2) + np.random.randint(-2, 3)
                y_pos = int((28 + h) / 2) + np.random.randint(-2, 3)
                
                # Trới giới hạn vị trí
                x_pos = max(1, min(x_pos, 28 - w - 1))
                y_pos = max(h, min(y_pos, 28 - baseline - 1))
                
                # Vẽ ký tự lên ảnh (màu trắng 255)
                cv2.putText(img, char, (x_pos, y_pos), font, scale, 255, thick, cv2.LINE_AA)
                
                # Áp dụng các biến đổi hình học ngẫu nhiên: xoay ảnh nhẹ (-10 đến 10 độ)
                angle = np.random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
                img = cv2.warpAffine(img, M, (28, 28))
                
                # Thêm nhiễu Gaussian nhẹ hoặc nhiễu muối tiêu
                noise = np.random.normal(0, 15, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Nhị phân hóa lại sau khi xử lý nhiễu
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                
                # Phẳng hóa ảnh từ 28x28 thành vector 784 chiều
                flat_vector = img.flatten() / 255.0  # Chuẩn hóa về [0, 1]
                
                X.append(flat_vector)
                y.append(char_idx)
                
        return np.array(X), np.array(y)

    def load_real_character_dataset(self, samples_per_char=None):
        """
        Merge 2 bộ dữ liệu:
          - CNN_letter_Dataset  : ~35,500 ảnh jpg 100x75 (nền sáng/tối hỗn hợp)
          - character_dataset   : ~1,839  ảnh png 28x28  (Kaggle - âm bản nền tối)
        Pipeline chuẩn hóa cho CẢ HAI:
          gray → resize 28x28 → AdaptiveThreshold(BINARY_INV) → border-check → flatten/normalize
        """
        X, y = [], []
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]

        cnn_dir    = os.path.join(self.models_dir, "CNN_letter_Dataset")
        kaggle_dir = os.path.join(self.models_dir, "character_dataset")

        print(f"📂 Merge: CNN_letter_Dataset + character_dataset (Kaggle)")

        def _img_to_vec(path):
            img = cv2.imread(path)
            if img is None:
                return None
            gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            binary  = cv2.adaptiveThreshold(resized, 255,
                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
            border  = np.concatenate([binary[0,:], binary[-1,:], binary[:,0], binary[:,-1]])
            if np.mean(border) > 127:
                binary = cv2.bitwise_not(binary)
            return binary.flatten() / 255.0

        for char_idx, char in enumerate(CHAR_LIST):
            vecs = []

            # --- CNN_letter_Dataset ---
            cnn_folder = os.path.join(cnn_dir, char)
            if os.path.exists(cnn_folder):
                files = [f for f in os.listdir(cnn_folder)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if samples_per_char is not None and len(files) > samples_per_char:
                    files = [files[i] for i in np.random.choice(len(files), samples_per_char, replace=False)]
                for fn in files:
                    v = _img_to_vec(os.path.join(cnn_folder, fn))
                    if v is not None:
                        vecs.append(v)
            cnn_count = len(vecs)

            # --- Kaggle character_dataset ---
            kag_folder = os.path.join(kaggle_dir, char)
            kag_count  = 0
            if os.path.exists(kag_folder):
                kfiles = [f for f in os.listdir(kag_folder)
                          if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
                for fn in kfiles:
                    v = _img_to_vec(os.path.join(kag_folder, fn))
                    if v is not None:
                        vecs.append(v)
                        kag_count += 1

            total = len(vecs)

            # --- Fallback tổng hợp nếu hoàn toàn không có data ---
            min_req  = samples_per_char if samples_per_char else 50
            synth    = max(0, min_req - total) if total == 0 else 0
            for _ in range(synth):
                img_s = np.zeros((28, 28), dtype=np.uint8)
                font  = np.random.choice(fonts)
                scale = np.random.uniform(0.68, 0.78)
                thick = np.random.randint(1, 3)
                (tw, th), bl = cv2.getTextSize(char, font, scale, thick)
                xp = max(2, min(int((28-tw)/2) + np.random.randint(-1,2), 28-tw-2))
                yp = max(th+2, min(int((28+th)/2) + np.random.randint(-1,2), 28-bl-2))
                cv2.putText(img_s, char, (xp, yp), font, scale, 255, thick, cv2.LINE_AA)
                M = cv2.getRotationMatrix2D((14,14), np.random.uniform(-8,8), 1.0)
                img_s = cv2.warpAffine(img_s, M, (28,28))
                vecs.append(img_s.flatten() / 255.0)

            for v in vecs:
                X.append(v)
                y.append(char_idx)

            kag_str   = f" +{kag_count}Kaggle" if kag_count > 0 else ""
            synth_str = f" +{synth}synth" if synth > 0 else ""
            print(f"  ✅ '{char}': {cnn_count}CNN{kag_str}{synth_str} = {len(vecs)}")

        print(f"\n📊 Tổng: {len(X)} mẫu | {len(set(y))} lớp | CNN+Kaggle merged ✅")
        return np.array(X), np.array(y)

    def train_and_evaluate(self, X=None, y=None):
        """
        Huấn luyện 3 bộ phân loại và so sánh hiệu suất.
        Mặc định load dữ liệu thực tế từ CNN_letter_Dataset.
        """
        if X is None or y is None:
            # Ưu tiên dùng dữ liệu thực tế từ CNN_letter_Dataset
            cnn_dir = os.path.join(self.models_dir, "CNN_letter_Dataset")
            if os.path.exists(cnn_dir) and len(os.listdir(cnn_dir)) > 0:
                print("📚 Load dữ liệu thực tế từ CNN_letter_Dataset...")
                X, y = self.load_real_character_dataset(samples_per_char=None)  # Lấy toàn bộ
            else:
                print("⚠️  Không tìm thấy CNN_letter_Dataset, dùng dữ liệu tổng hợp...")
                X, y = self.generate_synthetic_dataset()
            
        # Chia tập dữ liệu Train / Test là 80% / 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Định nghĩa các mô hình
        self.models["KNN"] = KNeighborsClassifier(n_neighbors=3)
        self.models["SVM"] = SVC(kernel='rbf', C=10.0, probability=True, random_state=42)
        self.models["MLP"] = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, alpha=1e-4,
                                           solver='adam', verbose=False, random_state=42)
                                           
        comparison_results = {}
        
        for m_name, model in self.models.items():
            # Đo thời gian huấn luyện
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train
            
            # Đo độ chính xác trên tập kiểm thử
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Tính Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            
            # Đo thời gian suy luận (latency) trên 100 mẫu ký tự đơn lẻ
            start_inf = time.time()
            for i in range(100):
                _ = model.predict([X_test[i % len(X_test)]])
            inf_time = ((time.time() - start_inf) / 100.0) * 1000.0  # ms/character
            
            comparison_results[m_name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "train_time_sec": float(train_time),
                "inference_time_ms": float(inf_time)
            }
            
        self.metrics = comparison_results
        self.save_models()
        return comparison_results

    def predict(self, char_img, model_type="SVM"):
        """
        Nhận diện ký tự từ ảnh đầu vào (kích thước 28x28).
        Trả về ký tự được dự đoán và mức độ tin cậy (Confidence).
        """
        # Nếu mô hình chưa được huấn luyện/load, tự huấn luyện nhanh để chạy
        if self.models[model_type] is None:
            self.train_and_evaluate()
            
        model = self.models[model_type]
        
        # Tiền xử lý ảnh ký tự: chuyển về vector phẳng và chuẩn hóa [0, 1]
        flat_char = char_img.flatten() / 255.0
        flat_char = flat_char.reshape(1, -1)
        
        # Dự đoán nhãn
        pred_idx = model.predict(flat_char)[0]
        predicted_char = INDEX_TO_CHAR[pred_idx]
        
        # Tính confidence score
        confidence = 0.95
        try:
            probs = model.predict_proba(flat_char)[0]
            confidence = float(probs[pred_idx])
        except (AttributeError, NotImplementedError):
            # Nếu mô hình không hỗ trợ predict_proba (như SVM chưa bật probability),
            # tính toán dựa trên khoảng cách (KNN) hoặc giá trị mặc định
            if model_type == "KNN":
                distances, indices = model.kneighbors(flat_char)
                # Tính tỷ lệ số láng giềng trùng với kết quả dự đoán
                neighbor_labels = model._y[indices[0]]
                matches = np.sum(neighbor_labels == pred_idx)
                confidence = float(matches / len(neighbor_labels))
                
        return predicted_char, confidence

    def predict_plate_text(self, char_imgs, model_type="SVM"):
        """
        Nhận diện chuỗi văn bản của cả biển số xe từ danh sách ảnh các ký tự.
        """
        plate_str = ""
        confidences = []
        
        for c_data in char_imgs:
            char_img = c_data["image"]
            char, conf = self.predict(char_img, model_type)
            plate_str += char
            confidences.append(conf)
            
        mean_confidence = np.mean(confidences) if confidences else 0.0
        
        # Định dạng lại biển số xe Việt Nam cho đẹp mắt
        # Ví dụ: "29A112345" -> "29-A1 123.45" hoặc "30F8888" -> "30F-8888"
        formatted_str = self.format_vietnamese_plate(plate_str)
        
        return formatted_str, mean_confidence

    @staticmethod
    def format_vietnamese_plate(plate_str):
        """
        Định dạng chuỗi ký tự liền mạch thành biển số xe chuẩn Việt Nam:
        - Xe máy/ô tô 4 số: e.g. 29H1234 -> 29H-1234
        - Xe máy/ô tô 5 số: e.g. 29A112345 -> 29-A1 123.45 hoặc 30F12345 -> 30F-123.45
        """
        # Nếu chuỗi rỗng hoặc quá ngắn, trả lại nguyên bản
        if len(plate_str) < 4:
            return plate_str
            
        # Thêm dấu gạch ngang và dấu chấm tương ứng với độ dài biển số
        # Thông thường biển Việt Nam bắt đầu bằng 2 số (Mã tỉnh) + 1 hoặc 2 chữ cái/số
        # Ví dụ: 29, 30, 59, 79...
        prov = plate_str[:2]
        rest = plate_str[2:]
        
        if len(rest) == 5: # Xe máy 5 số (e.g. 29A1 12345 -> rest = A112345)
            if rest[0].isalpha() and rest[1].isalnum():
                return f"{prov}-{rest[0]}{rest[1]} {rest[2:5]}.{rest[5:]}"
            else:
                return f"{prov}{rest[0]}-{rest[1:4]}.{rest[4:]}"
        elif len(rest) == 6: # Ô tô 5 số (e.g. 30F 12345 -> rest = F12345)
            return f"{prov}{rest[0]}-{rest[1:4]}.{rest[4:]}"
        elif len(rest) == 4: # Xe cũ 4 số (e.g. 29A 1234 -> rest = A1234)
            return f"{prov}{rest[0]}-{rest[1:]}"
        else:
            # Fallback nếu chiều dài khác biệt
            return plate_str
