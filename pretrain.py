import os
import sys

# Thêm thư mục hiện tại vào path để import được ml_core
sys.path.append("/Users/thongvuong/plate_ml")

print("Initializing LPR Machine Learning Core Pre-Trainer...")
from ml_core.ocr_engine import OCREngine
from ml_core.reducer import get_reduction_data_cached

# Khởi tạo OCREngine
DATA_DIR = "/Users/thongvuong/plate_ml/ml_core/data"
os.makedirs(DATA_DIR, exist_ok=True)
engine = OCREngine(DATA_DIR)

print("\n--- STAGE 1: Loading Real Digit & Character Dataset ---")
X, y = engine.load_real_character_dataset(samples_per_char=100)
print(f"Loaded {len(X)} samples in total (36 distinct character classes, up to 100 samples each).")
print(f"Vector dimension per sample: {X.shape[1]} (28x28 flattened pixels).")

print("\n--- STAGE 2: Training and Evaluating KNN, SVM, & MLP Classifiers ---")
metrics = engine.train_and_evaluate(X, y)
for model_name, info in metrics.items():
    print(f"Model: {model_name:3s} | Accuracy: {info['accuracy']*100:.2f}% | Precision: {info['precision']*100:.2f}% | Latency: {info['inference_time_ms']:.4f} ms/char")

print("\n--- STAGE 3: Pre-Computing PCA, t-SNE, & SVD Dimensionality Reductions ---")
# Hàm này sẽ sinh 15 mẫu mỗi ký tự (540 tổng) và lưu cache kết quả giảm chiều
results = get_reduction_data_cached(DATA_DIR, engine)
print("PCA coordinates pre-calculated.")
print("SVD coordinates pre-calculated.")
print("t-SNE coordinates pre-calculated and cached.")

print("\n--- SUCCESS: All Models and Visuals Pre-Trained! ---")
print("Saved models, comparison metrics, and clusters under 'ml_core/data/'.")
