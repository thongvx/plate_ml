import os
import pickle
import threading
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# Đặt số thread BLAS/OpenMP về 1 để tránh segfault khi scikit-learn chạy song song
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Lock để đảm bảo chỉ 1 thread tính reduction tại một thời điểm (tránh segfault TSNE)
_reduction_lock = threading.Lock()

# Danh sách các nhãn ký tự tương ứng
from ml_core.ocr_engine import CHAR_LIST


def run_dimensionality_reduction(X, y, max_samples=350, perplexity=30):
    """
    Thực hiện giảm chiều dữ liệu bằng 3 phương pháp:
    PCA, t-SNE và SVD trên tập mẫu chữ cái phẳng X (n_samples, 784).
    """
    # Khống chế số lượng mẫu tối đa để t-SNE chạy mượt, không bị treo máy
    if len(X) > max_samples:
        # Lấy mẫu đều từ mỗi class để biểu đồ đẹp
        unique_classes = np.unique(y)
        per_class = max(1, max_samples // len(unique_classes))
        sel_idx = []
        for c in unique_classes:
            idx = np.where(y == c)[0]
            chosen = np.random.choice(idx, min(per_class, len(idx)), replace=False)
            sel_idx.extend(chosen.tolist())
        sel_idx = np.array(sel_idx)
        X_subset = X[sel_idx]
        y_subset = y[sel_idx]
    else:
        X_subset = X
        y_subset = y

    results = {}

    # 1. PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_subset)
    explained_variance = float(np.sum(pca.explained_variance_ratio_) * 100)

    pca_data = []
    for i in range(len(X_subset)):
        lbl = CHAR_LIST[int(y_subset[i])]
        pca_data.append({
            "x": float(X_pca[i, 0]),
            "y": float(X_pca[i, 1]),
            "label": lbl,
            "is_digit": lbl.isdigit()
        })
    results["PCA"] = {
        "coords": pca_data,
        "explained_variance": explained_variance,
        "description": f"PCA bảo toàn phương sai lớn nhất. 2 trục chính giải thích {explained_variance:.1f}% thông tin bộ dữ liệu gốc."
    }

    # 2. SVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_svd = svd.fit_transform(X_subset)

    svd_data = []
    for i in range(len(X_subset)):
        lbl = CHAR_LIST[int(y_subset[i])]
        svd_data.append({
            "x": float(X_svd[i, 0]),
            "y": float(X_svd[i, 1]),
            "label": lbl,
            "is_digit": lbl.isdigit()
        })
    results["SVD"] = {
        "coords": svd_data,
        "description": "SVD là kỹ thuật thừa số hóa ma trận, giảm chiều tuyến tính hiệu quả với dữ liệu ảnh xám nhị phân."
    }

    # 3. t-SNE — chạy đơn luồng để tránh segfault
    try:
        actual_perp = min(perplexity, max(5, int(len(X_subset) / 5)))
        tsne = TSNE(
            n_components=2,
            perplexity=actual_perp,
            learning_rate="auto",
            init="pca",
            random_state=42,
            n_jobs=1  # Quan trọng: 1 thread duy nhất, tránh segfault BLAS
        )
        X_tsne = tsne.fit_transform(X_subset)

        tsne_data = []
        for i in range(len(X_subset)):
            lbl = CHAR_LIST[int(y_subset[i])]
            tsne_data.append({
                "x": float(X_tsne[i, 0]),
                "y": float(X_tsne[i, 1]),
                "label": lbl,
                "is_digit": lbl.isdigit()
            })
        results["t-SNE"] = {
            "coords": tsne_data,
            "description": "t-SNE giữ các điểm tương đồng gần nhau và đẩy xa điểm khác biệt, tạo cụm chữ số rất rõ ràng!"
        }
    except Exception as e:
        # Fallback nếu t-SNE crash: dùng kết quả PCA thay thế
        print(f"⚠️ t-SNE lỗi ({e}), dùng PCA thay thế")
        results["t-SNE"] = {
            "coords": pca_data,
            "description": f"t-SNE không khả dụng (lỗi: {e}). Đang hiển thị PCA thay thế."
        }

    return results


def get_reduction_data_cached(models_dir="/Users/thongvuong/plate_ml/ml_core/data", ocr_engine=None):
    """
    Đọc dữ liệu giảm chiều đã cache, nếu chưa có thì tính toán và lưu lại.
    Dùng threading.Lock để đảm bảo chỉ 1 thread tính tại một thời điểm.
    """
    cache_path = os.path.join(models_dir, "reduction_results.pkl")

    # Kiểm tra cache trước khi acquire lock
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            # Kiểm tra cache có đủ 3 key không
            if all(k in data for k in ("PCA", "SVD", "t-SNE")):
                return data
        except Exception:
            pass

    # Chỉ 1 thread được tính reduction tại một thời điểm
    with _reduction_lock:
        # Double-check sau khi acquire lock (thread khác có thể đã tính xong)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                if all(k in data for k in ("PCA", "SVD", "t-SNE")):
                    return data
            except Exception:
                pass

        # Tính toán mới
        if ocr_engine is None:
            from ml_core.ocr_engine import OCREngine
            ocr_engine = OCREngine(models_dir)

        print("🔄 Đang tính giảm chiều lần đầu, vui lòng chờ...")
        X, y = ocr_engine.load_real_character_dataset(samples_per_char=15)
        results = run_dimensionality_reduction(X, y)

        # Lưu cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(results, f)
            print("✅ Đã lưu cache giảm chiều")
        except Exception as e:
            print(f"⚠️ Không lưu được cache: {e}")

        return results
