"""
alpr_engine.py - Engine nhận dạng ký tự biển số dùng HOG + ANN
Pipeline của nhóm (Group4_MSA36HN - MLE501.22):
  ảnh ký tự → resize 32×32 → HOG (324 dims) → StandardScaler → ANN (Dense 128→64→30) → class

Không cần tensorflow – dùng h5py đọc weights, numpy làm forward pass.
"""
import os
import pickle
import numpy as np
import cv2
from skimage.feature import hog


# ─── Hằng số đường dẫn ────────────────────────────────────────────────────────
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_BUNDLE_PATH = os.path.join(_RESULTS_DIR, "alpr_deployment_bundle.pkl")
_MODEL_PATH  = os.path.join(_RESULTS_DIR, "alpr_model.h5")


# ─── Utility: numpy ANN ────────────────────────────────────────────────────────
def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _forward(x: np.ndarray, weights: list[tuple]) -> np.ndarray:
    """
    Forward pass thuần numpy qua Sequential Dense network.
    weights: [(W0,b0,'relu'), (W1,b1,'relu'), (W2,b2,'softmax')]
    """
    for W, b, act in weights:
        x = x @ W + b
        if act == "relu":
            x = _relu(x)
        elif act == "softmax":
            x = _softmax(x)
    return x


# ─── Engine chính ─────────────────────────────────────────────────────────────
class ALPREngine:
    """
    Engine nhận dạng ký tự biển số xe Việt Nam.
    Tải model đã train từ đồng đội (HOG + ANN).
    """

    def __init__(self):
        self.scaler: object      = None
        self.lb: object          = None
        self.hog_params: dict    = {}
        self.img_size: tuple     = (32, 32)
        self.classes: list[str]  = []
        self.weights: list       = []
        self._loaded: bool       = False

        self._load()

    # ── Load ─────────────────────────────────────────────────────────────────
    def _load(self):
        """Load bundle (pkl) và weights ANN (h5)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with open(_BUNDLE_PATH, "rb") as f:
                    bundle = pickle.load(f)
                self.scaler     = bundle["scaler"]
                self.lb         = bundle["lb"]
                self.hog_params = bundle["hog_params"]
                self.img_size   = bundle["img_size"]
                self.classes    = bundle["classes"]
            except Exception as e:
                raise RuntimeError(f"Không tải được deployment bundle: {e}")

        try:
            import h5py
            with h5py.File(_MODEL_PATH, "r") as f:
                def _get(layer_name):
                    g = f[f"model_weights/{layer_name}/sequential/{layer_name}"]
                    return g["kernel"][:], g["bias"][:]

                w0, b0 = _get("dense")
                w1, b1 = _get("dense_1")
                w2, b2 = _get("dense_2")

            self.weights = [
                (w0, b0, "relu"),
                (w1, b1, "relu"),
                (w2, b2, "softmax"),
            ]
        except Exception as e:
            raise RuntimeError(f"Không tải được model weights: {e}")

        self._loaded = True

    # ── HOG extraction ───────────────────────────────────────────────────────
    def extract_hog(self, char_img: np.ndarray, visualize: bool = False):
        """
        Trích xuất HOG feature vector từ ảnh ký tự.
        char_img: grayscale hoặc BGR numpy array bất kỳ kích thước.
        Trả về (feature_vec, hog_image) nếu visualize=True, hoặc chỉ feature_vec.
        """
        # Chuyển sang grayscale nếu cần
        if char_img.ndim == 3 and char_img.shape[2] == 3:
            gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        elif char_img.ndim == 3 and char_img.shape[2] == 1:
            gray = char_img[:, :, 0]
        else:
            gray = char_img.copy()

        # Resize về 32×32
        resized = cv2.resize(gray, self.img_size, interpolation=cv2.INTER_AREA)

        # Normalize pixel về [0, 1]
        resized = resized.astype(np.float32) / 255.0

        # HOG
        if visualize:
            feat, hog_img = hog(
                resized,
                orientations=self.hog_params["orientations"],
                pixels_per_cell=self.hog_params["pixels_per_cell"],
                cells_per_block=self.hog_params["cells_per_block"],
                block_norm=self.hog_params["block_norm"],
                visualize=True,
                feature_vector=True,
            )
            # Rescale HOG image để dễ nhìn
            hog_img = (hog_img * 255).clip(0, 255).astype(np.uint8)
            return feat, hog_img
        else:
            feat = hog(
                resized,
                orientations=self.hog_params["orientations"],
                pixels_per_cell=self.hog_params["pixels_per_cell"],
                cells_per_block=self.hog_params["cells_per_block"],
                block_norm=self.hog_params["block_norm"],
                visualize=False,
                feature_vector=True,
            )
            return feat

    # ── Predict single character ──────────────────────────────────────────────
    def predict(self, char_img: np.ndarray) -> tuple[str, float]:
        """
        Nhận dạng một ký tự từ ảnh.
        Trả về (char, confidence) với confidence ∈ [0,1].
        """
        feat = self.extract_hog(char_img)
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))          # (1, 324)
        probs = _forward(feat_scaled, self.weights)[0]                     # (30,)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        char = self.classes[pred_idx]
        return char, confidence

    # ── Predict full plate ────────────────────────────────────────────────────
    def predict_plate(self, char_imgs: list[dict]) -> tuple[str, float, list]:
        """
        Nhận dạng cả biển số từ danh sách ảnh ký tự.
        char_imgs: [{"image": np.ndarray}, ...]
        Trả về (plate_string, mean_confidence, per_char_results)
        per_char_results: [{"char": str, "confidence": float}, ...]
        """
        results = []
        for item in char_imgs:
            ch, conf = self.predict(item["image"])
            results.append({"char": ch, "confidence": conf})

        raw_str = "".join(r["char"] for r in results)
        mean_conf = float(np.mean([r["confidence"] for r in results])) if results else 0.0
        formatted = self.format_plate(raw_str)
        return formatted, mean_conf, results

    # ── Format biển số VN ────────────────────────────────────────────────────
    @staticmethod
    def format_plate(plate_str: str) -> str:
        """
        Định dạng chuỗi ký tự thô thành biển số xe Việt Nam chuẩn.
        Ví dụ: '29A12345' → '29A-123.45'
        """
        s = plate_str.strip().upper()
        if len(s) < 4:
            return s

        prov = s[:2]
        rest = s[2:]

        if len(rest) == 5:  # xe máy: 29A1-1234
            if rest[0].isalpha() and len(rest) >= 2:
                return f"{prov}-{rest[0]}{rest[1]} {rest[2:5]}.{rest[5:]}" if len(rest) > 5 else f"{prov}{rest[0]}-{rest[1:]}"
            return f"{prov}{rest[0]}-{rest[1:4]}.{rest[4:]}"
        elif len(rest) == 6:  # ô tô 5 số: 30F-123.45
            return f"{prov}{rest[0]}-{rest[1:4]}.{rest[4:]}"
        elif len(rest) == 4:  # 4 số: 30F-1234
            return f"{prov}{rest[0]}-{rest[1:]}"
        else:
            return s

    # ── HOG visualization batch ───────────────────────────────────────────────
    def get_hog_visualizations(self, char_imgs: list[dict]) -> list[dict]:
        """
        Tạo HOG visualization cho từng ký tự.
        Trả về list [{"original": ndarray, "hog": ndarray, "char": str, "confidence": float}]
        """
        output = []
        for item in char_imgs:
            img = item["image"]
            feat, hog_img = self.extract_hog(img, visualize=True)
            # Scale lên để dễ nhìn
            hog_big = cv2.resize(hog_img, (64, 64), interpolation=cv2.INTER_NEAREST)
            ch, conf = self.predict(img)
            output.append({
                "original": img,
                "hog": hog_big,
                "char": ch,
                "confidence": conf,
            })
        return output

    # ── Thông tin model ───────────────────────────────────────────────────────
    @property
    def model_info(self) -> dict:
        return {
            "name": "HOG + ANN (Group4_MSA36HN)",
            "architecture": "Dense(324→128) → Dropout(0.2) → Dense(128→64) → Dropout(0.2) → Dense(64→30,softmax)",
            "n_classes": len(self.classes),
            "classes": self.classes,
            "hog_params": self.hog_params,
            "img_size": self.img_size,
            "feature_dims": 324,
        }
