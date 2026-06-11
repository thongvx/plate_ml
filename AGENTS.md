# AGENTS.md

## Project Snapshot
- This is a Vietnamese license-plate recognition demo app (`app.py`) built around a **hybrid pipeline**: YOLO/OpenCV for plate localization + classic CV for character segmentation + HOG+ANN OCR (`ml_core/alpr_engine.py`).
- Runtime UI is Streamlit-only (no backend service split); `app.py` orchestrates everything and keeps state in `st.session_state` (`allowed_db`, `vehicle_logs`, `last_result`).
- Most comments/docs are Vietnamese; preserve domain wording when editing heuristics.

## Architecture You Need To Understand First
- Entry point: `app.py` tab-1 pipeline is `detect_plate(...) -> classify_plate_type(...) -> segment_characters(...) -> ALPREngine.predict_plate(...)`.
- Detection boundary: `ml_core/segmentation.py` tries YOLOv5 (`LP_detector.pt`) first, then falls back to a long OpenCV candidate+scoring pipeline.
- OCR boundary: `ALPREngine` is inference-only and lightweight (numpy forward pass over `.h5` weights + scaler/label bundle), so no TensorFlow dependency at runtime.
- Training/experiments are separate scripts (`pretrain.py`, `train_yolo.py`) and are not imported by `app.py` in normal inference.

## Critical Paths and Data Locations
- Character OCR assets required by `ALPREngine`:
  - `ml_core/results/alpr_deployment_bundle.pkl`
  - `ml_core/results/alpr_model.h5`
- Gate allowlist file used live by UI: `ml_core/data/data_allowed.json`.
- Plate detection dataset expected in: `ml_core/data/LP_detection/` (`images/{train,val}`, `labels/{train,val}`).
- Reference YOLO weights loaded by `torch.hub` from: `reference_src/model/LP_detector.pt` and `reference_src/model/LP_ocr.pt`.

## Developer Workflows (Observed)
- Install dependencies:
```bash
pip install -r requirements.txt
```
- Run app:
```bash
streamlit run app.py
```
- Generate deterministic sample vehicle images for quick manual checks:
```bash
python generate_samples.py
```
- Rebuild classic OCR models + dimensionality-reduction cache:
```bash
python pretrain.py
```
- Post-training deployment helper for detector model swap/quick verify:
```bash
python deploy_model.py
```

## Project-Specific Conventions
- Many scripts use **absolute local paths** (e.g., `/Users/thongvuong/plate_ml/...` in `pretrain.py`, `deploy_model.py`, `extract_character_dataset.py`); avoid introducing more, prefer repo-relative paths when touching those files.
- `segment_characters()` must return `list[{"image": np.ndarray, "box": [...] }]` + threshold image; `ALPREngine.predict_plate()` depends on that exact shape.
- Character images are normalized to `32x32` before HOG in production path (even if some training utilities still use `28x28`).
- Plate formatting is centralized in `ALPREngine.format_plate()`; UI and logs expect formatted strings, not raw char sequences.

## Integration / Runtime Caveats
- YOLO loading in `ml_core/segmentation.py` assumes a local yolov5 hub cache path `~/.cache/torch/hub/ultralytics_yolov5_master`; if unavailable, code silently degrades to OpenCV fallback.
- `train_yolo.py` uses Ultralytics YOLOv8 (`ultralytics` package), but runtime detection uses YOLOv5 via `torch.hub`; do not conflate training/inference stacks.
- No automated test suite is present in the repo; validation is primarily manual via Streamlit + sample images.

## Safe Change Strategy For Agents
- If touching detection heuristics, inspect both `detect_plate()` and `_detect_plate_opencv()` to avoid regressions in YOLO-fallback behavior.
- If touching OCR output formatting or class set, verify `app.py` allowlist matching (`_norm`) and UI confidence cards still behave correctly.
- Prefer small, localized edits in `ml_core/segmentation.py`; it is heavily heuristic and tightly coupled to Vietnamese plate geometry assumptions.

