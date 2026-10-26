#!/usr/bin/env python3
"""VisionAI web dashboard (Flask) — Track A defect classifier interface.

Run:
    python app/app.py                      # http://127.0.0.1:5000
    VISIONAI_CHECKPOINT=checkpoints/neu_resnet18_best.pt python app/app.py

Pages:
  /          upload a product image, see prediction + Grad-CAM heatmap
  /metrics   test-set metrics (accuracy, precision/recall/F1, AUROC, confusion matrix)
  /about     datasets, methods, citations, score-vs-probability note
  POST /api/predict  JSON API (multipart field "image")

Every prediction comes from a real forward pass of the trained checkpoint.
If no checkpoint exists yet, the app starts anyway and explains how to train.
"""
import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.infer import CALIBRATION_NOTE, DefectPredictor  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
ALLOWED_EXTS = {"png", "jpg", "jpeg", "bmp", "webp"}

CHECKPOINT = Path(os.environ.get("VISIONAI_CHECKPOINT",
                                 REPO / "checkpoints" / "neu_compactcnn_best.pt"))
RESULTS_JSON = REPO / "results" / "neu_results.json"
RESULTS_DIR = REPO / "results"

_predictor = None
_predictor_error = None


def get_predictor():
    """Lazily load the model; never crash the app if training hasn't run."""
    global _predictor, _predictor_error
    if _predictor is None and _predictor_error is None:
        try:
            _predictor = DefectPredictor(CHECKPOINT)
        except Exception as e:  # noqa: BLE001
            _predictor_error = str(e)
    return _predictor


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def valid_image(file) -> tuple[bool, str]:
    if file is None or not file.filename:
        return False, "No file selected. Please choose an image."
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTS:
        return False, (f"Unsupported file type '.{ext}'. "
                       f"Allowed: {', '.join(sorted(ALLOWED_EXTS))}.")
    return True, ""


@app.route("/")
def index():
    predictor = get_predictor()
    return render_template("index.html",
                           model_ready=predictor is not None,
                           model_error=_predictor_error,
                           checkpoint=str(CHECKPOINT),
                           classes=predictor.class_names if predictor else [])


@app.route("/predict", methods=["POST"])
def predict():
    predictor = get_predictor()
    if predictor is None:
        return render_template("result.html", error=(
            "No trained model found. Train one first:<br>"
            "<code>python src/train_classifier.py --config configs/neu_compactcnn.yaml</code><br>"
            f"Details: {_predictor_error}")), 503
    file = request.files.get("image")
    ok, msg = valid_image(file)
    if not ok:
        return render_template("result.html", error=msg), 400
    data = file.read()
    if not data:
        return render_template("result.html", error="Empty upload."), 400
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                suffix=Path(secure_filename(file.filename)).suffix,
                delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        # Validate it's a real image before inference.
        with Image.open(tmp_path) as im:
            im.verify()
        pred, overlay, _ = predictor.predict_with_heatmap(tmp_path)
        preview = Image.open(io.BytesIO(data)).convert("RGB")
        preview.thumbnail((512, 512))
        preview_uri = pil_to_data_uri(preview)
    except Exception as e:  # noqa: BLE001
        return render_template("result.html",
                               error=f"Could not process that file: {e}"), 400
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
    return render_template(
        "result.html",
        error=None,
        predicted=pred["predicted_class"],
        scores=pred["scores"],
        top3=pred["top3"],
        overlay_uri=pil_to_data_uri(overlay),
        preview_uri=preview_uri,
        calibration_note=pred["calibration_note"],
    )


@app.route("/metrics")
def metrics():
    if not RESULTS_JSON.exists():
        return render_template("metrics.html", results=None)
    with open(RESULTS_JSON, encoding="utf-8") as f:
        results = json.load(f)
    cm_file = RESULTS_JSON.with_name(RESULTS_JSON.stem + "_confusion_matrix.png").name
    cm_exists = (RESULTS_DIR / cm_file).exists()
    return render_template("metrics.html", results=results,
                           cm_file=cm_file if cm_exists else None)


@app.route("/results/<path:fname>")
def result_file(fname):
    return send_from_directory(RESULTS_DIR, fname)


@app.route("/about")
def about():
    return render_template("about.html", calibration_note=CALIBRATION_NOTE)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    predictor = get_predictor()
    if predictor is None:
        return jsonify({"error": "no trained model", "detail": _predictor_error}), 503
    file = request.files.get("image")
    ok, msg = valid_image(file)
    if not ok:
        return jsonify({"error": msg}), 400
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=Path(secure_filename(file.filename)).suffix, delete=False)
        file.save(tmp.name)
        tmp.close()
        pred = predictor.predict(tmp.name)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not process file: {e}"}), 400
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    return jsonify(pred)


@app.errorhandler(413)
def too_large(_):
    return render_template("result.html",
                           error="File too large (limit 8 MB). Please upload a smaller image."), 413


if __name__ == "__main__":
    print(f"Checkpoint: {CHECKPOINT} "
          f"({'found' if CHECKPOINT.exists() else 'NOT FOUND - train first'})")
    app.run(host="127.0.0.1", port=5000, debug=False)
