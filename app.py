# app.py
# ============================================================
# GTSRB Predictor GUI (Custom CNN vs MobileNetV2)
# ------------------------------------------------------------
# Fixes:
# - Result text visibility issue (guaranteed high contrast)
#   by using QTextEdit + explicit palette + inline style.
# - Clean layout and responsive preview scaling.
# ============================================================

import os
import sys
import numpy as np

# If you face GPU crashes or want stable CPU-only inference:
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QMessageBox, QFrame, QLineEdit,
    QSizePolicy, QSpacerItem, QTextEdit
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette
from PyQt6.QtCore import Qt
from PIL import Image


# -------------------------
# GTSRB Class Names (43)
# -------------------------
GTSRB_CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles > 3.5 metric tons", "Right-of-way at the next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles > 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles > 3.5 metric tons"
]


# -------------------------
# Utility: PIL -> QPixmap
# -------------------------
def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """Convert PIL RGB image to QPixmap."""
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img, dtype=np.uint8)
    qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def get_model_input_hw(model: keras.Model):
    """
    Extract expected (H, W) from model.input_shape.
    Handles models with multi-input (uses first input).
    """
    in_shape = model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]

    if not (isinstance(in_shape, tuple) and len(in_shape) == 4):
        raise ValueError(f"Unexpected model.input_shape: {model.input_shape}")

    h, w = int(in_shape[1]), int(in_shape[2])
    return h, w


def preprocess_for_model(img_path: str, model: keras.Model) -> np.ndarray:
    """
    Load image -> RGB -> resize to model input -> normalize [0,1] -> add batch dim.
    NOTE: Your MobileNetV2 model already contains preprocess_input in the model graph.
    """
    pil_img = Image.open(img_path).convert("RGB")
    h, w = get_model_input_hw(model)
    pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)

    x = np.asarray(pil_img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def topk_predictions(img_path: str, model: keras.Model, k=3):
    """Return top-k predictions as [(cls_id, cls_name, conf), ...]."""
    x = preprocess_for_model(img_path, model)
    probs = model.predict(x, verbose=0)[0]
    idx = np.argsort(probs)[::-1][:k]

    out = []
    for i in idx:
        cls = int(i)
        conf = float(probs[i])
        name = GTSRB_CLASS_NAMES[cls] if 0 <= cls < len(GTSRB_CLASS_NAMES) else f"Class {cls}"
        out.append((cls, name, conf))
    return out


# -------------------------
# Drop Zone Frame
# -------------------------
class DropFrame(QFrame):
    """Dashed drop target that accepts image files."""
    def __init__(self, on_drop_callback):
        super().__init__()
        self.on_drop_callback = on_drop_callback
        self.setAcceptDrops(True)
        self.setObjectName("DropFrame")

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hint = QLabel("Drag & Drop image here\nor click Browse")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setObjectName("DropHint")
        lay.addWidget(hint)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = url.toLocalFile().lower()
                if p.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                self.on_drop_callback(path)
                event.acceptProposedAction()
                return
        event.ignore()


# -------------------------
# Main Window
# -------------------------
class TrafficSignPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GTSRB Predictor (CNN vs MobileNetV2)")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_paths = {
            "Custom CNN": os.path.join(base_dir, "custom_cnn_gtsrb.keras"),
            "MobileNetV2 (pre-trained)": os.path.join(base_dir, "mobilenetv2_gtsrb_best.keras"),
        }

        self.models_cache = {}

        self.current_image_path = None
        self.current_pixmap_raw = None

        self._build_ui()
        self._apply_styles()

        # Start maximized (with taskbar)
        self.showMaximized()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # --- Top bar
        top = QFrame()
        top.setObjectName("TopBar")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(12, 10, 12, 10)
        top_lay.setSpacing(10)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.model_paths.keys()))
        self.model_combo.currentTextChanged.connect(self.on_model_change)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)

        self.btn_pick_model = QPushButton("Set Model File…")
        self.btn_pick_model.clicked.connect(self.pick_model_file)

        top_lay.addWidget(QLabel("Model:"))
        top_lay.addWidget(self.model_combo, 2)
        top_lay.addWidget(QLabel("Path:"))
        top_lay.addWidget(self.model_path_edit, 6)
        top_lay.addWidget(self.btn_pick_model, 2)
        root.addWidget(top)

        # --- Two cards row
        content = QHBoxLayout()
        content.setSpacing(12)

        # Left: upload
        left = QFrame()
        left.setObjectName("Card")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(14, 14, 14, 14)
        left_lay.setSpacing(10)

        t1 = QLabel("Upload Image")
        t1.setObjectName("CardTitle")
        left_lay.addWidget(t1)

        self.drop_area = DropFrame(self.load_image)
        self.drop_area.setMinimumHeight(480)
        left_lay.addWidget(self.drop_area, 1)

        row_btn = QHBoxLayout()
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self.browse_image)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_all)

        row_btn.addWidget(self.btn_browse)
        row_btn.addWidget(self.btn_clear)
        row_btn.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        left_lay.addLayout(row_btn)

        # Right: preview + predict
        right = QFrame()
        right.setObjectName("Card")
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(14, 14, 14, 14)
        right_lay.setSpacing(10)

        t2 = QLabel("Preview")
        t2.setObjectName("CardTitle")
        right_lay.addWidget(t2)

        self.preview = QLabel("No image loaded")
        self.preview.setObjectName("PreviewBox")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(480)
        right_lay.addWidget(self.preview, 1)

        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setObjectName("PrimaryBtn")
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self.run_prediction)
        right_lay.addWidget(self.btn_predict)

        content.addWidget(left, 4)
        content.addWidget(right, 7)
        root.addLayout(content, 1)

        # --- Result Panel (use QTextEdit for guaranteed visibility)
        result = QFrame()
        result.setObjectName("ResultFrame")
        result_lay = QVBoxLayout(result)
        result_lay.setContentsMargins(12, 12, 12, 12)
        result_lay.setSpacing(8)

        title = QLabel("Prediction Output")
        title.setObjectName("ResultTitle")
        result_lay.addWidget(title)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setObjectName("ResultBox")
        self.result_box.setMinimumHeight(180)

        # ✅ Force a high-contrast palette (this avoids theme overrides)
        pal = self.result_box.palette()
        pal.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))      # background
        pal.setColor(QPalette.ColorRole.Text, QColor("#0f172a"))      # text
        self.result_box.setPalette(pal)

        # ✅ Also force it via inline style (double guarantee)
        self.result_box.setStyleSheet("""
            QTextEdit#ResultBox {
                background: #ffffff;
                color: #0f172a;
                border: 2px solid #2563eb;
                border-radius: 14px;
                padding: 10px;
                font-size: 14px;
                font-weight: 700;
            }
        """)

        self.result_box.setText("Result: —")
        result_lay.addWidget(self.result_box)

        root.addWidget(result)

        # Init selection
        self.on_model_change(self.model_combo.currentText())

    def _apply_styles(self):
        """Dark theme for the main UI (result box is forced separately)."""
        self.setStyleSheet("""
            QWidget {
                background: #0b1220;
                color: #e5e7eb;
                font-size: 13px;
            }
            #TopBar {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 12px;
            }
            #Card {
                background: #0f172a;
                border: 1px solid #1f2937;
                border-radius: 16px;
            }
            #CardTitle {
                font-size: 15px;
                font-weight: 900;
                color: #f8fafc;
            }
            #DropFrame {
                background: #0b1220;
                border: 2px dashed #334155;
                border-radius: 16px;
            }
            #DropHint {
                color: #cbd5e1;
                font-size: 14px;
            }
            #PreviewBox {
                background: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 16px;
                color: #94a3b8;
            }
            QComboBox, QLineEdit {
                background: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 12px;
                padding: 8px 10px;
            }
            QPushButton {
                background: #111827;
                border: 1px solid #1f2937;
                padding: 10px 14px;
                border-radius: 12px;
                font-weight: 700;
            }
            QPushButton:hover {
                border: 1px solid #334155;
            }
            QPushButton:disabled {
                background: #0b1220;
                color: #64748b;
                border: 1px solid #1f2937;
            }
            #PrimaryBtn {
                background: #2563eb;
                border: 1px solid #1d4ed8;
                font-weight: 900;
                padding: 12px 14px;
                border-radius: 12px;
            }
            #PrimaryBtn:hover {
                background: #1d4ed8;
            }
            #ResultTitle {
                font-size: 14px;
                font-weight: 900;
                color: #e5e7eb;
            }
        """)

    # -------------------------
    # Model Handling
    # -------------------------
    def on_model_change(self, model_name: str):
        """Update UI when model changes."""
        self.model_path_edit.setText(self.model_paths[model_name])
        self.btn_predict.setEnabled(self.current_image_path is not None)

    def pick_model_file(self):
        """Pick a custom model file for the selected slot."""
        model_name = self.model_combo.currentText()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Keras model file",
            "",
            "Keras Model (*.keras *.h5);;All Files (*)"
        )
        if path:
            self.model_paths[model_name] = path
            self.model_path_edit.setText(path)
            if model_name in self.models_cache:
                del self.models_cache[model_name]

    def get_model(self) -> keras.Model:
        """Load the currently selected model (cached)."""
        name = self.model_combo.currentText()
        if name in self.models_cache:
            return self.models_cache[name]

        path = self.model_paths[name]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found:\n{path}\n\n"
                "Put the model next to app.py OR click 'Set Model File…'."
            )

        model = keras.models.load_model(path)
        self.models_cache[name] = model
        return model

    # -------------------------
    # Image Handling
    # -------------------------
    def browse_image(self):
        """Browse for an image file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        """Load image and update preview."""
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open image:\n{e}")
            return

        self.current_image_path = path
        self.current_pixmap_raw = pil_to_qpixmap(pil_img)

        self._refresh_preview()
        self.btn_predict.setEnabled(True)
        self.result_box.setText(f"Loaded: {os.path.basename(path)}\n\nClick Predict to get results.")

    def _refresh_preview(self):
        """Scale current preview image to fit label area."""
        if self.current_pixmap_raw is None:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No image loaded")
            return

        pad = 16  # smaller padding so image looks larger
        target_w = max(100, self.preview.width() - pad)
        target_h = max(100, self.preview.height() - pad)

        scaled = self.current_pixmap_raw.scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(scaled)

    def resizeEvent(self, event):
        """Keep preview fitted on resize."""
        super().resizeEvent(event)
        self._refresh_preview()

    def clear_all(self):
        """Reset everything."""
        self.current_image_path = None
        self.current_pixmap_raw = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText("No image loaded")
        self.btn_predict.setEnabled(False)
        self.result_box.setText("Result: —")

    # -------------------------
    # Prediction
    # -------------------------
    def run_prediction(self):
        """Run prediction and print a clear formatted output."""
        if not self.current_image_path:
            QMessageBox.information(self, "No image", "Load an image first (Browse or Drag & Drop).")
            return

        try:
            model = self.get_model()
            h, w = get_model_input_hw(model)
            results = topk_predictions(self.current_image_path, model, k=3)

            top1_cls, top1_name, top1_conf = results[0]

            # Build a readable output text
            lines = []
            lines.append(f"Model: {self.model_combo.currentText()}")
            lines.append(f"Input Size: {w}x{h}")
            lines.append("")
            lines.append(f"TOP-1: {top1_name} (Class {top1_cls})  |  {top1_conf*100:.2f}%")
            lines.append("")
            lines.append("Top-3:")
            for rank, (cls, name, conf) in enumerate(results, start=1):
                lines.append(f"  #{rank}  ->  {name} (Class {cls})  |  {conf*100:.2f}%")

            self.result_box.setText("\n".join(lines))

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TrafficSignPredictor()
    w.show()
    sys.exit(app.exec())