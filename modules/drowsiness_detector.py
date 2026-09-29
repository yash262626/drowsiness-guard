"""
drowsiness_detector.py  —  Real-time Drowsiness Detection
===========================================================
Pipeline:
  1. Webcam frame  →  dlib 68-landmark face detector
  2. Extract left + right eye regions (crop + pad)
  3. Feed into EfficientNetB0 (trained on MRL + CEW dataset)
  4. Track consecutive "closed" predictions
  5. Trigger alarm after CLOSED_THRESHOLD frames (~2 seconds at 30fps)

Fallback (no trained model):
  Uses EAR (Eye Aspect Ratio) geometric method — works without any ML model.
  Less accurate in extreme lighting but zero training required.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque

try:
    import dlib
    DLIB_OK = True
except ImportError:
    DLIB_OK = False

try:
    import tensorflow as tf
    TF_OK = True
except ImportError:
    TF_OK = False

try:
    from scipy.spatial import distance as dist
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR            = Path(__file__).parent.parent
MODEL_PATH          = BASE_DIR / "models" / "drowsiness_model.keras"
DLIB_PREDICTOR_PATH = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"

IMG_SIZE            = (224, 224)
CLOSED_THRESHOLD    = 20      # consecutive closed frames before alarm (≈ 2 sec at 30fps)
EAR_THRESHOLD       = 0.25    # Eye Aspect Ratio threshold for EAR fallback
CONFIDENCE_THRESHOLD= 0.5     # CNN output: < 0.5 → closed

# dlib landmark indices for left/right eye
LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))


# ─── Model Loader ────────────────────────────────────────────────────────────

_model  = None
_detector = None
_predictor = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not TF_OK or not MODEL_PATH.exists():
        return None
    _model = tf.keras.models.load_model(str(MODEL_PATH))
    return _model


def _load_dlib():
    global _detector, _predictor
    if not DLIB_OK:
        return None, None

    if _detector is None:
        _detector = dlib.get_frontal_face_detector()

    if _predictor is None:
        if not DLIB_PREDICTOR_PATH.exists():
            _download_dlib_predictor()
        if DLIB_PREDICTOR_PATH.exists():
            _predictor = dlib.shape_predictor(str(DLIB_PREDICTOR_PATH))

    return _detector, _predictor


def _download_dlib_predictor():
    """Auto-download shape_predictor_68_face_landmarks.dat if missing."""
    import urllib.request
    import bz2

    bz2_path = DLIB_PREDICTOR_PATH.with_suffix(".dat.bz2")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

    print("📥  Downloading dlib 68-landmark predictor (~100 MB) …")
    DLIB_PREDICTOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, str(bz2_path))
        with bz2.open(str(bz2_path), "rb") as f_in:
            data = f_in.read()
        with open(str(DLIB_PREDICTOR_PATH), "wb") as f_out:
            f_out.write(data)
        bz2_path.unlink()
        print("✅  Predictor downloaded.")
    except Exception as e:
        print(f"⚠️  Download failed: {e}. Using OpenCV fallback.")


# ─── EAR (Eye Aspect Ratio) — Geometric Fallback ─────────────────────────────

def _eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Soukupová & Čech, 2016
    """
    if not SCIPY_OK:
        return 0.3   # assume open if scipy unavailable
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


# ─── Eye Crop Extraction ─────────────────────────────────────────────────────

def _landmarks_to_array(shape) -> np.ndarray:
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def _crop_eye(frame: np.ndarray, eye_pts: np.ndarray, pad: int = 10) -> np.ndarray:
    x1, y1 = eye_pts[:, 0].min() - pad, eye_pts[:, 1].min() - pad
    x2, y2 = eye_pts[:, 0].max() + pad, eye_pts[:, 1].max() + pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    eye_crop = frame[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return None
    return cv2.resize(eye_crop, IMG_SIZE)


# ─── Core Detection ──────────────────────────────────────────────────────────

class DrowsinessState:
    """Tracks consecutive closed-eye frames and computes drowsiness score."""

    def __init__(self, window: int = 60):
        self.closed_count  = 0            # consecutive closed frames
        self.total_frames  = 0
        self.closed_history = deque(maxlen=window)   # last N frame states
        self.alarm_triggered = False
        self.last_ear      = 1.0
        self.last_conf     = 0.0          # CNN confidence (0=open, 1=closed)
        self.session_start = time.time()

    @property
    def drowsiness_score(self) -> float:
        """0.0 – 1.0 drowsiness score based on recent history."""
        if not self.closed_history:
            return 0.0
        return sum(self.closed_history) / len(self.closed_history)

    def update(self, eyes_closed: bool):
        self.total_frames += 1
        self.closed_history.append(int(eyes_closed))
        if eyes_closed:
            self.closed_count += 1
        else:
            self.closed_count = 0
            self.alarm_triggered = False

    @property
    def should_alarm(self) -> bool:
        return self.closed_count >= CLOSED_THRESHOLD

    def reset(self):
        self.closed_count    = 0
        self.alarm_triggered = False


state = DrowsinessState()


def detect_drowsiness(frame: np.ndarray) -> dict:
    """
    Main detection function called per frame.
    
    Returns:
    {
      "eyes_closed":    bool,
      "ear":            float,
      "confidence":     float,       # CNN probability eyes are closed
      "drowsy":         bool,        # alarm condition
      "drowsiness_score": float,     # 0-1 rolling score
      "closed_count":   int,
      "method":         str,         # "CNN" or "EAR"
      "annotated_frame": np.ndarray,
      "face_detected":  bool,
    }
    """
    result = {
        "eyes_closed":      False,
        "ear":              1.0,
        "confidence":       0.0,
        "drowsy":           False,
        "drowsiness_score": 0.0,
        "closed_count":     0,
        "method":           "EAR",
        "annotated_frame":  frame.copy(),
        "face_detected":    False,
    }

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector, predictor = _load_dlib()

    if detector is None:
        # Pure OpenCV fallback: Haar cascade
        return _detect_haar_fallback(frame, result)

    faces = detector(gray, 0)

    if len(faces) == 0:
        result["annotated_frame"] = _draw_no_face(frame)
        return result

    result["face_detected"] = True
    face  = faces[0]     # process first (closest) face

    if predictor is None:
        result["annotated_frame"] = _draw_no_predictor(frame)
        return result

    shape  = predictor(gray, face)
    coords = _landmarks_to_array(shape)

    left_eye  = coords[LEFT_EYE_IDX]
    right_eye = coords[RIGHT_EYE_IDX]

    ear_left  = _eye_aspect_ratio(left_eye)
    ear_right = _eye_aspect_ratio(right_eye)
    ear       = (ear_left + ear_right) / 2.0

    # ── CNN inference ──────────────────────────────────────────────────────
    model = _load_model()
    eyes_closed = False
    confidence  = 0.0

    if model is not None:
        # Crop both eyes, average prediction
        preds = []
        for eye_pts in [left_eye, right_eye]:
            crop = _crop_eye(frame, eye_pts)
            if crop is not None:
                inp = crop.astype("float32") / 255.0
                inp = np.expand_dims(inp, 0)
                pred = float(model.predict(inp, verbose=0)[0][0])
                preds.append(pred)

        if preds:
            # model output: 1 = open, 0 = closed (based on dataset class ordering)
            avg_pred   = np.mean(preds)
            confidence = 1.0 - avg_pred          # confidence eye is CLOSED
            eyes_closed = avg_pred < CONFIDENCE_THRESHOLD
            result["method"] = "CNN"
    else:
        # EAR geometric fallback
        eyes_closed = ear < EAR_THRESHOLD
        confidence  = max(0.0, (EAR_THRESHOLD - ear) / EAR_THRESHOLD)

    state.update(eyes_closed)
    state.last_ear  = ear
    state.last_conf = confidence

    result.update({
        "eyes_closed":      eyes_closed,
        "ear":              round(ear, 3),
        "confidence":       round(confidence, 3),
        "drowsy":           state.should_alarm,
        "drowsiness_score": round(state.drowsiness_score, 3),
        "closed_count":     state.closed_count,
        "annotated_frame":  _draw_overlay(frame, coords, left_eye, right_eye,
                                          ear, eyes_closed, state.should_alarm),
    })

    return result


# ─── OpenCV Haar Fallback (no dlib) ──────────────────────────────────────────

def _detect_haar_fallback(frame: np.ndarray, result: dict) -> dict:
    """Basic Haar cascade eye detection when dlib is not available."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes    = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    if len(eyes) >= 2:
        result["face_detected"] = True
        result["method"] = "Haar"
        # Assume open if 2+ eyes detected
        result["eyes_closed"] = False
    else:
        result["eyes_closed"] = True

    state.update(result["eyes_closed"])
    result["drowsy"]           = state.should_alarm
    result["drowsiness_score"] = state.drowsiness_score
    result["closed_count"]     = state.closed_count
    result["annotated_frame"]  = _draw_haar(frame, eyes, state.should_alarm)
    return result


# ─── Drawing Helpers ─────────────────────────────────────────────────────────

def _draw_overlay(frame, coords, left_eye, right_eye, ear, closed, alarm) -> np.ndarray:
    vis = frame.copy()

    # Eye contours
    color = (0, 0, 255) if closed else (0, 255, 0)
    for eye_pts in [left_eye, right_eye]:
        hull = cv2.convexHull(eye_pts)
        cv2.drawContours(vis, [hull], -1, color, 2)

    # All 68 landmarks (small dots)
    for (x, y) in coords:
        cv2.circle(vis, (x, y), 2, (255, 255, 0), -1)

    # EAR text
    cv2.putText(vis, f"EAR: {ear:.2f}", (10, vis.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Status bar
    if alarm:
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(vis, "⚠  DROWSINESS DETECTED! WAKE UP!", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    else:
        status = "EYES CLOSED" if closed else "EYES OPEN"
        bar_color = (0, 128, 255) if closed else (0, 180, 0)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), bar_color, -1)
        cv2.putText(vis, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    return vis


def _draw_no_face(frame):
    vis = frame.copy()
    cv2.rectangle(vis, (0,0), (vis.shape[1], 40), (50,50,50), -1)
    cv2.putText(vis, "No face detected", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    return vis


def _draw_no_predictor(frame):
    vis = frame.copy()
    cv2.putText(vis, "Downloading landmark model …", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    return vis


def _draw_haar(frame, eyes, alarm):
    vis = frame.copy()
    for (x,y,w,h) in eyes:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
    if alarm:
        cv2.rectangle(vis, (0,0), (vis.shape[1],50), (0,0,255), -1)
        cv2.putText(vis, "DROWSINESS DETECTED!", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return vis


def reset_state():
    """Call this when starting a new detection session."""
    global state
    state = DrowsinessState()
