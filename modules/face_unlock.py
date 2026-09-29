"""
face_unlock.py  —  Face Registration & Unlock using DeepFace (Facenet backend)
================================================================================
DeepFace + Facenet:
  - 128-dim face embedding via FaceNet (Google, 99.65% LFW accuracy)
  - Cosine similarity for 1:N matching
  - No training required — works out of the box
  - Supports multiple registered faces (multi-user)
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import base64

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent.parent
FACES_DIR       = BASE_DIR / "registered_faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE   = FACES_DIR / "registry.json"

MODEL_NAME      = "Facenet"          # Most accurate; alternatives: VGG-Face, ArcFace
DETECTOR        = "opencv"           # Fast detector; alternatives: retinaface (more accurate)
DISTANCE_METRIC = "cosine"
VERIFY_THRESHOLD= 0.40               # Lower = stricter. Facenet cosine: 0.40 recommended
MAX_ATTEMPTS    = 3                  # Unlock attempts before lockout


# ─── Registry Helpers ────────────────────────────────────────────────────────

def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def get_registered_users() -> list[str]:
    """Return list of registered user names."""
    return list(_load_registry().keys())


# ─── Registration ────────────────────────────────────────────────────────────

def register_face(name: str, frame: np.ndarray) -> dict:
    """
    Register a new face from a NumPy BGR frame.
    Saves the image and records metadata in registry.json.
    
    Returns: {"success": bool, "message": str, "face_path": str or None}
    """
    if not DEEPFACE_OK:
        return {"success": False, "message": "DeepFace not installed. Run: pip install deepface"}

    name = name.strip().replace(" ", "_")
    if not name:
        return {"success": False, "message": "Name cannot be empty.", "face_path": None}

    # Detect face in frame before saving
    try:
        faces = DeepFace.extract_faces(
            img_path      = frame,
            detector_backend = DETECTOR,
            enforce_detection = True,
        )
        if not faces:
            return {"success": False, "message": "No face detected in image.", "face_path": None}
    except Exception as e:
        return {"success": False, "message": f"Face detection failed: {e}", "face_path": None}

    # Save image
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path   = FACES_DIR / f"{name}_{timestamp}.jpg"
    cv2.imwrite(str(img_path), frame)

    # Update registry
    registry = _load_registry()
    if name not in registry:
        registry[name] = []
    registry[name].append({
        "path":       str(img_path),
        "registered": timestamp,
    })
    _save_registry(registry)

    return {
        "success":   True,
        "message":   f"✅ Face registered for '{name}'!",
        "face_path": str(img_path),
    }


def delete_user(name: str) -> bool:
    """Remove a registered user and their face images."""
    registry = _load_registry()
    if name not in registry:
        return False
    for entry in registry[name]:
        p = Path(entry["path"])
        if p.exists():
            p.unlink()
    del registry[name]
    _save_registry(registry)
    return True


# ─── Verification ────────────────────────────────────────────────────────────

def verify_face(frame: np.ndarray, target_user: str = None) -> dict:
    """
    Verify a face against registered users.
    
    Args:
        frame       : NumPy BGR image from webcam
        target_user : If given, verify only against this user (faster 1:1).
                      If None, verify against ALL registered users (1:N).
    
    Returns:
        {
          "verified": bool,
          "user":     str or None,
          "distance": float,
          "message":  str,
        }
    """
    if not DEEPFACE_OK:
        return {"verified": False, "user": None, "distance": 1.0,
                "message": "DeepFace not installed."}

    registry = _load_registry()
    if not registry:
        return {"verified": False, "user": None, "distance": 1.0,
                "message": "No faces registered. Please register first."}

    # Determine which users to check
    users_to_check = {target_user: registry[target_user]} if target_user else registry

    best_match = {"verified": False, "user": None, "distance": 1.0, "message": ""}

    for user_name, entries in users_to_check.items():
        if not entries:
            continue

        for entry in entries:
            ref_path = entry["path"]
            if not Path(ref_path).exists():
                continue

            try:
                result = DeepFace.verify(
                    img1_path        = frame,
                    img2_path        = ref_path,
                    model_name       = MODEL_NAME,
                    detector_backend = DETECTOR,
                    distance_metric  = DISTANCE_METRIC,
                    enforce_detection= True,
                    silent           = True,
                )
                dist = result.get("distance", 1.0)

                if result.get("verified", False) and dist < best_match["distance"]:
                    best_match = {
                        "verified": True,
                        "user":     user_name,
                        "distance": dist,
                        "message":  f"✅ Unlocked! Welcome, {user_name.replace('_', ' ')}!",
                    }
            except Exception:
                continue   # Face not detected in this frame

    if not best_match["verified"]:
        best_match["message"] = "❌ Face not recognized. Access denied."

    return best_match


# ─── Live Frame Analyzer ─────────────────────────────────────────────────────

def draw_face_boxes(frame: np.ndarray, verified: bool, user: str = None) -> np.ndarray:
    """Draw bounding boxes + status overlay on frame."""
    vis  = frame.copy()
    color = (0, 255, 0) if verified else (0, 0, 255)
    label = f"UNLOCKED: {user.replace('_',' ')}" if verified else "ACCESS DENIED"

    try:
        faces = DeepFace.extract_faces(
            img_path         = vis,
            detector_backend = DETECTOR,
            enforce_detection= False,
        )
        for face_obj in faces:
            area = face_obj.get("facial_area", {})
            x, y, w, h = area.get("x",0), area.get("y",0), area.get("w",50), area.get("h",50)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    except Exception:
        pass

    # Status banner
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), color, -1)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    return vis


# ─── Utility: encode frame to base64 for Streamlit display ───────────────────

def frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode("utf-8")
