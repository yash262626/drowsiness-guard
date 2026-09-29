<div align="center">

# 👁️ Drowsiness Guard

**AI-powered driver safety system with face unlock: EfficientNetB0 + DeepFace/FaceNet**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

---

## 🚀 Features

| Feature | Technology | Details |
|---------|-----------|---------|
| 😴 **Drowsiness Detection** | EfficientNetB0 (Transfer Learning) | ~97% expected accuracy, ~20ms inference |
| 🔐 **Face Unlock** | DeepFace + FaceNet | 1-shot registration, cosine similarity |
| 📊 **Dataset** | MRL + CEW merged | 87,000+ eye images |
| 🔔 **Alarm System** | pygame audio + visual flash | Triggers at 2 seconds closed |

---

## 📁 Project Structure

```
drowsiness_guard/
├── app.py                         ← Main Streamlit app
├── requirements.txt
│
├── modules/
│   ├── face_unlock.py             ← DeepFace registration & verification
│   ├── drowsiness_detector.py     ← EfficientNetB0 + dlib eye detection
│   └── alarm.py                   ← Pygame audio + visual alarm
│
├── train/
│   └── train_model.py             ← EfficientNetB0 training (2-phase)
│
├── dataset/
│   └── download_datasets.py       ← Auto-download MRL + CEW + merge
│
├── models/
│   ├── drowsiness_model.keras     ← (generated after training)
│   └── shape_predictor_68_face_landmarks.dat  ← (auto-downloaded)
│
├── registered_faces/              ← Face photos + registry.json
└── sounds/
    └── alarm.wav                  ← (auto-generated)
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & Merge Datasets
```bash
python dataset/download_datasets.py
```
Downloads **MRL Eye Dataset** (84,898 images) + **CEW Dataset** (2,420 images) and merges into `dataset/merged/open/` and `dataset/merged/closed/`.

### 3. Train the Model
```bash
python train/train_model.py
```
- **Phase 1** (10 epochs): Trains classification head with frozen EfficientNetB0
- **Phase 2** (10 epochs): Fine-tunes full network
- Expected accuracy: **~96-98%** (train it yourself to confirm; the trained model is not included in this repo)
- GPU: ~45 minutes | CPU: ~4-6 hours

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🧠 ML Model: EfficientNetB0

**Why EfficientNetB0 over other models?**

| Model | Accuracy | CPU Speed | Size | Choice |
|-------|----------|-----------|------|--------|
| **EfficientNetB0** | **~97%** | **~20ms** | **20MB** | ✅ **Best** |
| MobileNetV2 | ~94% | ~15ms | 14MB | Runner-up |
| ResNet50 | ~96% | ~45ms | 98MB | Too slow |
| VGG16 | ~95% | ~80ms | 528MB | Too heavy |

**Pipeline:**
```
Webcam Frame
     ↓
dlib 68 Landmark Detector
     ↓
Eye Region Crop (left + right)
     ↓
EfficientNetB0 Classifier
     ↓
Consecutive Closed Frame Counter
     ↓
20 frames (≈2 sec) → ALARM! 🚨
```

---

## 🔐 Face Unlock

**How it works:**
1. Register your face via webcam (1 photo minimum)
2. DeepFace extracts 128-dim FaceNet embedding
3. Live webcam frame compared using cosine similarity
4. Threshold: 0.40 (adjustable in `modules/face_unlock.py`)

**Supports:** Multiple users, delete/re-register, 1:1 or 1:N verification

---

## 📊 Datasets

### MRL Eye Dataset
- **Source:** http://mrl.cs.vsb.cz/eyedataset
- **Images:** 84,898 (multiple subjects, lighting conditions, IR camera)
- **Classes:** open / closed
- **Kaggle mirror:** `prasadvpatil/mrl-dataset`

### CEW (Closed Eyes in the Wild)
- **Source:** http://parnec.nuaa.edu.cn/xtan/data/ClosedEyes.zip
- **Images:** 2,420 (real-world photos)
- **Classes:** openFace / closedFace
- **Kaggle mirror:** `serenaraju/closed-eyes-in-the-wild-cew-dataset`

---

## 🔔 Alarm System

- **Audio:** Dual-tone siren (880Hz/660Hz) via pygame — loops until dismissed
- **Visual:** Red flash overlay on webcam feed
- **Trigger:** 20 consecutive closed-eye frames (≈2 seconds at 30fps)
- **Dismiss:** Click "Dismiss Alarm" button

---

## 🛠️ Tech Stack

```
Python 3.10+
├── streamlit          — UI
├── deepface           — Face verification (FaceNet backend)
├── tensorflow/keras   — EfficientNetB0 training & inference
├── dlib               — 68-point facial landmark detection
├── opencv-python      — Webcam + image processing
├── pygame             — Audio alarm
└── scipy              — EAR calculation (fallback)
```

---

## 🔒 Privacy

Registered face photos and the trained model stay on your machine. They are git-ignored and never uploaded.

---

## 📄 License

This project is source-available under the [Yash AIL Source-Available License](LICENSE): you may view, copy and modify it for personal, educational and non-commercial local use. **Deploying/hosting it online and selling it are not allowed.** Built for educational and portfolio purposes.

<div align="center">

Built by [Yash Dhanraj Ail](https://github.com/yash262626) · [Portfolio](https://yashail.netlify.app)

</div>
