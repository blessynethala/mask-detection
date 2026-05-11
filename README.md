# 😷 Face Mask Detection

A real-time face mask detection system using OpenCV and Deep Learning.

## 🔍 How it Works
- Detects faces using OpenCV's DNN module
- Classifies each face as **Mask** or **No Mask**
- Draws bounding boxes with confidence scores

## 🛠️ Tech Stack
- Python
- OpenCV
- TensorFlow / Keras
- NumPy

## 📁 Files
- `mask_detection.py` - Main detection script
- `mask_detector.h5` - Trained mask classifier model
- `deploy.prototxt` - Face detector config

## ▶️ How to Run
```bash
pip install -r requirements.txt
python mask_detection.py
```
