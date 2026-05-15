# 😷 Face Mask Detection

A real-time face mask detection system using OpenCV and Deep Learning.

## 🔍 How it Works

- Detects faces using OpenCV's DNN module (Caffe model)
- Classifies each face as **Mask** or **No Mask**
- Draws bounding boxes with confidence scores

## 🛠️ Tech Stack

- Python 3.11
- OpenCV 4.8.0
- TensorFlow 2.13.0 / Keras
- NumPy 1.24.3

## 📁 Files

- `mask_detection.py` - Main detection script
- `mask_detector.h5` - Trained mask classifier model
- `deploy.prototxt` - Face detector config
- `res10_300x300_ssd_iter_140000.caffemodel` - Pre-trained face detector weights

## ▶️ How to Run

Install dependencies:
```bash
pip install tensorflow==2.13.0 opencv-python==4.8.0.76 numpy==1.24.3
```

Run the detector:
```bash
python mask_detection.py
```

Press **Q** to quit the webcam window.
