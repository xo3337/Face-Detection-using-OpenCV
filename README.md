
# 👤 Face Detection in Video using OpenCV

This project uses OpenCV's deep learning module (`cv2.dnn`) to detect human faces in a video file using a pre-trained model based on SSD and ResNet10.


## 🎞️ Demonstration video of the output
https://github.com/user-attachments/assets/ed514561-d78c-47bc-8ce9-b3a7d2c3aaf9











## 📦 Requirements

- OpenCV
- NumPy

### Install dependencies:
```bash
pip install opencv-python numpy
```

---

## 📁 Files Needed

1. `res10_300x300_ssd_iter_140000.caffemodel`  
 - [Download Link](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
 - made py [mshabunin](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
2. `deploy.prototxt`  
 - [Download Link](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
 - made py [mshabunin](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

4. A video file to test, e.g., `your_video.mp4`.

---

## 📄 Code

```python
import cv2

# Load the pre-trained face detector model
config_file = " file_path/deploy.prototxt"
model_file = "C:file_path/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Load video file instead of webcam
video_path = "file_path/your_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Face: {confidence*100:.2f}%"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection - Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ▶️ How to Run

1. Make sure `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`, and your video file are placed correctly.
2. Update the file paths in the script to match your system.
3. Run the script with:

```bash
python face_detection_video.py
```

---

## 📌 Notes

- Press **`q`** to quit while the video is playing.
- Works with common formats like `.mp4`, `.avi`, etc.
