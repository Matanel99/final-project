import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLOSignDetector:
    def __init__(self, onnx_path, yaml_path, input_wh=640):
        self.input_wh = input_wh

        with open(yaml_path, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)
        self.labels = data_yaml["names"]

        self.yolo = cv2.dnn.readNetFromONNX(onnx_path)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_sign(self, frame, conf_threshold=0.2, score_threshold=0.2, nms_threshold=0.5, min_detect_percent=70):
        row, col, _ = frame.shape
        max_dim = max(row, col)
        input_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = frame

        blob = cv2.dnn.blobFromImage(
            input_image,
            scalefactor=1 / 255.0,
            size=(self.input_wh, self.input_wh),
            swapRB=True,
            crop=False
        )
        self.yolo.setInput(blob)
        detections = self.yolo.forward()[0]

        boxes, confidences, class_ids = [], [], []

        x_factor, y_factor = input_image.shape[1] / self.input_wh, input_image.shape[0] / self.input_wh
        for det in detections:
            confidence = det[4]
            if confidence > conf_threshold:
                class_scores = det[5:]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                if class_score > score_threshold:
                    cx, cy, w, h = det[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) == 0:
            return None

        detected_signs = []
        for i in indices.flatten():
            if confidences[i] * 100 >= min_detect_percent:
                detected_signs.append((self.labels[class_ids[i]], boxes[i]))

        return detected_signs if detected_signs else None
