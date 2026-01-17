# Server3.py
# Flask server: receives JPEG (multipart/form-data), decodes with OpenCV, runs YOLO,
# returns JSON: {"success": True, "result": [...], "t": {"read_ms":..,"decode_ms":..,"infer_ms":..}}

from flask import Flask, request, jsonify
import time
import cv2
import numpy as np
from YOLOSignDetector_file import YOLOSignDetector


sign_detector = YOLOSignDetector(
    onnx_path="/Users/matanelchoen/Desktop/raspberry pi/final_best.onnx",
    yaml_path="/Users/matanelchoen/Desktop/raspberry pi/final_data.yaml",
    input_wh=640
)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return "POST /analyzeImage (multipart/form-data, key=image)", 200

@app.route("/analyzeImage", methods=["POST"])
def analyze_image():
    t0 = time.perf_counter()
    try:
        
        if "image" not in request.files:
            return jsonify({"success": False, "error": 'no file "image"'}), 400

        file = request.files["image"]
        buf  = file.read()
        t_read = (time.perf_counter() - t0) * 1000.0

        # Decode JPEG → OpenCV BGR
        t1 = time.perf_counter()
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "bad image"}), 400
        t_decode = (time.perf_counter() - t1) * 1000.0

        
        t2 = time.perf_counter()
        result = sign_detector.detect_sign(img)  # [("label", [x,y,w,h], conf), ...]
        t_infer = (time.perf_counter() - t2) * 1000.0

        

        return jsonify({
            "success": True,
            "result": result,
            "t": {"read_ms": round(t_read), "decode_ms": round(t_decode), "infer_ms": round(t_infer)}
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)


