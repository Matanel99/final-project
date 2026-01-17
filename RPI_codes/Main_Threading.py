from collections import deque
import time
import cv2
import threading
import queue

from WebcamModule import Webcam
from LaneDetectionModule import process_frame
from MotorModule import Move
from PIDModule import PIDController
from picarx import Picarx
from SignHandlerModule import SignHandler


def clamp(v, lo, hi):
    # Clamp value v into [lo, hi]
    return lo if v < lo else hi if v > hi else v


def median_of_deque(d):
    # Return median of a deque (or 0.0 if empty)
    if not d:
        return 0.0
    s = sorted(d)
    return s[len(s) // 2]


def bbox_xywh_to_xyxy(bbox_xywh):
    """
    Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2).
    """
    x, y, w, h = bbox_xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return (x1, y1, x2, y2)


def ensure_xyxy(bbox):
    """
    Ensure bbox is in (x1, y1, x2, y2) format.
    If it looks like (x, y, w, h), convert it accordingly.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 numbers")

    x1, y1, a, b = bbox

    # If a < x1 or b < y1 then treat a,b as width/height and convert to xyxy
    if (a < x1) or (b < y1):
        return bbox_xywh_to_xyxy(bbox)

    return (x1, y1, a, b)


def filter_detections(detections, frame_shape,
                      min_box_area_frac=0.01,
                      edge_margin_frac=0.07,
                      min_conf=0.40):
    """
    Filter raw detections to remove obvious false positives:
    - very small boxes
    - boxes too close to frame edges
    - low confidence detections
    """
    if detections is None:
        return []

    h, w = frame_shape[0], frame_shape[1]
    frame_area = w * h

    cleaned = []
    for det in detections:
        if len(det) == 2:
            sign_name, bbox_xyxy = det
            conf = 1.0
        else:
            sign_name, bbox_xyxy, conf = det

        (x1, y1, x2, y2) = bbox_xyxy
        bw = max(1, (x2 - x1))
        bh = max(1, (y2 - y1))
        box_area = bw * bh

        # Reject boxes that are too small relative to the frame
        if (box_area / frame_area) < min_box_area_frac:
            continue

        # Reject boxes too close to the image edges
        margin_x = w * edge_margin_frac
        margin_y = h * edge_margin_frac
        near_left   = x1 <= margin_x
        near_right  = x2 >= (w - margin_x)
        near_top    = y1 <= margin_y
        near_bottom = y2 >= (h - margin_y)
        if near_left or near_right or near_top or near_bottom:
            continue

        # Reject by confidence threshold
        if conf < min_conf:
            continue

        cleaned.append((sign_name, bbox_xyxy, conf))

    return cleaned


# ------------- Background YOLO worker -------------
class SignDetectWorker:
    """
    Background thread that runs YOLO sign detection:
    - Receives frames via a queue (keeps at most 1 newest)
    - Stores last detection result + timestamp
    """
    def __init__(self, detector, min_gap=1.5):
        self.detector = detector
        self.q = queue.Queue(maxsize=1)

        self.result = None
        self.result_time = 0.0
        self.last_run = 0.0

        self.min_gap = float(min_gap)
        self.lock = threading.Lock()
        self.stop_flag = False

        self.worker_busy = False

        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while not self.stop_flag:
            try:
                frame = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            self.worker_busy = True
            t0 = time.perf_counter()

            det = self.detector.detect_sign(frame)
            # det is expected to be:
            # [("red_light", [x,y,w,h], conf), ("stop_sign",[x,y,w,h], conf), ...]

            dt_ms = (time.perf_counter() - t0) * 1000.0
            now_t = time.time()

            with self.lock:
                self.result = det
                self.result_time = now_t
                self.last_run = now_t

            self.worker_busy = False

            print(f"took {dt_ms:.1f} ms, detections={len(det) if det else 0}")

    def push_frame(self, frame):
        # Request a new detection if worker is idle and enough time passed
        now = time.time()

        # If worker still running on previous frame, skip
        if self.worker_busy:
            return

        # Enforce minimum time gap between runs
        if (now - self.last_run) < self.min_gap:
            return

        # Drop stale frame in queue, keep only fresh one
        try:
            self.q.get_nowait()
            print("dropped stale frame")
        except queue.Empty:
            pass

        try:
            self.q.put_nowait(frame)
        except queue.Full:
            pass

    def get_result_with_age(self):
        # Return last detection and how old it is (in seconds)
        with self.lock:
            det = self.result
            det_time = self.result_time
        if det_time == 0.0:
            return None, float("inf")
        age = time.time() - det_time
        return det, age

    def stop(self):
        # Signal the worker thread to stop
        self.stop_flag = True


def normalize_detections_to_xyxy(det_list):
    """
    Normalize all detection bboxes to (x1, y1, x2, y2) format.
    Keeps (sign_name, bbox[, conf]) structure.
    """
    if not det_list:
        return []

    norm = []
    for item in det_list:
        if len(item) == 2:
            sign_name, bbox = item
            conf = None
        else:
            sign_name, bbox, conf = item

        bbox_xyxy = ensure_xyxy(bbox)

        if conf is None:
            norm.append((sign_name, bbox_xyxy))
        else:
            norm.append((sign_name, bbox_xyxy, conf))

        return norm


# ---------------- Main ----------------
def main():
    px = Picarx()
    webcam = Webcam()
    motor_controller = Move(px, speed=0, curve=0, cam_angle=0)
    sign_handler = SignHandler(motor_controller)

    # PID levels tuned for different curve strengths
    pid_very_sharp = PIDController(kp=0.45, ki=0.0, kd=0.25)
    pid_sharp      = PIDController(kp=0.40, ki=0.0, kd=0.22)
    pid_med        = PIDController(kp=0.35, ki=0.0, kd=0.18)
    pid_smooth     = PIDController(kp=0.28, ki=0.0, kd=0.12)

    DEADBAND = 10                          # Small curve values around zero are treated as straight
    curve_hist = deque(maxlen=5)           # Short history of curves for smoothing
    EMA_ALPHA = 0.30                       # EMA weight for curve smoothing

    last_sign = 0                          # Last sign of curve (left/right/0)
    sign_stable_count = 0                  # How many frames sign direction stayed consistent
    SIGN_HOLD_FRAMES = 4                   # Require sign stability before applying curve sign flip

    last_steer = 0                         # Last steering command
    MAX_STEER_DELTA = 6                    # Max steering change per frame

    last_cam_angle = 0                     # Camera angle state
    CAM_FOLLOW = 0.15                      # How strongly camera follows steering
    max_cam_speed = 5                      # Max camera angle change per frame
    CAM_CENTER_BIAS = 0.4                  # How much camera prefers to stay near center

    # === YOLO sign detector setup ===
    from YOLOSignDetector_file import YOLOSignDetector
    sign_detector = YOLOSignDetector(
        onnx_path="/home/matanelcohenpi/python_scripts/final_project_scripts/final_best.onnx",
        yaml_path="/home/matanelcohenpi/python_scripts/final_project_scripts/final_data.yaml",
        input_wh=640
    )
    detect_worker = SignDetectWorker(sign_detector, min_gap=1.5)

    # === Sign-handling state machine ===
    waiting_for_green = False       # True while we are stopped at a red/yellow light
    last_green_check  = 0.0
    GREEN_CHECK_GAP   = 0.8         # Minimum gap between green-light checks while waiting

    sign_cooldown_until = 0.0       # Time until which we ignore new signs
    last_action_time    = 0.0

    CALM_CURVE_THRESH   = 40        # Max allowed curve to consider path "calm" for sign checks
    SPEED_ACTIVE_MIN    = 4         # Minimal forward speed considered as "moving"
    FORWARD_STEADY_REQ  = 1.5       # Required time of forward movement before checking new signs

    forward_motion_start = time.time()

    MAX_DET_AGE = 2.0              # Ignore detection results older than this (seconds)

    try:
        while True:
            now = time.time()
            frame = webcam.get_frame()

            # -------- Lane detection / steering base --------
            curve_raw, imgResult = process_frame(frame)

            if -DEADBAND < curve_raw < DEADBAND:
                curve_raw = 0.0

            curve_hist.append(curve_raw)
            m_val = median_of_deque(curve_hist)
            last_val = curve_hist[-1] if len(curve_hist) > 0 else m_val
            curve_smoothed = EMA_ALPHA * m_val + (1.0 - EMA_ALPHA) * last_val

            # Prevent sudden sign flips: require stable sign for a few frames
            current_sign = 0 if curve_smoothed == 0 else (1 if curve_smoothed > 0 else -1)
            if current_sign != 0 and current_sign != last_sign:
                sign_stable_count += 1
                if sign_stable_count < SIGN_HOLD_FRAMES:
                    curve_for_pid = 0.0
                else:
                    last_sign = current_sign
                    sign_stable_count = 0
                    curve_for_pid = curve_smoothed
            else:
                sign_stable_count = 0
                curve_for_pid = curve_smoothed

            # Select PID tier based on absolute curve magnitude
            abs_curve = abs(curve_for_pid)
            if abs_curve >= 110:
                pid = pid_very_sharp
            elif abs_curve >= 70:
                pid = pid_sharp
            elif abs_curve >= 50:
                pid = pid_med
            else:
                pid = pid_smooth

            correction = pid.compute(curve_for_pid)

            desired = clamp(int(0.5 * correction), -50, 50)
            if desired > last_steer + MAX_STEER_DELTA:
                steering_angle = last_steer + MAX_STEER_DELTA
            elif desired < last_steer - MAX_STEER_DELTA:
                steering_angle = last_steer - MAX_STEER_DELTA
            else:
                steering_angle = desired
            last_steer = steering_angle

            # Basic speed logic based on curve: slower in sharp turns
            drive_speed = 5 if abs_curve > 70 else 10

            # Track forward motion time only while actively moving and not waiting at a light
            if (not waiting_for_green) and (drive_speed >= SPEED_ACTIVE_MIN):
                # Keep forward_motion_start unchanged while driving forward
                pass
            else:
                # Reset forward timer when stopped or in waiting state
                forward_motion_start = now

            # Camera steering: follow wheels but stay smoother and slightly recentred
            cooldown_active = (now < sign_cooldown_until)
            if cooldown_active:
                effective_follow = CAM_FOLLOW * (1.0 - CAM_CENTER_BIAS)
            else:
                effective_follow = CAM_FOLLOW

            cam_target_raw = (
                effective_follow * steering_angle +
                (1.0 - effective_follow) * last_cam_angle
            )

            cam_target_limited = clamp(
                cam_target_raw,
                last_cam_angle - max_cam_speed,
                last_cam_angle + max_cam_speed
            )

            cam_angle = clamp(cam_target_limited, -45, 45)
            last_cam_angle = cam_angle

            # -------- Schedule YOLO work to background thread --------
            time_forward = now - forward_motion_start
            moved_enough_after_action = (time_forward >= FORWARD_STEADY_REQ)

            if waiting_for_green:
                # While waiting at light: push frames occasionally to look for green light
                if (now - last_green_check) >= GREEN_CHECK_GAP:
                    detect_worker.push_frame(frame)
                    last_green_check = now
            else:
                # Only check for new signs when path is calm, no cooldown, and we moved enough
                can_check_new_signs = (
                    (now >= sign_cooldown_until) and
                    (abs(curve_smoothed) < CALM_CURVE_THRESH) and
                    moved_enough_after_action
                )
                if can_check_new_signs:
                    detect_worker.push_frame(frame)

            # -------- Read latest detection result from worker --------
            det_raw, age_sec = detect_worker.get_result_with_age()

            if det_raw:
                # Optional debug: show raw detections and age
                # print(f"[YOLO][RAW] age={age_sec:.2f}s det_raw={det_raw}")
                pass

            if age_sec > MAX_DET_AGE:
                det_recent = []
            else:
                det_recent = det_raw if det_raw else []

            det_xyxy = normalize_detections_to_xyxy(det_recent)
            det_clean = filter_detections(det_xyxy, frame.shape)

            for dbg_item in det_clean:
                nm, bb, cf = dbg_item
                # Optional debug per clean detection
                # print(f"[YOLO] Detected(clean): {nm} bbox={bb} conf={cf:.2f}")

            # -------- Sign-handling state machine --------
            if waiting_for_green:
                # We are currently stopped at traffic light; look for green_light among detections
                saw_green = False

                for item in det_xyxy:
                    if len(item) == 2:
                        sign_name, bbox_xyxy = item
                    else:
                        sign_name, bbox_xyxy, _c = item

                    if sign_name == "green_light":
                        saw_green = True

                if saw_green:
                    # Green light detected => resume normal driving with short cooldown
                    # print("[STATE] GREEN detected -> resume driving")
                    waiting_for_green = False

                    now2 = time.time()
                    sign_cooldown_until = now2 + 2.0
                    last_action_time = now2
                    forward_motion_start = now2
            else:
                # Not waiting: check if we have fresh signs and no active cooldown
                if det_xyxy and now >= sign_cooldown_until:
                    handled_any = False

                    for item in det_xyxy:
                        if len(item) == 2:
                            sign_name, bbox_xyxy = item
                            conf = None
                        else:
                            sign_name, bbox_xyxy, conf = item

                        # print(f"[YOLO] Detected(norm): {sign_name} bbox={bbox_xyxy} conf={conf}")

                        # Let SignHandler decide what to do with each sign:
                        # - stop_sign / crosswalk_sign -> "handled" (full stop + wait)
                        # - red_light / yellow_light    -> "wait_for_green"
                        flag = sign_handler.handle_sign(sign_name, bbox_xyxy)

                        if flag == "wait_for_green":
                            waiting_for_green = True
                            print("[STATE] Enter waiting_for_green")
                            handled_any = True
                            break

                        elif flag == "handled":
                            # Sign fully handled (e.g., stop or crosswalk logic already done)
                            # print("[STATE] Action handled (stop/crosswalk)")
                            handled_any = True
                            break

                        elif flag == "detected_but_too_small":
                            # Sign is valid but too far/small; no action yet
                            pass

                        else:
                            # "unhandled": sign not mapped to a behavior
                            pass

                    if handled_any:
                        now2 = time.time()
                        # Start a longer cooldown after we reacted to a sign
                        sign_cooldown_until = now2 + 8.0
                        last_action_time = now2
                        forward_motion_start = now2

            # -------- Motion commands --------
            if waiting_for_green:
                # While waiting at traffic light, keep vehicle fully stopped
                motor_controller.move(speed=0, curve=0, cam_angle=cam_angle)
            else:
                motor_controller.move(
                    speed=drive_speed,
                    curve=steering_angle,
                    cam_angle=cam_angle
                )

            # -------- Obstacle avoidance with ultrasonic sensor --------
            if not waiting_for_green:
                distance = round(px.ultrasonic.read(), 2)
                if 0 < distance < 35:
                    motor_controller.move(speed=drive_speed, curve=30,  cam_angle=cam_angle); time.sleep(0.6)
                    motor_controller.move(speed=drive_speed, curve=-10, cam_angle=cam_angle); time.sleep(0.2)
                    motor_controller.move(speed=drive_speed, curve=0,   cam_angle=cam_angle); time.sleep(1.0)
                    motor_controller.move(speed=drive_speed, curve=-30, cam_angle=cam_angle); time.sleep(1.3)
                    motor_controller.move(speed=drive_speed, curve=0,   cam_angle=cam_angle); time.sleep(0.2)

            # -------- Visualization window --------
            cv2.namedWindow("Curve Detection", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Curve Detection", 0, 0)
            cv2.resizeWindow("Curve Detection", 800, 640)
            cv2.imshow("Curve Detection", imgResult)

            # cv2.imshow("Frame", frame)
            # cv2.imshow("Curve Detection", imgResult)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting program (Keyboard Interrupt).")

    finally:
        print("Stopping vehicle and releasing resources.")
        try:
            detect_worker.stop()
        except Exception:
            pass
        motor_controller.move(speed=0, curve=0, cam_angle=0)
        webcam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
