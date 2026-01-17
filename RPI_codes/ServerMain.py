from collections import deque
import time
import cv2
import asyncio

from WebcamModule import Webcam
from LaneDetectionModule import process_frame
from MotorModule import Move
from PIDModule import PIDController
from picarx import Picarx

import SendFrame3 as SendFrame

from SignHandlerModule2 import SignHandler


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def resize_square_bgr(img_bgr, out_size):
    if out_size is None or out_size <= 0 or img_bgr is None:
        return img_bgr
    return cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)


def to_xywh(bbox):
    if len(bbox) != 4:
        return (0, 0, 0, 0)
    x1, y1, a, b = bbox
    if a > x1 and b > y1:
        return (x1, y1, int(a - x1), int(b - y1))
    return (int(x1), int(y1), int(a), int(b))


async def main():
    px = Picarx()
    webcam = Webcam()
    motor_controller = Move(px, speed=0, curve=0, cam_angle=0)
    sign_handler = SignHandler(motor_controller)

    pid_very_sharp = PIDController(kp=0.45, ki=0.0, kd=0.25)
    pid_sharp      = PIDController(kp=0.40, ki=0.0, kd=0.22)
    pid_med        = PIDController(kp=0.35, ki=0.0, kd=0.18)
    pid_smooth     = PIDController(kp=0.28, ki=0.0, kd=0.12)

    DEADBAND   = 10
    curve_hist = deque(maxlen=5)
    EMA_ALPHA  = 0.30

    last_sign         = 0
    sign_stable_count = 0
    SIGN_HOLD_FRAMES  = 4

    last_steer      = 0
    MAX_STEER_DELTA = 6

    last_cam_angle = 0
    CAM_FOLLOW     = 0.15
    max_cam_speed  = 5

    CALM_CURVE_THRESH = 40
    SEND_MIN_GAP      = 0.0
    GREEN_CHECK_GAP   = 0.5
    MIN_BOX_W         = 45
    MIN_BOX_H         = 45
    SEND_IMG_SIZE     = 480

    last_send_time      = 0.0
    sign_cooldown_until = 0.0
    waiting_for_green   = False
    send_pause_until    = 0.0

    detectionQueue = asyncio.Queue()
    resultsQueue   = asyncio.Queue(maxsize=1)
    in_flight_sem  = asyncio.Semaphore(1)

    async def send_latest(frame_bgr):
        frame_small = resize_square_bgr(frame_bgr, SEND_IMG_SIZE)
        start = time.perf_counter()

        async with in_flight_sem:
            await SendFrame.handleRemoteDetection(frame_small, detectionQueue)

        dets = None
        while not detectionQueue.empty():
            dets = await detectionQueue.get()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if dets and len(dets) > 0:
            label = dets[0][0]
            scale = float(SEND_IMG_SIZE) / 640.0
            min_w, min_h = int(MIN_BOX_W * scale), int(MIN_BOX_H * scale)
            try:
                w, h = int(dets[0][1][2]), int(dets[0][1][3])
            except Exception:
                w = h = 0

            if w >= min_w and h >= min_h:
                print(f"{elapsed_ms:.0f} ms - sign detected: {label}")
            else:
                print(f"{elapsed_ms:.0f} ms - No sign detected")
        else:
            print(f"{elapsed_ms:.0f} ms - No sign detected")

        try:
            while not resultsQueue.empty():
                await resultsQueue.get()
        except Exception:
            pass
        await resultsQueue.put(dets)

    try:
        while True:
            now = time.time()
            frame = webcam.get_frame()
            copy_frame = frame.copy()

            curve_raw, imgResult = process_frame(frame)
            if -DEADBAND < curve_raw < DEADBAND:
                curve_raw = 0

            curve_hist.append(curve_raw)
            median_val   = sorted(curve_hist)[len(curve_hist)//2]
            last_val     = curve_hist[-1] if len(curve_hist) > 0 else median_val
            curve_smooth = EMA_ALPHA * median_val + (1.0 - EMA_ALPHA) * last_val

            current_sign = 0 if curve_smooth == 0 else (1 if curve_smooth > 0 else -1)
            if current_sign != 0 and current_sign != last_sign:
                sign_stable_count += 1
                if sign_stable_count < SIGN_HOLD_FRAMES:
                    curve_for_pid = 0.0
                else:
                    last_sign = current_sign
                    sign_stable_count = 0
                    curve_for_pid = curve_smooth
            else:
                sign_stable_count = 0
                curve_for_pid = curve_smooth

            abs_curve = abs(curve_for_pid)
            pid = (
                pid_very_sharp if abs_curve >= 110 else
                (pid_sharp if abs_curve >= 70 else
                 (pid_med if abs_curve >= 50 else pid_smooth))
            )
            correction = pid.compute(curve_for_pid)

            desired = max(-50, min(50, int(0.5 * correction)))
            if desired > last_steer + MAX_STEER_DELTA:
                steering_angle = last_steer + MAX_STEER_DELTA
            elif desired < last_steer - MAX_STEER_DELTA:
                steering_angle = last_steer - MAX_STEER_DELTA
            else:
                steering_angle = desired
            last_steer = steering_angle

            cam_target = CAM_FOLLOW * steering_angle + (1.0 - CAM_FOLLOW) * last_cam_angle
            cam_target = clamp(cam_target, last_cam_angle - max_cam_speed, last_cam_angle + max_cam_speed)
            cam_angle  = clamp(cam_target, -45, 45)
            last_cam_angle = cam_angle

            drive_speed = 5 if abs_curve > 70 else 10

            can_send_normal = (
                (now >= sign_cooldown_until) and
                (now >= send_pause_until) and
                (abs_curve < CALM_CURVE_THRESH) and
                ((now - last_send_time) >= SEND_MIN_GAP) and
                (in_flight_sem.locked() is False)
            )
            can_send_waiting = (
                waiting_for_green and
                ((now - last_send_time) >= GREEN_CHECK_GAP) and
                (in_flight_sem.locked() is False)
            )

            if can_send_waiting or can_send_normal:
                asyncio.create_task(send_latest(copy_frame))
                last_send_time = now

            latest_dets = None
            while not resultsQueue.empty():
                latest_dets = await resultsQueue.get()

            if latest_dets is not None:
                if waiting_for_green:
                    labels = [d[0] for d in latest_dets] if latest_dets else []
                    if "green_light" in labels:
                        waiting_for_green = False
                        sign_cooldown_until = time.time() + 3.0
                else:
                    decision_made = False
                    for lbl, bbox in latest_dets:
                        x, y, w, h = to_xywh(bbox)

                        if w < MIN_BOX_W or h < MIN_BOX_H:
                            continue

                        action = sign_handler.handle_sign(lbl, [x, y, w, h])

                        if action == "stop3s":
                            motor_controller.move(speed=0, curve=0, cam_angle=cam_angle)
                            send_pause_until = time.time() + 3.0
                            await asyncio.sleep(3.0)
                            sign_cooldown_until = time.time() + 3.0
                            decision_made = True
                            break

                        elif action == "wait_for_green":
                            waiting_for_green = True
                            motor_controller.move(speed=0, curve=0, cam_angle=cam_angle)
                            decision_made = True
                            break

                        elif action == "detected_but_too_small":
                            continue
                        else:
                            continue

                    if decision_made:
                        pass

            if waiting_for_green:
                motor_controller.move(speed=0, curve=0, cam_angle=cam_angle)
            else:
                motor_controller.move(speed=drive_speed, curve=steering_angle, cam_angle=cam_angle)

            cv2.namedWindow("Curve Detection", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Curve Detection", 0, 0)
            cv2.resizeWindow("Curve Detection", 800, 640)
            cv2.imshow("Curve Detection", imgResult)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

            await asyncio.sleep(0)

    except KeyboardInterrupt:
        pass
    finally:
        motor_controller.move(speed=0, curve=0, cam_angle=0)
        webcam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import SendFrame3 as SendFrame
    print("SendFrame module:", getattr(SendFrame, "__file__", "n/a"))
    print("SERVER_ENDPOINT:", getattr(SendFrame, "SERVER_ENDPOINT", "n/a"))
    asyncio.run(main())
