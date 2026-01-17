from WebcamModule import Webcam           
from LaneDetectionModule import process_frame  
from MotorModule import Move       
import cv2
import time
from PIDModule import PIDController  
from picarx import Picarx        
from YOLOSignDetector_file import YOLOSignDetector
from SignHandlerModule import SignHandler          


def to_xyxy(bbox):
    # Convert bbox from (x, y, w, h) into a standard (x1, y1, x2, y2) format
    if bbox is None or len(bbox) != 4:
        return bbox

    x1, y1, a, b = bbox

    # If a,b look like bottom-right corner (greater than top-left), assume (x1, y1, x2, y2)
    if a > x1 and b > y1:
        return (int(x1), int(y1), int(a), int(b))
    else:
        # Otherwise treat a,b as width/height and convert (x, y, w, h) -> (x1, y1, x2, y2)
        w, h = a, b
        return (int(x1), int(y1), int(x1 + w), int(y1 + h))


def main():
    px = Picarx()
    webcam = Webcam()
    motor_controller = Move(px, speed=0, curve=0, cam_angle=0)
    sign_handler = SignHandler(motor_controller)


    pid_very_sharp = PIDController(kp=0.55, ki=0.03, kd=0.2)
    pid_sharp      = PIDController(kp=0.5,  ki=0.02, kd=0.15)
    pid_med        = PIDController(kp=0.4,  ki=0.02, kd=0.1)
    pid_smooth     = PIDController(kp=0.3,  ki=0.01, kd=0.03)

    last_cam_angle = 0        # Last camera servo angle for smoothing
    max_cam_speed  = 5        # Max change in camera angle per frame

    # YOLO-based traffic sign detector configuration
    sign_detector = YOLOSignDetector(
        onnx_path="/home/matanelcohenpi/python_scripts/final_project_scripts/final_best.onnx",
        yaml_path="/home/matanelcohenpi/python_scripts/final_project_scripts/final_data.yaml",
        input_wh=640
    )

    last_detection_time = 0.0 # Last time a detection was triggered
    DETECTION_INTERVAL  = 2.0 # Minimum seconds between detection attempts
    ignore_signs_until  = 0.0 # Cooldown time until signs are ignored

    try:
        while True:
            current_time = time.time()
            frame = webcam.get_frame()

            # ----- Lane detection & steering control -----
            curve, imgResult = process_frame(frame)

            abs_curve = abs(curve)
            # Choose PID profile according to how strong the curve is
            if abs_curve >= 110:
                pid = pid_very_sharp
            elif abs_curve >= 70:
                pid = pid_sharp
            elif abs_curve >= 50:
                pid = pid_med
            else:
                pid = pid_smooth

            error = curve                      # PID error is simply the curve value
            correction = pid.compute(error)    # PID output for steering

            steering_angle = int(0.5 * correction)
            steering_angle = max(-50, min(50, steering_angle))  # Clamp steering to allowed range

            # Camera follows steering with smoothing and limited movement speed
            cam_angle_target = steering_angle
            cam_angle = 0.6 * last_cam_angle + 0.4 * cam_angle_target
            cam_angle = max(last_cam_angle - max_cam_speed,
                            min(last_cam_angle + max_cam_speed, cam_angle))
            last_cam_angle = cam_angle

            # Simple speed logic: slower on strong curves, faster on smoother segments
            speed = 5 if abs_curve > 70 else 10
            cam_angle = max(-45, min(45, cam_angle))  # Clamp camera angle to physical limits

            # ----- Sign detection  -----
            if (current_time >= ignore_signs_until and
                current_time - last_detection_time >= DETECTION_INTERVAL and
                abs_curve < 50):

                detected_signs = sign_detector.detect_sign(frame)
                print("try to detect")

                if detected_signs:
                    for sign_name, bbox in detected_signs:
                        bbox_xyxy = to_xyxy(bbox)
                        print(f"[MAIN] {sign_name} raw bbox={bbox} -> xyxy={bbox_xyxy}")

                        # Use SignHandler to decide how to react to the detected sign
                        flag = sign_handler.handle_sign(sign_name, bbox_xyxy)

                        # ----- Special blocking behavior for traffic lights -----
                        if flag == "wait_for_green":
                            motor_controller.move(speed=0, curve=0, cam_angle=cam_angle)
                            print("[MAIN] Waiting for GREEN light ...")

                            while True:
                                time.sleep(0.15)
                                frame2 = webcam.get_frame()

                                # We still run lane processing only for visualization / consistency
                                _, imgResult2 = process_frame(frame2)

                                lights = sign_detector.detect_sign(frame2) or []

                                cv2.imshow("Frame", frame2)
                                cv2.imshow("Curve Detection", imgResult2)
                                # Allow user to exit even while waiting at red light
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    raise KeyboardInterrupt

                                if any(sn == "green_light" for sn, _ in lights):
                                    print("[MAIN] GREEN detected, resuming")
                                    break

                                # While waiting, keep the vehicle fully stopped
                                motor_controller.move(speed=0, curve=0, cam_angle=cam_angle)

                            # After green, treat the sign as fully handled and start cooldown
                            flag = "handled"

                        # ----- Cooldown logic after handling a sign -----
                        if flag == "handled":
                            ignore_signs_until = current_time + 6.0  # Ignore new signs for a few seconds

                        elif flag == "detected_but_too_small":
                            print(f"Detected sign: {sign_name}, bbox: {bbox_xyxy}, but sign is too far/small")
                            # Force next detection attempt immediately to re-check when closer
                            last_detection_time = current_time - DETECTION_INTERVAL
                            continue

                last_detection_time = current_time

         
            motor_controller.move(speed=speed, curve=steering_angle, cam_angle=cam_angle)

            # ----- Simple obstacle avoidance using ultrasonic sensor -----
            distance = round(px.ultrasonic.read(), 2)
            if 0 < distance < 28:
                motor_controller.move(speed=speed, curve=30, cam_angle=cam_angle)
                time.sleep(0.8)
                motor_controller.move(speed=speed, curve=-10, cam_angle=cam_angle)
                time.sleep(0.2)
                motor_controller.move(speed=speed, curve=0, cam_angle=cam_angle)
                time.sleep(1.5)
                motor_controller.move(speed=speed, curve=-30, cam_angle=cam_angle)
                time.sleep(0.8)
                motor_controller.move(speed=speed, curve=0, cam_angle=cam_angle)
                time.sleep(0.2)

            # ----- Visualization windows -----
            cv2.imshow("Frame", frame)
            cv2.imshow("Curve Detection", imgResult)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting program (Keyboard Interrupt).")
    finally:
        
        print("Stopping vehicle and releasing resources.")
        motor_controller.move(speed=0, curve=0, cam_angle=0)
        webcam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
