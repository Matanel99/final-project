from WebcamModule import Webcam          
from LaneDetectionModule import process_frame  
from MotorModule import Move             
import cv2, time
from PIDModule import PIDController     
from picarx import Picarx               
from SignHandlerModule import SignHandler  
from run_hailo_model import HailoRunner    


def main():
    print("[MAIN] Starting...")

    
    px = Picarx()
    webcam = Webcam()
    motor_controller = Move(px, speed=0, curve=0, cam_angle=0)
    sign_handler = SignHandler(motor_controller)

   
    pid_very_sharp = PIDController(kp=0.55, ki=0.03, kd=0.2)
    pid_sharp      = PIDController(kp=0.5,  ki=0.02, kd=0.15)
    pid_med        = PIDController(kp=0.4,  ki=0.02, kd=0.1)
    pid_smooth     = PIDController(kp=0.3,  ki=0.01, kd=0.03)

    # Camera angle smoothing state and max speed for camera movement
    last_cam_angle = 0
    max_cam_speed  = 5

    # Hailo inference timing (run YOLO every few seconds, not every frame)
    last_hailo_time = 0.0
    HAILO_INTERVAL  = 3.0

    # Initialize Hailo sign detection wrapper
    hailo = HailoRunner()

    try:
        got_first = False
        while True:
            # Grab current frame from the camera
            frame = webcam.get_frame()
            if not got_first:
                print("[MAIN] Got first frame.")
                got_first = True

            # Run lane detection and get curvature + visualization image
            curve, imgResult = process_frame(frame)

            # Choose PID profile based on how strong the curve is
            pid = (
                pid_very_sharp if abs(curve) >= 110 else
                pid_sharp      if abs(curve) >= 70  else
                pid_med        if abs(curve) >= 50  else
                pid_smooth
            )

            # Compute steering correction using PID on the curve value
            correction = pid.compute(curve)
            steering_angle = max(-50, min(50, int(0.5 * correction)))

            # Camera follows steering angle with smoothing and speed limit
            cam_angle = steering_angle
            cam_angle = 0.6 * last_cam_angle + 0.4 * cam_angle
            cam_angle = max(last_cam_angle - max_cam_speed,
                            min(last_cam_angle + max_cam_speed, cam_angle))
            last_cam_angle = cam_angle

            # Lower speed on strong curves, higher speed on softer curves
            speed = 5 if abs(curve) > 70 else 10

            # Clamp camera angle to physical limits
            cam_angle = max(-45, min(45, cam_angle))

            # Periodically run Hailo inference (not every frame to save resources)
            now = time.time()
            if hailo.ready and (now - last_hailo_time >= HAILO_INTERVAL):
                last_hailo_time = now
                sign_name = hailo.infer(frame)
                if sign_name:
                    try:
                        # Let SignHandler decide what to do with the detected sign
                        sign_handler.handle_sign(sign_name, None)
                    except Exception as e:
                        print("SignHandler error:", e)
                else:
                    print("[Hailo] no dets")

            # Send drive command to the car (steering + speed + camera angle)
            motor_controller.move(speed=speed, curve=steering_angle, cam_angle=cam_angle)

            # Basic obstacle avoidance using ultrasonic distance sensor
            distance = round(px.ultrasonic.read(), 2)
            if 0 < distance < 28:
                motor_controller.move(speed=speed, curve=30,  cam_angle=cam_angle); time.sleep(0.8)
                motor_controller.move(speed=speed, curve=-10, cam_angle=cam_angle); time.sleep(0.2)
                motor_controller.move(speed=speed, curve=0,   cam_angle=cam_angle); time.sleep(1.5)
                motor_controller.move(speed=speed, curve=-30, cam_angle=cam_angle); time.sleep(0.8)
                motor_controller.move(speed=speed, curve=0,   cam_angle=cam_angle); time.sleep(0.2)

            # Show raw frame and lane detection result window
            cv2.imshow("Frame", frame)
            cv2.imshow("Curve Detection", imgResult)

            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting program (Keyboard Interrupt).")
    finally:
        # Cleanup: stop Hailo, stop car, stop camera, close windows
        print("[MAIN] Stopping vehicle and releasing resources.")
        try:
            hailo.close()
        except Exception:
            pass
        motor_controller.move(speed=0, curve=0, cam_angle=0)
        webcam.stop()
        cv2.destroyAllWindows()
        print("[MAIN] Bye.")


if __name__ == "__main__":
    main()
