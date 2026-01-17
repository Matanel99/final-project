import time

class SignHandler:
    def __init__(self, motor_controller):
        self.motor_controller = motor_controller

    def handle_sign(self, sign_name, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        if sign_name in ("stop_sign", "crosswalk_sign"):
            min_width  = 30
            min_height = 30
            if (w >= min_width) and (h >= min_height):
                print("STOP/CROSSWALK Sign Detected -> full stop for 3s")
                self.motor_controller.move(speed=0, curve=0, cam_angle=0)
                time.sleep(3)
                return "handled"
            else:
                print(f"{sign_name} too small: ({w}x{h}) < ({min_width}x{min_height})")
                return "detected_but_too_small"

        if sign_name in ("red_light", "yellow_light"):
            min_width  = 10
            min_height = 10
            if (w >= min_width) and (h >= min_height):
                print("RED/YELLOW -> stop now, wait for green")
                self.motor_controller.move(speed=0, curve=0, cam_angle=0)
                return "wait_for_green"
            else:
                print(f" {sign_name} too small: ({w}x{h}) < ({min_width}x{min_height})")
                return "detected_but_too_small"

        if sign_name == "green_light":
            print("GREEN light detected")
            return "unhandled"

        return "unhandled"
