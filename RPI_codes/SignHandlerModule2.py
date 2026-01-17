class SignHandler:
    def __init__(self, motor_controller):
        self.motor_controller = motor_controller

    def handle_sign(self, sign_name, bbox):
        print(f"Handling sign: {sign_name}, bbox: {bbox}")

        if sign_name in ("stop_sign", "crosswalk_sign"):
            min_width  = 40
            min_height = 40
            _, _, width, height = bbox
            if width >= min_width and height >= min_height:
                print("[SIGN] STOP/CROSSWALK -> request stop 3s")
                return "stop3s"
            else:
                print(f"[SIGN] {sign_name} too small: ({width}x{height}) < ({min_width}x{min_height})")
                return "detected_but_too_small"

        if sign_name in ("red_light", "yellow_light"):
            min_width  = 20
            min_height = 20
            _, _, width, height = bbox
            if width >= min_width and height >= min_height:
                print("[SIGN] RED/YELLOW -> wait for green")
                return "wait_for_green"

        return "unhandled"
