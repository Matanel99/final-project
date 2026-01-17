from picarx import Picarx
import time

class Move:

    def __init__(self, px=None, speed=0, curve=0, cam_angle= 0):
        self.px = px if px is not None else Picarx()
        self.speed = speed
        self.curve = curve
        self.cam_angle = cam_angle
        self.px.forward(self.speed)
        self.px.set_dir_servo_angle(self.curve)
        
    def move(self, speed=None, curve=None, cam_angle = None):
        speed = speed if speed is not None else self.speed
        curve = curve if curve is not None else self.curve
        cam_angle = cam_angle if cam_angle is not None else self.cam_angle
        cam_angle = max(-60, min(60, cam_angle))
        self.px.set_dir_servo_angle(curve)
        self.px.set_cam_pan_angle(cam_angle)
        self.px.forward(speed)
