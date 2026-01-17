import time

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        delta_time = current_time - self.last_time
        if delta_time <= 0:
            delta_time = 1e-6

        proportional = self.kp * error
        self.integral += error * delta_time
        integral = self.ki * self.integral
        derivative = self.kd * (error - self.previous_error) / delta_time

        self.previous_error = error
        self.last_time = current_time

        return proportional + integral + derivative

if __name__ == "__main__":
    pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
    test_error = -0.3
    correction = pid.compute(test_error)
    print(f"Correction value: {correction}")
