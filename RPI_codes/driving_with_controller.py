import smbus
import time
from MotorModule import Move

# I2C address of the joystick sender (ESP)
I2C_ADDR = 0x08
bus = smbus.SMBus(1)

# Allowed speed range for the car (forward/backward)
SPEED_MIN, SPEED_MAX = -30, 30
# Allowed turning range for the car (left/right)
TURN_MIN,  TURN_MAX  = -30, 23

# Raw 16-bit analog range coming from the joystick
ANALOG_MIN, ANALOG_MAX = 0, 65535

# Deadzone thresholds for X axis (turn) with hysteresis
DZ_X_ENTER = 2200
DZ_X_EXIT  = 2600

# Deadzone thresholds for Y axis (speed) with hysteresis
DZ_Y_ENTER = 900
DZ_Y_EXIT  = 1100

# Gamma values to make joystick response softer near center
SPEED_GAMMA = 2.0
TURN_GAMMA  = 2.5

# Small active braking when releasing from strong forward motion
BRAKE_THRESH = 3
BRAKE_PULSE  = -5
BRAKE_MS     = 80

# Main car controller from your MotorModule
car = Move()

# Joystick neutral points (will be measured at runtime)
center_x = None
center_y = None
last_speed = 0

# Flags to tell if each axis is currently considered "zero" (inside deadzone)
in_zero_x = True
in_zero_y = True

# Calibrate joystick center by averaging several I2C readings
def calibrate_center(samples=50, delay=0.01):
    sx = sy = 0
    for _ in range(samples):
        data = bus.read_i2c_block_data(I2C_ADDR, 0, 6)
        x_low, x_high, y_low, y_high, _, _ = data
        rx = x_low + (x_high << 8)
        ry = y_low + (y_high << 8)
        sx += rx
        sy += ry
        time.sleep(delay)
    cx = sx // samples
    cy = sy // samples
    print(f"[CAL] center_x={cx}, center_y={cy}")
    return cx, cy

# Map joystick delta (around center) to motor output with non-linear curve
def map_delta_curve(delta, from_abs, out_min, out_max, gamma):
    from_abs = max(1, from_abs)
    n = max(-1.0, min(1.0, delta / from_abs))  # keep in [-1, 1]
    mag = abs(n) ** gamma
    out_pos = int(round(out_max * mag))
    out_neg = int(round(out_min * mag))
    return out_pos if n >= 0 else out_neg

# Apply hysteresis logic so joystick does not jitter around zero
def hysteresis_delta(raw, center, in_zero, enter_dz, exit_dz):
    d = raw - center
    ad = abs(d)
    if in_zero:
        if ad <= enter_dz:
            return 0, True
        elif ad >= exit_dz:
            return d, False
        else:
            return 0, True
    else:
        if ad <= enter_dz:
            return 0, True
        else:
            return d, False

print("Listening for joystick data and controlling car...")

try:
    # First step: find joystick neutral position for X and Y
    center_x, center_y = calibrate_center()

    # Pre-calc max possible delta on each axis for normalization
    max_delta_x = max(center_x - ANALOG_MIN, ANALOG_MAX - center_x)
    max_delta_y = max(center_y - ANALOG_MIN, ANALOG_MAX - center_y)

    while True:
        try:
            # Read 6 bytes from I2C carrying X, Y and button state
            data = bus.read_i2c_block_data(I2C_ADDR, 0, 6)
            x_low, x_high, y_low, y_high, sw_low, _ = data

            # Rebuild 16-bit raw joystick values (little-endian)
            raw_x = x_low + (x_high << 8)
            raw_y = y_low + (y_high << 8)
            joy_sw = sw_low  # 0 means button pressed

            # Use hysteresis to decide if axis is zero or active
            dx, in_zero_x = hysteresis_delta(raw_x, center_x, in_zero_x, DZ_X_ENTER, DZ_X_EXIT)
            dy, in_zero_y = hysteresis_delta(raw_y, center_y, in_zero_y, DZ_Y_ENTER, DZ_Y_EXIT)

            # Convert deltas to speed (Y) and turn (X) using gamma curve
            speed = map_delta_curve(dx, max_delta_x, SPEED_MIN, SPEED_MAX, SPEED_GAMMA)
            curve = map_delta_curve(dy, max_delta_y, TURN_MIN,  TURN_MAX,  TURN_GAMMA)

            # If axis is still in deadzone, force full zero command
            if in_zero_x:
                speed = 0
            if in_zero_y:
                curve = 0

            # Short braking pulse when releasing strong forward motion
            if last_speed > BRAKE_THRESH and speed == 0:
                car.move(BRAKE_PULSE, 0, 0)
                time.sleep(BRAKE_MS / 1000.0)

            # Send final speed and curve commands to the car
            car.move(speed, curve, 0)
            last_speed = speed

            print(
                f"Joystick -> rawX:{raw_x} rawY:{raw_y} | "
                f"dx:{dx} dy:{dy} | Speed:{speed} Turn:{curve} | "
                f"Button:{'Pressed' if joy_sw == 0 else 'Released'}"
            )
            print("------------------------------------------------")

        except Exception as e:
            # If there is an I2C error, stop the car for safety
            print(f"Error reading I2C: {e}")
            car.move(0, 0, 0)

        # Main loop rate (control how fast we read and update)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping car and resetting wheels...")
    car.move(0, 0, 0)

finally:
    print("Car stopped. Exiting safely.")
    car.move(0, 0, 0)
