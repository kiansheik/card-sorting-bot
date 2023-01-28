import glob
import time

import aruco
import cv2
import pressure_svm as psvm
import serial
from mtg_card_detector import detect_and_guess

GRBL_POLL_TIME = 0.25
Z_STEP_OFF = 20


def serial_write(ser, msg):
    ser.flushInput()
    ser.flushOutput()
    ser.write(f"{msg}\r".encode())
    time.sleep(0.02)


def wait_till_on(ser):
    # time.sleep(0.5)
    while True:
        status = check_status(ser)
        if len(status) > 4:
            break
        time.sleep(GRBL_POLL_TIME)


# Function to parse GRBL status and check for alarms
def check_status(ser):
    serial_write(ser, "?")
    status = ser.readline().decode()
    print(status)
    return status


class GRBL_Controller:
    def __init__(self, grbl, arduino, cap):
        self.grbl = grbl
        self.arduino = arduino
        self.cap = cap
        self.calibrations = {
            # (stack_id) are the key
            # Format: (stack_row, stack_col): [(stepper_mm_x, stepper_mm_y), (aruco_center_target_x, aruco_center_target_y), aruco_id]
            (0, 0): [(-53.975, 77.638), (179, 727), 0],
            (0, 1): [(-139.823, 71.882), (102, 646), 49],
            (0, 2): [(-226.401, 72.876), (96, 634), 48],
            (0, 3): [(-308.324, 83.872), (915, 683), 48],
            (1, 0): [(-52.959, 181.817), (178, 662), 39],
            (1, 1): [(-143.873, 183.817), (194, 677), 2],
            (1, 2): [(-227.497, 186.824), (136, 704), 42],
            (1, 3): [(-306.427, 181.829), (907, 646), 42],
        }
        self.aruco_locator = aruco.ArucoLocator(
            cap, "calibration_images/calibration.json"
        )

    def init(self):
        self.wait_till_ready()
        print("TEST RELEASE...")
        self.release()
        print("HOME...")
        self.home()
        print("CALIBRATE_VACUUM...")
        self.calibrate_vacuum()
        self.wait_till_ready()
        print("INITIALIZED...")

    def deinit(self):
        self.go_to(0, 0, 0)

    def wait_till_on(self, ser):
        wait_till_on(ser)

    # Function to parse GRBL status and check for alarms
    def check_status(self, ser):
        return check_status(ser)

    def wait_till_ready(self):
        while True:
            status = self.check_status(self.grbl)
            print(status)
            if "Run" in status:
                time.sleep(0.001)
            elif "Idle" in status:
                break
            time.sleep(
                GRBL_POLL_TIME
            )  # self.GRBL wiki recommends no more than 5 times per second polling

    # Function to send step command to self.GRBL device
    def step(self, steps, axis, direction):
        if direction not in ["+", "-"]:
            raise ValueError("Invalid direction")
        command = (
            f"G21G91{axis}{direction if direction == '-' else ''}{steps-0.001}F6000"
        )
        serial_write(self.grbl, command)

    def reset_zero(self):
        command = "G10 P0 L20 X0 Y0 Z0\r"
        serial_write(self.grbl, command)

    # Function to send go-to command to self.GRBL device
    def go_to(self, x=None, y=None, z=None, relative=False):
        command = "G21G91" if relative else "G21G90"  # Absolute positioning or relative
        if x is not None:
            command += f" X{x}"
        if y is not None:
            command += f" Y{y}"
        if z is not None:
            command += f" Z{z}"
        if x is not None or y is not None:
            serial_write(self.grbl, "G90 Z0")
            self.wait_till_ready()
        serial_write(self.grbl, f"{command}F6000")

    def get_position(self):
        status = self.check_status(self.grbl)
        while "MPos" not in status:
            time.sleep(GRBL_POLL_TIME)
            status = self.check_status(self.grbl)
        position = status.split("MPos:")[1].split("|")[0].strip()
        x, y, z = [float(val) for val in position.split(",")]
        return x, y, z

    def home(self):
        # Home z
        status = self.check_status(self.grbl)
        while (
            "Pn:" not in status or "Z" not in status[1:-1].split("Pn:")[1].split("|")[0]
        ):
            self.step(2, "Z", "+")
            self.wait_till_ready()
            status = self.check_status(self.grbl)
        self.step(Z_STEP_OFF, "Z", "-")
        # Home x
        status = self.check_status(self.grbl)
        while (
            "Pn:" not in status or "X" not in status[1:-1].split("Pn:")[1].split("|")[0]
        ):
            self.step(2, "X", "+")
            self.wait_till_ready()
            status = self.check_status(self.grbl)
        self.step(4, "X", "-")
        # Home y
        status = self.check_status(self.grbl)
        while (
            "Pn:" not in status or "Y" not in status[1:-1].split("Pn:")[1].split("|")[0]
        ):
            self.step(2, "Y", "-")
            self.wait_till_ready()
            status = self.check_status(self.grbl)
        self.step(4, "Y", "+")
        self.reset_zero()

    def calibrate_vacuum(self):
        serial_write(self.arduino, "CALIBRATE_VACUUM")
        status = ""
        while "VACUUM_CALIBRATED" not in status:
            status += self.arduino.readline().decode()

    def vacuum_on(self):
        serial_write(self.arduino, "VACUUM_ON")
        status = ""
        while "VACUUM_ON" not in status:
            status += self.arduino.readline().decode()

    def vacuum_off(self):
        serial_write(self.arduino, "VACUUM_OFF")
        status = ""
        while "VACUUM_OFF" not in status:
            status += self.arduino.readline().decode()

    def pickup(self):
        # Home x
        print("vac on")
        self.vacuum_on()
        self.wait_till_ready()
        print("ready")
        self.step(60, "Z", "-")
        self.wait_till_ready()
        print("z moved, ready")
        while not self.card_on():
            print("card not on")
            self.step(10, "Z", "-")
        print("card on")
        self.go_to(z=22)
        self.wait_till_ready()
        self.go_to(z=0)

    def release(self):
        self.vacuum_off()
        serial_write(self.arduino, "RELEASE")
        status = ""
        while "RELEASED" not in status:
            status += self.arduino.readline().decode()

    def card_on(self):
        subarray = []
        serial_write(self.arduino, "CURRENT")
        while True:
            line = self.arduino.readline().decode()
            if "FIN" in line.strip():
                if len(subarray) == len(psvm.X_train[0]):
                    res = psvm.clf.predict([subarray])
                    return 1 in res
                print(subarray)
                subarray = []
            else:
                try:
                    subarray.append(int(line.split(" ")[0].strip()))
                except Exception:
                    pass

    def move_card(self, stack_id_1, stack_id_2):
        self.align_stack(stack_id_1)
        self.wait_till_ready()
        self.pickup()
        self.wait_till_ready()
        self.align_stack(stack_id_2)
        self.wait_till_ready()
        self.go_to(z=-60, relative=True)
        time.sleep(1000 / 1000)
        self.wait_till_ready()
        self.release()
        self.go_to(z=0)
        self.wait_till_ready()

    def align_stack(self, stack_id):
        initial_jog, aruco_target, aruco_id = self.calibrations[stack_id]
        self.go_to(x=initial_jog[0], y=initial_jog[1])
        self.wait_till_ready()
        bboxes, frame = self.aruco_locator.get_aruco_bboxes(aruco_id=aruco_id)
        if aruco_id in bboxes.keys():
            print("boxes", bboxes)
            print("aruco_target", aruco_target)
            x_dist, y_dist = aruco.distance_from_target(bboxes[aruco_id], aruco_target)
            # modify to stick with closest tag
            thresh = 3
            while abs(x_dist + y_dist) > thresh * 2:
                print(x_dist, y_dist)
                self.go_to(x=x_dist, y=y_dist, relative=True)
                self.wait_till_ready()
                bboxes, frame = self.aruco_locator.get_aruco_bboxes(aruco_id=aruco_id)
                x_dist, y_dist = aruco.distance_from_target(
                    bboxes[aruco_id], aruco_target
                )
        self.wait_till_ready()

    def read_card(self, stack_id):
        _, _, aruco_id = self.calibrations[stack_id]
        bboxes, frame = self.aruco_locator.get_aruco_bboxes(aruco_id=aruco_id)
        if aruco_id in bboxes.keys():
            box = bboxes[aruco_id]
            frame = aruco.crop_based_on_aruco(box, frame)
        return detect_and_guess(frame)

    def display_read_card(self, stack=(0, 0), go_to=True, offset=-50):
        if go_to:
            self.align_stack(stack)
            self.go_to(y=offset, relative=True)
        self.wait_till_ready()
        frame, name = self.read_card(stack)
        print("GUESSED NAME", name)
        cv2.imshow("Aruco Markers", frame)
        # Break the loop if the 'q' key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if go_to:
            self.go_to(y=-1 * offset, relative=True)


if __name__ == "__main__":
    arduino = None
    grbl = None
    serials = glob.glob("/dev/tty.usbserial-*")
    if not serials:
        raise Exception("No serial connections found")
    # Query each serial connection
    for s in serials:
        try:
            ser = serial.Serial(s, 115200)
            wait_till_on(ser)
            while True:
                response = check_status(ser)
                if "MPos:" in response:
                    grbl = ser
                    break
                elif "ARDUINO" in response:
                    arduino = ser
                    break
        except Exception as e:
            print("nothing", s, e)
            pass
    # Raise an error if both grbl and arduino are not found
    if grbl is None or arduino is None:
        print(grbl, arduino)
        raise Exception("Unable to find both grbl and arduino")

    else:
        print("grbl connection: ", grbl)
        print("arduino connection: ", arduino)
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    robot = GRBL_Controller(grbl, arduino, cap)

    print("Initializing")
    robot.init()
    # print("Moving Card")
    # robot.move_card((1, 1), (1, 2))
    # time.sleep(2)
    # robot.move_card((1, 2), (1, 1))

    print("Enter 'c' to return robot to home and exit...")
    breakpoint()
    robot.deinit()
