import glob
import time
from random import uniform

import aruco
import cv2
import numpy as np
import pressure_svm as psvm
import serial
from mtg_card_detector import detect_and_guess

GRBL_POLL_TIME = 0.25
Z_STEP_OFF = 20
SUCTION_CAMERA_POS = (536, 1123)
STACK_ALIGN_THRESH = 1


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
            (3, 0): [(-8.987, 81.909), [6, 16]],
            (2, 0): [(-89.949, 82.903), [3, 6]],
            (1, 0): [(-162.876, 83.897), [4, 3]],
            (0, 0): [(-235.802, 86.891), [41, 4]],
            # (1, 0): [(-52.959, 181.817), (178, 662), 39],
            # (1, 1): [(-143.873, 183.817), (194, 677), 2],
            # (1, 2): [(-227.497, 186.824), (136, 704), 42],
            # (1, 3): [(-306.427, 181.829), (907, 646), 42],
        }
        self.aruco_locator = aruco.ArucoLocator(
            cap, "calibration_images/calibration.json"
        )
        self.aruco_matrix = aruco.ArucoMatrix()

    def init(self):
        self.wait_till_ready()
        print("TEST RELEASE...")
        self.release()
        print("HOME...")
        self.home()
        print("CALIBRATE_VACUUM...")
        self.calibrate_vacuum()
        self.wait_till_ready()
        print("Aligning with first aruco tags...")
        self.align_stack((0, 0))
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

    def stack_camera_alignment(self):
        for stack_id in self.calibrations.keys():
            print(f"Aligning stack {stack_id}...")
            self.align_stack(stack_id)
            self.wait_till_ready()
            time.sleep(1)
        print("Done with alignment")

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
            self.step(6, "Z", "+")
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
        self.step(20, "Z", "-")
        self.wait_till_ready()
        print("z moved, ready")
        while not self.card_on():
            print("card not on")
            self.step(6, "Z", "-")
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
        self.release()
        self.go_to(z=0)
        self.wait_till_ready()

    def align_stack(self, stack_id, try_limit=30):
        initial_jog, aruco_ids = self.calibrations[stack_id]
        self.go_to(x=initial_jog[0], y=initial_jog[1])
        self.wait_till_ready()
        bboxes, frame = self.aruco_locator.get_aruco_bboxes(aruco_ids=aruco_ids)
        i = 0
        while i < try_limit and not set(aruco_ids).issubset(set(bboxes.keys())):
            bboxes, frame = self.aruco_locator.get_aruco_bboxes(aruco_ids=aruco_ids)
            i += 1
        if set(aruco_ids).issubset(set(bboxes.keys())):
            print("boxes", bboxes)
            box1 = bboxes[aruco_ids[0]]
            box2 = bboxes[aruco_ids[1]]
            center1 = (
                sum([x[0] for x in box1]) // len(box1),
                sum([y[1] for y in box1]) // len(box1),
            )
            center2 = (
                sum([x[0] for x in box2]) // len(box2),
                sum([y[1] for y in box2]) // len(box2),
            )
            aruco_target = aruco.equilateral_triangle_point(center1, center2)
            x_dist, y_dist = aruco.get_real_distance(
                aruco_target, SUCTION_CAMERA_POS, box1
            )
            while abs(x_dist + y_dist) > STACK_ALIGN_THRESH:
                print(x_dist, y_dist)
                self.go_to(x=x_dist * 0.75, y=y_dist * 0.75, relative=True)
                self.wait_till_ready()
                bboxes, frame = self.aruco_locator.get_aruco_bboxes(aruco_ids=aruco_ids)
                cv2.imshow("Aruco Markers", frame)
                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                i = 0
                while not set(aruco_ids).issubset(set(bboxes.keys())):
                    if i > try_limit:
                        self.wait_till_ready()
                        return
                    bboxes, _ = self.aruco_locator.get_aruco_bboxes(aruco_ids=aruco_ids)
                    i += 1
                box1 = bboxes[aruco_ids[0]]
                box2 = bboxes[aruco_ids[1]]
                center1 = (
                    sum([x[0] for x in box1]) // len(box1),
                    sum([y[1] for y in box1]) // len(box1),
                )
                center2 = (
                    sum([x[0] for x in box2]) // len(box2),
                    sum([y[1] for y in box2]) // len(box2),
                )
                aruco_target = aruco.equilateral_triangle_point(center1, center2)
                x_dist, y_dist = aruco.get_real_distance(
                    aruco_target, SUCTION_CAMERA_POS, box1
                )
        self.wait_till_ready()
        time.sleep(0.5)
        x, y, _ = self.get_position()
        # Save current position for faster recall than saved calibration
        self.calibrations[stack_id][0] = (x, y)

    # Running this assuming we are already over a stack, must move to stack in init
    def align_stack_aruco(self, stack_id, try_limit=30):
        aruco_ids = self.calibrations[stack_id][-1]
        self.wait_till_ready()
        bboxes, frame = self.aruco_locator.get_aruco_bboxes()
        for i in range(try_limit):
            if len(bboxes) > 2:
                break
            bboxes, frame = self.aruco_locator.get_aruco_bboxes()
        _, leftm_id = aruco.leftmost_highest(bboxes, frame)
        x_dist, y_dist = self.aruco_matrix.dist_between_tags(leftm_id, aruco_ids[0])
        self.go_to(x=x_dist * 0.999, y=y_dist * 0.999, relative=True)
        self.wait_till_ready()
        if set(aruco_ids).issubset(set(bboxes.keys())):
            print("boxes", bboxes)
            box1 = bboxes[aruco_ids[0]]
            box2 = bboxes[aruco_ids[1]]
            center1 = (
                sum([x[0] for x in box1]) // len(box1),
                sum([y[1] for y in box1]) // len(box1),
            )
            center2 = (
                sum([x[0] for x in box2]) // len(box2),
                sum([y[1] for y in box2]) // len(box2),
            )
            aruco_target = aruco.equilateral_triangle_point(center1, center2)
            x_dist, y_dist = aruco.get_real_distance(
                aruco_target, SUCTION_CAMERA_POS, box1
            )
            while abs(x_dist + y_dist) > STACK_ALIGN_THRESH:
                print(x_dist, y_dist)
                self.go_to(x=x_dist * 0.75, y=y_dist * 0.75, relative=True)
                self.wait_till_ready()
                bboxes, frame = self.aruco_locator.get_aruco_bboxes(aruco_ids=aruco_ids)
                cv2.imshow("Aruco Markers", frame)
                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                i = 0
                while not set(aruco_ids).issubset(set(bboxes.keys())):
                    if i > try_limit:
                        self.wait_till_ready()
                        return
                    bboxes, _ = self.aruco_locator.get_aruco_bboxes(aruco_ids=aruco_ids)
                    i += 1
                box1 = bboxes[aruco_ids[0]]
                box2 = bboxes[aruco_ids[1]]
                center1 = (
                    sum([x[0] for x in box1]) // len(box1),
                    sum([y[1] for y in box1]) // len(box1),
                )
                center2 = (
                    sum([x[0] for x in box2]) // len(box2),
                    sum([y[1] for y in box2]) // len(box2),
                )
                aruco_target = aruco.equilateral_triangle_point(center1, center2)
                x_dist, y_dist = aruco.get_real_distance(
                    aruco_target, SUCTION_CAMERA_POS, box1
                )
        self.wait_till_ready()
        # TODO: Save drift or new location in relation to each other in MM over the course of the sort to account for drift if needed, not sure if necessary

    def read_card(self, stack_id):
        card_name = None
        _, _, aruco_id = self.calibrations[stack_id]
        bboxes, frame = self.aruco_locator.get_aruco_bboxes(aruco_id=aruco_id)
        # Get the image dimensions
        height, width = frame.shape[:2]

        # Cut out the bottom 1/3 of the image
        # frame = frame[: int(height * (5 / 11)), :]
        if aruco_id in bboxes.keys():
            box = bboxes[aruco_id]
            frame = aruco.crop_based_on_aruco(box, frame)
        res_frame, card_name = detect_and_guess(np.copy(frame))
        if card_name is None:
            # frame = cv2.transpose(frame)
            # Flip the image
            frame = cv2.flip(frame, -1)
            res_frame, card_name = detect_and_guess(np.copy(frame))
        return res_frame, card_name

    def display_read_card(self, stack=(0, 0), go_to=True, offset=-70):
        if go_to:
            self.align_stack(stack)
            self.go_to(y=offset, relative=True)
        self.wait_till_ready()
        frame, name = self.read_card(stack)
        neg = -1
        if name is None:
            dist = 3
            self.go_to(y=uniform(neg * 0.1, neg * dist), relative=True)
            self.wait_till_ready()
            neg *= -1
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
    # robot.display_read_card()

    # print("Enter 'c' to return robot to home and exit...")
    # breakpoint()
    # robot.deinit()
