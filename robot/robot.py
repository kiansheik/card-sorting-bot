import glob
import time
import json
import cv2
import numpy as np
import serial
from pyzbar import pyzbar
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm


SAMPLE_WINDOW_SIZE = 100
GRBL_POLL_TIME = 0.25
Z_STEP_OFF = 20

calibrations = {
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

def serial_write(ser, msg):
    ser.flushInput()
    ser.flushOutput()
    ser.write(f"{msg}\r".encode())
    time.sleep(0.02)

class GRBL_Controller:
    def __init__(self):
        self.grbl = None
        self.arduino = None
        self.calibrations = {
            (0, 0): [(-53.975, 77.638), (179, 727), 0],
            (0, 1): [(-139.823, 71.882), (102, 646), 49],
            (0, 2): [(-226.401, 72.876), (96, 634), 48],
            (0, 3): [(-308.324, 83.872), (915, 683), 48],
            (1, 0): [(-52.959, 181.817), (178, 662), 39, 39],
            (1, 1): [(-143.873, 183.817), (194, 677), 2],
            (1, 2): [(-227.497, 186.824), (136, 704), 42],
            (1, 3): [(-306.427, 181.829), (907, 646), 42],
        }
    # Function to parse GRBL status and check for alarms
    def check_status(self, ser):
        serial_write(ser, "?")
        status = ser.read_line().decode()
        return status
    def wait_till_on(self, ser):
        time.sleep(0.5)
        while True:
            status = self.check_status(ser)
            if len(status) > 4:
                break
            time.sleep(GRBL_POLL_TIME)
    def wait_till_ready(self):
        while True:
            status = self.check_status(self.grbl)
            if "Run" in status:
                time.sleep(0.001)
            elif "Idle" in status:
                break
            time.sleep(GRBL_POLL_TIME) # self.GRBL wiki recommends no more than 5 times per second polling
    # Function to send step command to self.GRBL device
    def step(self, steps, axis, direction):
        if direction not in ["+", "-"]:
            raise ValueError("Invalid direction")
        command = f"G21G91{axis}{direction if direction == '-' else ''}{steps-0.001}F6000"
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
# def align_stack(stack_id, cap, thresh=25):
#     target_closest_aruco(cap, camera_matrix, dist_coeffs, stack_id)
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
    def pick_up(self):
        # Home x
        self.vacuum_on()
        self.wait_till_ready()
        self.step(60, "Z", "-")
        self.wait_till_ready()
        time.sleep(0.5)
        while not self.card_on():
            self.step(10, "Z", "-")
        self.go_to(z=18)
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
        serial_write(self.arduino, "CURRRENT()")
        while True:
            line = self.arduino.readline().decode()
            if "FIN" in line.strip():
                if len(subarray) == len(X_train[0]):
                    res = clf.predict([subarray])
                    return 1 in res
                print(subarray)
                subarray = []
            else:
                try:
                    subarray.append(int(line.split(" ")[0].strip()))
                except Exception:
                    pass

# TODO: add SVM, Aruco tags, and MaskRCNN to their own files and import them for integration