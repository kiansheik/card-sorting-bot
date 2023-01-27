import glob
import json
import time

import cv2
import numpy as np
import serial
from pyzbar import pyzbar
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

SAMPLE_WINDOW_SIZE = 100
Z_STEP_OFF = 20
grbl = None
arduino = None

with open("calibration_images/calibration.json") as f:
    calib = json.load(f)
    camera_matrix = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["dist_coeff"])


def wait_till_ready():
    ser.flushInput()
    ser.flushOutput()
    while True:
        grbl.write(b"?\r")
        time.sleep(0.02)
        status = grbl.readline().decode()
        if "Run" in status:
            pass
        elif "Idle" in status:
            break
        time.sleep(0.25)  # GRBL wiki recommends no more than 5 times per second polling


def wait_till_on(ser):
    time.sleep(0.5)
    ser.flushInput()
    ser.flushOutput()
    while True:
        status = check_status(ser)
        if len(status) > 4:
            time.sleep(0.025)
            break


# Function to send step command to GRBL device
def step(steps, axis, direction):
    if direction not in ["+", "-"]:
        raise ValueError("Invalid direction")
    command = f"G21G91{axis}{direction if direction == '-' else ''}{steps-0.001}F6000\r"
    grbl.write(command.encode())


def reset_zero():
    command = "G10 P0 L20 X0 Y0 Z0\r"
    grbl.write(command.encode())


# Function to send go-to command to GRBL device
def go_to(x=None, y=None, z=None, relative=False):
    command = "G21G91" if relative else "G21G90"  # Absolute positioning or relative
    if x is not None:
        command += f" X{x}"
    if y is not None:
        command += f" Y{y}"
    if z is not None:
        command += f" Z{z}"
    if x is not None or y is not None:
        grbl.write("G90 Z0\r".encode())
        wait_till_ready()
    grbl.write(f"{command}F6000\r".encode())


# Function to parse GRBL status and check for alarms
def check_status(ser):
    ser.flushInput()
    ser.flushOutput()
    time.sleep(0.1)
    ser.write(b"?\r")
    time.sleep(0.1)
    status = ser.read_all().decode()
    if ser == grbl:
        while "MPos" not in status:
            status = grbl.readline().decode()
    if ser == arduino:
        while "ARDUINO" not in status:
            status = arduino.readline().decode()
    return status


def find_center(x, y, w, h):
    center_x = x + (w / 2)
    center_y = y + (h / 2)
    return center_x, center_y


def get_qr_bbox(cap):
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, -1)  # flip the image vertically
        adap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        adap = cv2.adaptiveThreshold(
            adap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1
        )
        adap = cv2.threshold(adap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
            1
        ]  # thresholding
        qr_codes = pyzbar.decode(adap)
        cv2.imshow("QR Code", adap)
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # Iterate through the detected QR codes
        for qr_code in qr_codes:
            # Get the bounding box coordinates of the QR code
            # cv2.putText(adap, 1, tuple(find_center(959, 455, 319, 322)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return (qr_code.rect), (adap.shape)


def average_side_length(bounding_box):
    # Get the x,y coordinates of the bounding box
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bounding_box
    # Get the length of each side
    top = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    right = np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3]))
    bottom = np.linalg.norm(np.array([x3, y3]) - np.array([x4, y4]))
    left = np.linalg.norm(np.array([x4, y4]) - np.array([x1, y1]))
    # Average of all sides
    average = (top + right + bottom + left) / 4
    return average


def distance_from_target(bounding_box, target_point, real_tag_size=6.3):
    # Get the center of the bounding box
    center = (
        sum([x[0] for x in bounding_box]) // len(bounding_box),
        sum([y[1] for y in bounding_box]) // len(bounding_box),
    )

    # Calculate the x and y distances between the center of the bounding box and the target point in pixels
    x_distance = center[0] - target_point[0]
    y_distance = center[1] - target_point[1]

    # Convert the distance in pixels to millimeters using the real tag size
    x_distance_mm = x_distance * real_tag_size / average_side_length(bounding_box)
    y_distance_mm = y_distance * real_tag_size / average_side_length(bounding_box)

    return x_distance_mm, -1 * y_distance_mm


def get_aruco_bboxes(cap, camera_matrix, dist_coeffs, aruco_id=None):
    boxes = dict()
    # while len(boxes) == 0 or (aruco_id is not None and aruco_id not in boxes.keys()):
    # Read a frame from the webcam
    ret, frame = cap.read()
    # images = glob.glob("sample_robot_images/*.jpg")
    # frame = cv2.imread(images[1])
    if not ret:
        return None
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    frame = cv2.flip(frame, 1)  # flip the image vertically
    frame = cv2.transpose(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

    # Detect aruco markers
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )
    if corners:
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.05, camera_matrix, dist_coeffs
        )
        for i in range(len(ids)):
            if aruco_id in ids[i]:
                corner = corners[i].reshape(-1, 2)
                rect = cv2.minAreaRect(corner)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                print(ids[i][0], box)
                boxes[ids[i][0]] = box
    return boxes, frame


def closest_tag(tags, target_point):
    closest_tag = None
    closest_distance = float("inf")
    for aruco_id, bounding_box in tags.items():
        # Get the center of the bounding box
        center = np.array(
            [
                (bounding_box[0] + bounding_box[2]) / 2,
                (bounding_box[1] + bounding_box[3]) / 2,
            ]
        )

        # Calculate the distance between the center of the bounding box and the target point in pixels
        distance = np.linalg.norm(center - target_point)

        # Update the closest tag and distance if necessary
        if distance < closest_distance:
            closest_distance = distance
            closest_tag = aruco_id

    return closest_tag


def target_closest_aruco(cap, camera_matrix, dist_coeffs, stack_id):
    initial_jog, aruco_target, aruco_id = calibrations[stack_id]
    go_to(x=initial_jog[0], y=initial_jog[1])
    wait_till_ready()
    time.sleep(0.5)
    bboxes, frame = get_aruco_bboxes(cap, camera_matrix, dist_coeffs, aruco_id=aruco_id)
    print(bboxes)
    if len(bboxes) > 0:
        print("boxes", bboxes)
        print("aruco_target", aruco_target)
        x_dist, y_dist = distance_from_target(bboxes[aruco_id], aruco_target)
        # modify to stick with closest tag
        thresh = 4
        while abs(x_dist + y_dist) > thresh * 2:
            print(x_dist, y_dist)
            # breakpoint()
            go_to(x=x_dist, y=y_dist, relative=True)
            # if abs(x_dist) > thresh:
            #     step(abs(x_dist), 'X', '+' if x_dist > 0 else '-')
            # if abs(y_dist) > thresh:
            #     step(abs(y_dist), 'y', '+' if y_dist > 0 else '-')
            wait_till_ready()
            bboxes, frame = get_aruco_bboxes(
                cap, camera_matrix, dist_coeffs, aruco_id=aruco_id
            )
            # print('boxes', bboxes)
            # print('aruco_target', aruco_target)
            x_dist, y_dist = distance_from_target(bboxes[aruco_id], aruco_target)


calibrations = {
    # Format: (stack_row, stack_col): [(stepper_mm_x, stepper_mm_y), (aruco_center_target_x, aruco_center_target_y), aruco_id]
    (0, 0): [(-53.975, 77.638), (179, 727), 0],
    (0, 1): [(-139.823, 71.882), (102, 646), 49],
    (0, 2): [(-226.401, 72.876), (96, 634), 48],
    (0, 3): [(-308.324, 83.872), (915, 683), 48],
    (1, 0): [(-52.959, 181.817), (178, 662), 39, 39],
    (1, 1): [(-143.873, 183.817), (194, 677), 2],
    (1, 2): [(-227.497, 186.824), (136, 704), 42],
    (1, 3): [(-306.427, 181.829), (907, 646), 42],
}


def map_function(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def get_position():
    grbl.flushInput()
    grbl.flushOutput()
    grbl.write(b"?\r")
    time.sleep(0.1)
    status = ""
    while "MPos" not in status:
        status += grbl.readline().decode()
    if "MPos" in status:
        position = status.split("MPos:")[1].split("|")[0].strip()
        x, y, z = [float(val) for val in position.split(",")]
        return x, y, z
    else:
        print("Cannot get current position.")


def align_stack(stack_id, cap, thresh=25):
    target_closest_aruco(cap, camera_matrix, dist_coeffs, stack_id)


def release():
    vacuum_off()
    arduino.flushInput()
    arduino.flushOutput()
    arduino.write(b"RELEASE\r")
    time.sleep(0.1)
    status = ""
    while "RELEASED" not in status:
        status += arduino.readline().decode()


def home():
    # Home z
    status = check_status(grbl)
    while "Pn:" not in status or "Z" not in status[1:-1].split("Pn:")[1].split("|")[0]:
        step(2, "Z", "+")
        wait_till_ready()
        status = check_status(grbl)
    step(Z_STEP_OFF, "Z", "-")
    # Home x
    status = check_status(grbl)
    while "Pn:" not in status or "X" not in status[1:-1].split("Pn:")[1].split("|")[0]:
        step(2, "X", "+")
        wait_till_ready()
        status = check_status(grbl)
    step(4, "X", "-")
    # Home y
    status = check_status(grbl)
    while "Pn:" not in status or "Y" not in status[1:-1].split("Pn:")[1].split("|")[0]:
        step(2, "Y", "-")
        wait_till_ready()
        status = check_status(grbl)
    step(4, "Y", "+")
    reset_zero()


def vacuum_on(speed=280):
    # vacuum_on_cmd = f"G1 F4000 M03 S{speed}\r".encode()
    # grbl.write(vacuum_on_cmd)
    # wait_till_ready()
    vacuum_on_cmd = f"VACUUM_ON\r".encode()
    arduino.write(vacuum_on_cmd)
    time.sleep(0.1)
    status = ""
    while "VACUUM_ON" not in status:
        status += arduino.readline().decode()


def vacuum_off():
    # vacuum_off_cmd = b"M05\r"
    # grbl.write(vacuum_off_cmd)
    # wait_till_ready()
    vacuum_off_cmd = f"VACUUM_OFF\r".encode()
    arduino.write(vacuum_off_cmd)
    time.sleep(0.1)
    status = ""
    while "VACUUM_OFF" not in status:
        status += arduino.readline().decode()


def calibrate_vacuum():
    vacuum_off_cmd = f"CALIBRATE_VACUUM\r".encode()
    arduino.write(vacuum_off_cmd)
    time.sleep(0.1)
    status = ""
    while "VACUUM_CALIBRATED" not in status:
        status += arduino.readline().decode()


def shake():
    vacuum_on(400)
    arduino.flushInput()
    arduino.flushOutput()
    arduino.write(b"SHAKE\r")
    time.sleep(0.1)
    status = ""
    while "SHOOK" not in status:
        status += arduino.readline().decode()
    vacuum_off()


def pick_up():
    # Home x
    vacuum_on()
    wait_till_ready()
    step(60, "Z", "-")
    wait_till_ready()
    time.sleep(0.5)
    while not card_on():
        # wait_till_ready()
        step(10, "Z", "-")
    go_to(z=18)
    wait_till_ready()
    go_to(z=0)
    # shake()
    # wait_till_ready()


def card_on():
    subarray = []
    # Throw away first array in case it's incomplete
    arduino.flushInput()
    arduino.flushOutput()
    time.sleep(0.1)
    arduino.write(b"CURRENT\r")
    time.sleep(0.1)
    while True:
        line = arduino.readline().decode()
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


def move_card(stack_id_1, stack_id_2, cap):
    align_stack(stack_id_1, cap)
    wait_till_ready()
    pick_up()
    wait_till_ready()
    align_stack(stack_id_2, cap)
    wait_till_ready()
    go_to(z=-60, relative=True)
    time.sleep(1000 / 1000)
    # step(60, "Z", "-")
    wait_till_ready()
    release()
    go_to(z=0)
    wait_till_ready()


# Collect sensor readings for each state
def store_integers_from_file(filepath):
    with open(filepath, "r") as file:
        subarrays = []
        subarray = []
        for line in file:
            if "FIN" in line.strip():
                if len(subarray) == SAMPLE_WINDOW_SIZE:
                    subarrays.append(subarray)
                subarray = []
            else:
                try:
                    subarray.append(int(line.split(" ")[0].strip()))
                except Exception:
                    pass
    return subarrays


# Calibrate Current sensor SVM for detecting cards with readings from these 3 states
vacuum_on_readings = store_integers_from_file("sig_dump_vacuum_on.txt")[1:-1]
vacuum_off_readings = store_integers_from_file("sig_dump_vacuum_off.txt")[1:-1]
vacuum_sucking_card_readings = store_integers_from_file("sig_dump_card_on.txt")[1:-1]

# Create labels for the sensor readings
vacuum_off_labels = [0] * len(vacuum_off_readings)
vacuum_on_labels = [0] * len(vacuum_on_readings)
vacuum_sucking_card_labels = [1] * len(vacuum_sucking_card_readings)

# Combine sensor readings and labels into a single dataset
X = []
for arr in (vacuum_off_readings, vacuum_on_readings, vacuum_sucking_card_readings):
    for sub in arr:
        X.append(sub)
y = np.concatenate((vacuum_off_labels, vacuum_on_labels, vacuum_sucking_card_labels))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a support vector machine (SVM) model on the training data
clf = SVC()
clf.fit(X_train, y_train)
# Evaluate the model on the test data
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Current Sensor SVM Accuracy:", accuracy)


if __name__ == "__main__":
    serials = glob.glob("/dev/tty.usbserial-*")
    if not serials:
        raise Exception("No serial connections found")
    # Query each serial connection
    for s in serials:
        try:
            ser = serial.Serial(s, 115200)
            wait_till_on(ser)
            print("ON ", s)
            while True:
                response = check_status(ser)
                if "MPos:" in response:
                    grbl = ser
                    break
                elif "ARDUINO" in response:
                    arduino = ser
                    break
        except Exception:
            print("nothing", s)
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
    wait_till_ready()
    release()
    home()
    calibrate_vacuum()
    # home()
    wait_till_ready()
    # target_closest_aruco(cap, camera_matrix, dist_coeffs, (0,0))
    # t1 = time.time()
    # pick_up()
    # wait_till_ready()
    # print(time.time() - t1)

    move_card((1, 1), (1, 2), cap)
    time.sleep(2)

    move_card((1, 2), (1, 1), cap)
    # pick_up()
    # breakpoint()
    # print(get_position())
    # stacks = []
    # for stack_id in calibrations.keys():
    #     stacks.append(stack_id)
    #     print("POS", get_position())
    #     # align_stack((0,0), cap)
    #     align_stack(stack_id, cap)
    #     time.sleep(0.5)
    # stacks.reverse()
    # for stack_id in stacks:
    #     print("POS", get_position())
    #     # align_stack((0,0), cap)
    #     align_stack(stack_id, cap)
    #     time.sleep(0.5)
    # print("Done aligning test")
    breakpoint()
    go_to(0, 0)
    # home()
