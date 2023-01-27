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
    while True:
        status = check_status(grbl)
        if "Run" in status:
            time.sleep(0.001)
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
            break
        time.sleep(0.25)


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


def get_position():
    status = check_status(grbl)
    while "MPos" not in status:
        time.sleep(0.2)
        status = check_status(grbl)
    position = status.split("MPos:")[1].split("|")[0].strip()
    x, y, z = [float(val) for val in position.split(",")]
    return x, y, z


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
