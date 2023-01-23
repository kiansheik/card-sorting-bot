import serial
import time
import cv2
from pyzbar import pyzbar
import glob

grbl = None
arduino = None


def wait_till_ready():
    while True:
        grbl.write(b"?\r")
        status = grbl.readline().decode()
        if "Idle" in status:
            time.sleep(0.025)
            break


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
    command = f"G21G91{axis}{direction if direction == '-' else ''}{steps-0.001}F4000\r"
    grbl.write(command.encode())


def reset_zero():
    command = "G10 P0 L20 X0 Y0 Z0\r"
    grbl.write(command.encode())


# Function to send go-to command to GRBL device
def go_to(x=None, y=None, z=None):
    command = "G90"  # Absolute positioning
    if x is not None:
        command += f" X{x}"
    if y is not None:
        command += f" Y{y}"
    if z is not None:
        command += f" Z{z}"
    if x is not None or y is not None:
        grbl.write("G90 Z0\r".encode())
        wait_till_ready()
    grbl.write(f"{command}\r".encode())


# Function to parse GRBL status and check for alarms
def check_status(ser):
    ser.flushInput()
    ser.flushOutput()
    ser.write(b"?\r")
    time.sleep(0.1)
    status = ser.readline().decode()
    if ser == grbl:
        while "MPos" not in status:
            status = grbl.readline().decode()
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
            adap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 171, 2
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


calibrations = {
    (0, 0): [(17.967, 40.928), (959, 455, 319, 322)],
    (0, 1): [(22.962, 126.853), (969, 560, 323, 309)],
    (0, 2): [(27.957, 216.777), (965, 621, 338, 324)],
    (0, 3): [(27.957, 294.666), (500, 284, 567, 582)],
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
    loc, bbox = calibrations[stack_id]
    center = find_center(*bbox)
    print(f"Going to {loc}...")
    go_to(x=loc[0], y=loc[1])
    wait_till_ready()
    done = set()
    print("Reading QR code...")
    while len(done) < 2:
        (curr_bbox), im_dim = get_qr_bbox(cap)
        curr_center = find_center(*curr_bbox)
        for i, axis in enumerate(["Y", "X"]):
            diff = curr_center[i] - center[i]
            if abs(diff) > thresh:
                wait_till_ready()
                step_val = map_function(abs(diff), 0, im_dim[i], 0.5, 5)
                step(step_val, axis, "-" if diff > 0 else "+")
            else:
                done.add(axis)
    print(f"{stack_id} has been visually aligned")
    curr_pos = get_position()
    # Uodate in memory location so that we can save time next time we go to this place
    calibrations[stack_id] = [curr_pos[:2], bbox]


def release():
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
        step(3, "Z", "+")
        wait_till_ready()
        status = check_status(grbl)
    step(34, "Z", "-")
    # Home x
    status = check_status(grbl)
    while "Pn:" not in status or "X" not in status[1:-1].split("Pn:")[1].split("|")[0]:
        step(1, "X", "-")
        wait_till_ready()
        status = check_status(grbl)
    step(4, "X", "+")
    # Home y
    status = check_status(grbl)
    while "Pn:" not in status or "Y" not in status[1:-1].split("Pn:")[1].split("|")[0]:
        step(1, "Y", "-")
        wait_till_ready()
        status = check_status(grbl)
    step(4, "Y", "+")
    reset_zero()


def vacuum_on():
    vacuum_on_cmd = b"G1 F4000 M03 S280\r"
    grbl.write(vacuum_on_cmd)
    wait_till_ready()


def vacuum_off():
    vacuum_off_cmd = b"M05\r"
    grbl.write(vacuum_off_cmd)
    wait_till_ready()


def pick_up():
    # Home x
    vacuum_on()
    wait_till_ready()
    step(40, "Z", "-")
    wait_till_ready()
    status = "NOTHING"
    while "Pn:" not in status or "Z" not in status[1:-1].split("Pn:")[1].split("|")[0]:
        step(5, "Z", "-")
        time.sleep(0.2)
        status = check_status(grbl)
        print(status)
    vacuum_off()
    go_to(z=0)


def move_card(stack_id_1, stack_id_2, cap):
    align_stack(stack_id_1, cap)
    wait_till_ready()
    pick_up()
    wait_till_ready()
    align_stack(stack_id_2, cap)
    wait_till_ready()
    step(40, "Z", "-")
    wait_till_ready()
    release()
    go_to(z=0)
    wait_till_ready()


if __name__ == "__main__":
    serials = glob.glob("/dev/tty.usbserial-*")
    if not serials:
        raise Exception("No serial connections found")
    # Query each serial connection
    for s in serials:
        try:
            ser = serial.Serial(s, 115200)
            time.sleep(2)
            wait_till_on(ser)
            response = check_status(ser)
            print(s, response)
            if "MPos:" in response:
                grbl = ser
            elif "ARDUINO" in response:
                arduino = ser
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
    home()
    wait_till_ready()

    move_card((0, 0), (0, 1), cap)
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
    home()
