import glob
import time

import serial
from tqdm import tqdm


def wait_till_on(ser):
    time.sleep(0.5)
    ser.flushInput()
    ser.flushOutput()
    while True:
        status = check_status(ser)
        if len(status) > 5:
            time.sleep(0.025)
            break


# Function to parse GRBL status and check for alarms
def check_status(ser):
    ser.flushInput()
    ser.flushOutput()
    ser.write(b"?\r")
    time.sleep(0.1)
    status = ser.readline().decode()
    return status


arduino = serial.Serial("/dev/tty.usbserial-1420", 115200)
states = ("vacuum_off", "vacuum_on", "card_on")
tl = (100 * 10) * 30
for state in states:
    print(f"Set to state: {state} then press 'c'...")
    if "vacuum" in state:
        arduino.write(f"{state.upper()}\n".encode())
        time.sleep(1)
    breakpoint()
    text = ""
    for i in range(10000):
        arduino.read_all()
    for i in tqdm(range(tl)):
        if state != "card_on":
            if i % (500 * 5) == 0:
                arduino.write(b"VACUUM_OFF\r")
                time.sleep(0.2)
                ln = arduino.readline().decode()
            if i % (500 * 5) == 500 * 2:
                arduino.write(b"VACUUM_ON\r")
                time.sleep(0.2)
                ln = arduino.readline().decode()
        if i % (101 * 1) == 0:
            arduino.flushInput()
            arduino.flushOutput()
            arduino.write(b"CURRENT\r")
            time.sleep(0.2)
        ln = arduino.readline().decode()
        # print(ln)
        text += ln
    fname = f"sig_dump_{state}.txt"
    with open(fname, "w") as f:
        f.write(text)
    print(f"Wrote {tl} lines to {fname}")
