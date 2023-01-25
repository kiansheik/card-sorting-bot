import glob
import time

import numpy as np
import serial
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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


# Collect sensor readings for each state
def store_integers_from_file(filepath):
    with open(filepath, "r") as file:
        subarrays = []
        subarray = []
        for line in file:
            if line.strip() == "FIN":
                subarrays.append(subarray)
                subarray = []
            else:
                subarray.append(int(line.strip()))
    return subarrays


vacuum_on_readings = store_integers_from_file("sig_dump_vacuum_on.txt")[1:-1]
vacuum_off_readings = store_integers_from_file("sig_dump_vacuum_off.txt")[1:-1]
vacuum_sucking_card_readings = store_integers_from_file("sig_dump_card_on.txt")[1:-1]

# Create labels for the sensor readings
vacuum_off_labels = [0] * len(vacuum_off_readings)
vacuum_on_labels = [1] * len(vacuum_on_readings)
vacuum_sucking_card_labels = [2] * len(vacuum_sucking_card_readings)

# Combine sensor readings and labels into a single dataset
# breakpoint()
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
print("SVM Accuracy:", accuracy)

arduino = serial.Serial("/dev/tty.usbserial-1420", 115200)

subarray = []
while True:
    line = arduino.readline().decode()
    if line.strip() == "FIN":
        print(clf.predict([subarray]))
        subarray = []
    else:
        subarray.append(int(line.strip()))
