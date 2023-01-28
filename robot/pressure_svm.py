import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

SAMPLE_WINDOW_SIZE = 100


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


if __name__ == "__main__":
    # Evaluate the model on the test data
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Current Sensor SVM Accuracy:", accuracy)
