import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the set symbol images and labels
symbols = []
labels = []

# Load each set symbol image and label
for symbol in ["set1", "set2", "set3"]:
    for i in range(1, 101):
        image = cv2.imread(f"{symbol}/{i}.jpg")
        symbols.append(image)
        labels.append(symbol)

# Preprocess the set symbol images
symbols = [cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY) for symbol in symbols]
symbols = [cv2.resize(symbol, (100, 100)) for symbol in symbols]
symbols = np.array(symbols)
symbols = symbols.reshape(len(symbols), -1)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(symbols, labels, test_size=0.2)

# Train a support vector machine classifier on the training set
classifier = SVC()
classifier.fit(X_train, y_train)

# Test the model on the test set
y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
