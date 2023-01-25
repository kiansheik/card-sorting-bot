import cv2
import numpy as np

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    # rotation_matrix = cv2.getRotationMatrix2D((frame.shape[0]/2, frame.shape[1]/2), 180, 1)
    # frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
    frame = cv2.flip(frame, -1)  # flip the image vertically
    frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1
    )
    adap = cv2.threshold(adap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
        1
    ]  # thresholding
    corners = cv2.goodFeaturesToTrack(adap, 100, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(adap, (x, y), 5, (255, 0, 0), -1)

    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            corner1 = tuple(corners[i][0])
            corner2 = tuple(corners[j][0])
            color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
            cv2.line(adap, corner1, corner2, color, 1)

    # Show the adap with the bounding boxes and text content
    cv2.imshow("QR Code", adap)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("p"):
        print("test")

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
