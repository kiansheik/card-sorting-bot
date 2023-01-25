import json

import cv2
import numpy as np

with open("calibration_images/calibration.json") as f:
    calib = json.load(f)
    camera_matrix = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["dist_coeff"])


def get_aruco_bbox(cap, marker_id):
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        frame = cv2.flip(frame, 1)  # flip the image vertically
        frame = cv2.transpose(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        # gray = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1
        # )
        # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
        #     1
        # ]

        # Detect aruco markers
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )
        # Draw bounding boxes around detected markers
        if corners:
            for i in range(len(ids)):
                # if ids[i] == marker_id:
                corner = corners[i]
                x, y, w, h = cv2.boundingRect(corner)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                print(ids[i])
                # return (x, y, w, h), (frame.shape)
        cv2.imshow("Aruco Markers", frame)
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    get_aruco_bbox(cap, 0)
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
