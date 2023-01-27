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


def crop_based_on_aruco(bbox, img):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
    center_x = int((x1 + x2 + x3 + x4) / 4)
    center_y = int((y1 + y2 + y3 + y4) / 4)

    # Draw horizontal line
    cv2.line(img, (0, center_y), (img.shape[1], center_y), (0, 255, 0), 2)

    # Draw vertical line
    cv2.line(img, (center_x, 0), (center_x, img.shape[0]), (0, 255, 0), 2)

    return img[center_y:, center_x:]


def leftmost_highest(bboxes, img):
    leftmost_x = img.shape[1]  # Initialize with the maximum width of the image
    highest_y = img.shape[0]  # Initialize with the maximum height of the image
    leftmost_highest_bbox = None
    for bbox in bboxes:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.reshape(-1, 2)
        x = min(x1, x2, x3, x4)
        y = min(y1, y2, y3, y4)
        if x < leftmost_x and y < highest_y:
            leftmost_x = x
            highest_y = y
            leftmost_highest_bbox = bbox
    return leftmost_highest_bbox


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


class ArucoLocator:
    def __init__(self, cap, calibration_json_file_path=None):
        self.cap = cap
        self.distortion = False
        if calibration_json_file_path:
            with open(calibration_json_file_path) as f:
                calib = json.load(f)
                self.camera_matrix = np.array(calib["camera_matrix"])
                self.dist_coeffs = np.array(calib["dist_coeff"])
            self.distortion = True

    def get_aruco_bboxes(self, aruco_id=None):
        boxes = dict()
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.distortion:
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            frame = cv2.flip(frame, 1)  # flip the image vertically
            frame = cv2.transpose(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        # Detect aruco markers
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )
        if corners:
            for i in range(len(ids)):
                corner = corners[i].reshape(-1, 2)
                rect = cv2.minAreaRect(corner)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if aruco_id is None or aruco_id in ids[i]:
                    boxes[ids[i][0]] = box
        return boxes, frame


# If ran directly, open main camera and locate aruco tags live, print bbox centers to terminal
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    al = ArucoLocator(cap, "calibration_images/calibration.json")
    while True:
        bboxes, frame = al.get_aruco_bboxes()
        for id, box in bboxes.items():
            center_x = sum([x[0] for x in box]) // len(box)
            center_y = sum([y[1] for y in box]) // len(box)
            center = (center_x, center_y)
            print(f"{center}, {id}")
            # Write the ID of the Aruco marker in the center of the bounding box
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            cv2.putText(
                frame,
                str(id),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("Aruco Markers", frame)
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
