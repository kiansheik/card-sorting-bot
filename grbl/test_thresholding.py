import glob
import json

import cv2
import numpy as np
from tqdm import tqdm

with open("calibration_images/calibration.json") as f:
    calib = json.load(f)
    camera_matrix = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["dist_coeff"])


images = glob.glob("sample_robot_images/*.jpg")

for fname in tqdm(images):
    img = cv2.imread(fname)
    img = cv2.undistort(img, camera_matrix, dist_coeffs)
    img = cv2.flip(img, 1)  # flip the image vertically
    img = cv2.transpose(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect aruco markers
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )
    # Draw bounding boxes around detected markers
    if corners:
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.05, camera_matrix, dist_coeffs
        )
        for i in range(len(ids)):
            corner = corners[i].reshape(-1, 2)
            rect = cv2.minAreaRect(corner)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            print(ids[i])
    cv2.imshow("img", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
