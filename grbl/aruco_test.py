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
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                
                #Calculate the center of the bounding box
                center_x = sum([x[0] for x in box])//len(box)
                center_y = sum([y[1] for y in box])//len(box)
                center = (center_x, center_y)
                print(f"{center}, {ids[i][0]}")
                # Write the ID of the Aruco marker in the center of the bounding box
                cv2.putText(frame, str(ids[i][0]), center, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # corner = leftmost_highest(corners, frame)
        # rect = cv2.minAreaRect(corner)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # frame = crop_based_on_aruco(box, frame)

        cv2.imshow("Aruco Markers", frame)
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


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


def crop_based_on_aruco(bbox, img):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
    center_x = int((x1 + x2 + x3 + x4) / 4)
    center_y = int((y1 + y2 + y3 + y4) / 4)

    # Draw horizontal line
    cv2.line(img, (0, center_y), (img.shape[1], center_y), (0, 255, 0), 2)

    # Draw vertical line
    cv2.line(img, (center_x, 0), (center_x, img.shape[0]), (0, 255, 0), 2)

    return img[center_y:, center_x:]


if __name__ == "__main__":
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    get_aruco_bbox(cap, 0)
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
