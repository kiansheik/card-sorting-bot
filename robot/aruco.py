import json

import cv2
import numpy as np

COLUMN_WIDTH = 71
ROW_WIDTH = 96


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
    leftmost_highest_id = None
    for aruco_id, bbox in bboxes.items():
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.reshape(-1, 2)
        x = min(x1, x2, x3, x4)
        y = min(y1, y2, y3, y4)
        if x < leftmost_x and y < highest_y:
            leftmost_x = x
            highest_y = y
            leftmost_highest_bbox = bbox
            leftmost_highest_id = aruco_id
    return leftmost_highest_bbox, leftmost_highest_id


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


def equilateral_triangle_point(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # calculate slope of the line between the two points
    if x1 == x2:
        m = None
    else:
        m = (y2 - y1) / (x2 - x1)

    # calculate slope of the perpendicular line
    if m is None:
        m_perp = 0
    else:
        m_perp = -1 / m

    # calculate midpoint of the line between the two points
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    dist = np.linalg.norm(np.array(point2) - np.array(point1)) * 0.3
    # calculate the point that is a distance of sqrt(3) away from the midpoint
    # along the perpendicular line
    print("mperp", m_perp)
    if m_perp == 0 or abs(m_perp) == float("inf"):
        # print('slope 0')
        x3 = midpoint[0]
        y3 = midpoint[1] + dist
    # elif m_perp == float('inf'):
    #     print('slope inf')
    #     x3 = midpoint[0] + dist
    #     y3 = midpoint[1]
    else:
        # print('slope normal')
        x3 = midpoint[0] + dist / np.sqrt(1 + m_perp**2)
        y3 = midpoint[1] + abs(m_perp * (x3 - midpoint[0]))

    return (int(x3), int(y3))


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


def get_real_distance(start_point, target_point, bounding_box, real_tag_size=6.3):
    # Calculate the x and y distances between the target_point of the bounding box and the target point in pixels
    x_distance = target_point[0] - start_point[0]
    y_distance = target_point[1] - start_point[1]

    # Convert the distance in pixels to millimeters using the real tag size
    x_distance_mm = x_distance * real_tag_size / average_side_length(bounding_box)
    y_distance_mm = y_distance * real_tag_size / average_side_length(bounding_box)

    return -1 * x_distance_mm, y_distance_mm


class ArucoLocator:
    def __init__(self, cap, calibration_json_file_path=None):
        self.cap = cap
        self.undistort = False
        if calibration_json_file_path:
            with open(calibration_json_file_path) as f:
                calib = json.load(f)
                self.camera_matrix = np.array(calib["camera_matrix"])
                self.dist_coeffs = np.array(calib["dist_coeff"])
            self.undistort = True

    def get_aruco_bboxes(self, aruco_ids=None):
        boxes = dict()
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.undistort:
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
                if aruco_ids is None or ids[i][0] in aruco_ids:
                    boxes[ids[i][0]] = box
        return boxes, frame


class ArucoMatrix:
    # (3,3) (2,3) (1,3) (0,3)
    # (3,2) (2,2) (1,2) (0,2)
    # (3,1) (2,1) (1,1) (0,1)
    # (3,0) (2,0) (1,0) (0,0)
    def __init__(self, column_width=COLUMN_WIDTH, row_width=ROW_WIDTH):
        self.upper_left_box_loc = {
            y[-1][0]: (x[0] * column_width, x[1] * row_width)
            for x, y in self.calibrations.items()
        }

    def dist_between_tags(self, start_id, target_id):
        start_point = self.upper_left_box_loc[start_id]
        target_point = self.upper_left_box_loc[target_id]
        # Calculate the x and y distances between the center of the bounding box and the target point in pixels
        x_distance_mm = start_point[0] - target_point[0]
        y_distance_mm = start_point[1] - target_point[1]

        return -1 * x_distance_mm, y_distance_mm


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
        if len(bboxes) == 2:
            vals = list(bboxes.values())
            center_1 = (
                sum([x[0] for x in vals[0]]) // len(vals[0]),
                sum([y[1] for y in vals[0]]) // len(vals[0]),
            )
            center_2 = (
                sum([x[0] for x in vals[1]]) // len(vals[1]),
                sum([y[1] for y in vals[1]]) // len(vals[1]),
            )

            center_card = equilateral_triangle_point(center_1, center_2)
            print("target", center_card)
            cv2.putText(
                frame,
                str("target"),
                center_card,
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
