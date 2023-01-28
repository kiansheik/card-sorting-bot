import json
import os

import cv2
import fuzzy_search
import mrcnn.model as modellib
import numpy as np
import tensorflow.compat.v1 as tf
from mrcnn.config import Config
from paddleocr import PaddleOCR

tf.config.set_visible_devices([], "GPU")
ocr = PaddleOCR(lang="en")
OBJ_CLASSES = ["card", "name", "set_symbol"]
BASE_SIZE = (1024, 1024)
ROOT_DIR = os.path.abspath("../")


def get_file_paths(dir_path):
    file_paths = []
    for root, directories, files in os.walk(dir_path):
        for file in files:
            if not file.startswith("."):
                file_path = os.path.join(root, file)
                file_path = os.path.abspath(file_path)
                file_paths.append(file_path)
    return file_paths


class MTGCardInferConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "mtg_card"
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = len(OBJ_CLASSES) + 1  # Background + mtg_card classes
    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7
    USE_MINI_MASK = False
    IMAGE_MIN_DIM = BASE_SIZE[0]
    IMAGE_MAX_DIM = BASE_SIZE[1]
    BACKBONE = "resnet101"


logs_dir = os.path.join(ROOT_DIR, "vision", "models")
config = MTGCardInferConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs_dir)
weights_path = model.find_last()
print(weights_path)
tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True)


def reverse_resize_and_pad(padded_image, original_size):
    # Get the current height and width of the padded image
    height, width = padded_image.shape[:2]
    # Get the current height and width of the image
    rows, cols = original_size[:2]

    # Calculate the scaling factor for the height and width
    height_scale = height / rows
    width_scale = width / cols
    scale = min(height_scale, width_scale)

    # Calculate the new height and width of the image
    new_height = int(rows * scale)
    new_width = int(cols * scale)

    # Calculate the padding needed
    top_pad = 0
    bottom_pad = 0
    left_pad = 0
    right_pad = 0
    if new_height < height:
        top_pad = (height - new_height) // 2
        bottom_pad = height - new_height - top_pad
    if new_width < width:
        left_pad = (width - new_width) // 2
        right_pad = width - new_width - left_pad

    # Crop the image to remove the padding
    cropped_image = padded_image[
        top_pad : height - bottom_pad, left_pad : width - right_pad
    ]
    # Resize the image
    resized_image = cv2.resize(cropped_image, (cols, rows))

    return resized_image


def resize_and_pad(image, target_shape):
    height, width = target_shape
    # Get the current height and width of the image
    rows, cols, channels = image.shape

    # Calculate the scaling factor for the height and width
    height_scale = height / rows
    width_scale = width / cols
    scale = min(height_scale, width_scale)

    # Calculate the new height and width of the image
    new_height = int(rows * scale)
    new_width = int(cols * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate the padding needed
    top_pad = 0
    bottom_pad = 0
    left_pad = 0
    right_pad = 0
    if new_height < height:
        top_pad = (height - new_height) // 2
        bottom_pad = height - new_height - top_pad
    if new_width < width:
        left_pad = (width - new_width) // 2
        right_pad = width - new_width - left_pad

    # Pad the image
    padded_image = cv2.copyMakeBorder(
        resized_image,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    return padded_image


def bitmask_to_bounding_box(bitmask):
    # Find the non-zero elements in the bitmask
    non_zero_elements = cv2.findNonZero(bitmask)
    # Use the minAreaRect function to find the bounding box of the rotated rectangle
    rect = cv2.minAreaRect(non_zero_elements)
    # Extract the bounding box points from the rect object
    bounding_box = cv2.boxPoints(rect)
    # Convert the bounding box points to integer values
    bounding_box = np.int0(bounding_box)
    return bounding_box


def draw_boxes(image, masks, class_ids, scores):
    best_guess = {k: (-1 * float("Inf"), None) for k in set(class_ids)}
    for i, class_id in enumerate(class_ids):
        top = best_guess[class_id]
        if top[0] < scores[i]:
            best_guess[class_id] = (scores[i], i)
    for class_id, (score, i) in best_guess.items():
        mask = masks[:, :, i]
        label = OBJ_CLASSES[class_id - 1]
        coords = bitmask_to_bounding_box(mask.astype(np.uint8))
        cv2.polylines(image, [coords], True, (0, 0, 255), 2)
    for class_id, (score, i) in best_guess.items():
        mask = masks[:, :, i]
        # class_id = class_ids[i]
        label = OBJ_CLASSES[class_id - 1]
        coords = bitmask_to_bounding_box(mask.astype(np.uint8))
        x, y, w, h = cv2.boundingRect(coords)
        center_x, center_y = x + w // 2, y + h // 2
        # Convert the coordinates to a numpy array
        coords = np.array(coords, dtype=np.int32)
        # Add black border
        cv2.putText(
            image,
            label,
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        # Put the label text on the image
        cv2.putText(
            image,
            label,
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    card_name = None
    for class_id, (score, i) in best_guess.items():
        mask = masks[:, :, i]
        label = OBJ_CLASSES[class_id - 1]
        # class_id = class_ids[i]
        coords = bitmask_to_bounding_box(mask.astype(np.uint8))
        if label == "name":
            card_mask = masks[:, :, best_guess[OBJ_CLASSES.index("card") + 1][1]]
            set_mask = masks[:, :, best_guess[OBJ_CLASSES.index("set_symbol") + 1][1]]
            card_coords = bitmask_to_bounding_box(card_mask.astype(np.uint8))
            set_coords = bitmask_to_bounding_box(set_mask.astype(np.uint8))
            name_points = np.array(coords, dtype=np.int32)
            card_points = np.array(card_coords, dtype=np.int32)
            set_points = np.array(set_coords, dtype=np.int32)
            # Get the center-point of the card
            x, y, w, h = cv2.boundingRect(card_coords)
            center = (x + w // 2, y + h // 2)
            x, y, w, h = cv2.boundingRect(coords)
            center_name = (x + w // 2, y + h // 2)
            # Find top left corner as it's the closest to the farthest corner
            top_left = sorted(
                card_points, key=lambda point: np.linalg.norm(point - center_name)
            )[0]
            top_left_name = sorted(
                name_points, key=lambda point: np.linalg.norm(point - top_left)
            )[0]
            top_left_set = sorted(
                set_points, key=lambda point: np.linalg.norm(point - top_left_name)
            )[0]
            if (
                top_left_name[0] < top_left_set[0]
                and top_left_name[1] < top_left_set[1]
            ):
                angle = 0
            elif (
                top_left_name[0] >= top_left_set[0]
                and top_left_name[1] < top_left_set[1]
            ):
                angle = -90
            elif (
                top_left_name[0] >= top_left_set[0]
                and top_left_name[1] >= top_left_set[1]
            ):
                angle = 180
            elif (
                top_left_name[0] < top_left_set[0]
                and top_left_name[1] >= top_left_set[1]
            ):
                angle = 90
            # Coarse rotation trim
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_image = cv2.warpAffine(
                image.copy(), rotation_matrix, (image.shape[1], image.shape[0])
            )
            rotated_mask = cv2.warpAffine(
                mask.copy().astype(np.uint8),
                rotation_matrix,
                (mask.shape[1], mask.shape[0]),
            )
            # Create a kernel for the dilation operation
            kernel = np.ones(
                (int(mask.shape[1] * 0.01), int(mask.shape[0] * 0.025)), np.uint8
            )
            # Apply the dilation operation to the mask
            rotated_mask = cv2.dilate(rotated_mask, kernel, iterations=1)
            rotated_card_mask = cv2.warpAffine(
                masks[:, :, best_guess[OBJ_CLASSES.index("card") + 1][1]]
                .copy()
                .astype(np.uint8),
                rotation_matrix,
                (mask.shape[1], mask.shape[0]),
            )
            rotated_set_mask = cv2.warpAffine(
                masks[:, :, best_guess[OBJ_CLASSES.index("set_symbol") + 1][1]]
                .copy()
                .astype(np.uint8),
                rotation_matrix,
                (mask.shape[1], mask.shape[0]),
            )
            card_coords = bitmask_to_bounding_box(rotated_card_mask.astype(np.uint8))
            set_coords = bitmask_to_bounding_box(rotated_set_mask.astype(np.uint8))
            coords = bitmask_to_bounding_box(rotated_mask.astype(np.uint8))
            name_points = np.array(coords, dtype=np.int32)
            card_points = np.array(card_coords, dtype=np.int32)
            set_points = np.array(set_coords, dtype=np.int32)
            # Get the center-point of the card
            x, y, w, h = cv2.boundingRect(card_coords)
            center = (x + w // 2, y + h // 2)
            x, y, w, h = cv2.boundingRect(coords)
            center_name = (x + w // 2, y + h // 2)
            # Find top left corner as it's the closest to the farthest corner
            top_left = sorted(
                card_points, key=lambda point: np.linalg.norm(point - center_name)
            )[0]
            top_left_name = sorted(
                name_points, key=lambda point: np.linalg.norm(point - top_left)
            )[0]
            bottom_left_name, _, bottom_right_name = sorted(
                name_points, key=lambda point: np.linalg.norm(point - top_left_name)
            )[1:]
            top_left_set = sorted(
                set_points, key=lambda point: np.linalg.norm(point - top_left_name)
            )[0]
            third_point = [bottom_left_name[0], bottom_right_name[1]]
            opp = np.linalg.norm(third_point - bottom_left_name)
            adj = np.linalg.norm(third_point - bottom_right_name)
            # Get the angle of rotation
            angle = np.arctan2(opp, adj)
            angle = np.rad2deg(angle)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_image = cv2.warpAffine(
                rotated_image.copy(),
                rotation_matrix,
                (rotated_image.shape[1], rotated_image.shape[0]),
            )
            rotated_mask = cv2.warpAffine(
                rotated_mask.copy().astype(np.uint8),
                rotation_matrix,
                (rotated_mask.shape[1], rotated_mask.shape[0]),
            )
            rotated_set_mask = cv2.warpAffine(
                rotated_set_mask.copy().astype(np.uint8),
                rotation_matrix,
                (rotated_set_mask.shape[1], rotated_set_mask.shape[0]),
            )
            # Create a kernel for the dilation operation
            kernel = np.ones(
                (int(rotated_mask.shape[1] * 0.01), int(rotated_mask.shape[0] * 0.025)),
                np.uint8,
            )
            # Apply the dilation operation to the mask
            rotated_mask = cv2.dilate(rotated_mask, kernel, iterations=1)
            coords = bitmask_to_bounding_box(rotated_mask.astype(np.uint8))
            name_points = np.array(coords, dtype=np.int32)
            # Find the minimum and maximum x and y coordinates of the original points
            min_x, min_y = np.min(name_points, axis=0)
            max_x, max_y = np.max(name_points, axis=0)
            # Crop the image using the minimum and maximum x and y coordinates
            cropped_image = rotated_image.copy()[min_y:max_y, min_x:max_x]
            # Define the destination image size
            dst_size = (max_x - min_x, max_y - min_y)
            # Resize the cropped image to the destination size
            cropped_image = cv2.resize(cropped_image, dst_size)
            res = ocr.ocr(cropped_image, det=False, cls=False)
            card_name = fuzzy_search.match_card_name(res[0][0][0])

    # Find top left corner as it's the closest to the farthest corner
    return image, card_name


def detect_and_guess(image):
    orig_shape = image.shape[:2]
    resized_image = resize_and_pad(image, (1024, 1024))
    # Detect objects
    r = model.detect([resized_image], verbose=1)[0]
    resized_masks = []
    for i in range(r["masks"].shape[2]):
        mask = r["masks"][:, :, i]
        nm = reverse_resize_and_pad(mask.astype(np.uint8), orig_shape)
        resized_masks.append(nm)
    r_masks = np.stack(resized_masks, axis=2)
    return draw_boxes(image, r_masks, r["class_ids"], r["scores"])


if __name__ == "__main__":
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    with open("calibration_images/calibration.json") as f:
        calib = json.load(f)
        camera_matrix = np.array(calib["camera_matrix"])
        dist_coeffs = np.array(calib["dist_coeff"])
    while True:
        undistort = False
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        if undistort:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            frame = cv2.flip(frame, 1)  # flip the image vertically
            frame = cv2.transpose(frame)
        try:
            frame, card_name = detect_and_guess(model, frame)
            print("Predicted Card:", card_name)
        except Exception:
            print("Nothing to print")

        # Show the frame with the bounding boxes and text content
        cv2.imshow("QR Code", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
            # Break the loop if the 'q' key is pressed
        if cv2.waitKey(100) & 0xFF == ord("p"):
            print(card_name)
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
