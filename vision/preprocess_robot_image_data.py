import json
import os
import random

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import imutils
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_annotations(json_path):
    with open(json_path) as f:
        a = json.load(f)
    return a


def get_file_paths(dir_path):
    file_paths = []
    for root, directories, files in os.walk(dir_path):
        for file in files:
            if not file.startswith("."):
                file_path = os.path.join(root, file)
                file_path = os.path.abspath(file_path)
                file_paths.append(file_path)
    return file_paths


def rotate_bound(image, angle, bounding_boxes):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    diagonal_length = np.sqrt(w**2 + h**2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, w / diagonal_length)
    # perform the actual rotation and return the image
    rotated_image = cv2.warpAffine(image, M, (w, h))
    # create a blank mask with the same size as the rotated image
    mask = np.zeros((h, w), dtype=np.uint8)
    # draw a filled rectangle on the mask with the same dimensions as the original image
    cv2.rectangle(
        mask,
        (cX - w // 2, cY - h // 2),
        (cX + w // 2, cY + h // 2),
        (255, 255, 255),
        -1,
    )
    # apply the rotation transformation to the mask
    mask = cv2.warpAffine(mask, M, (w, h))
    boxes = dict()
    for label, bounding_box in bounding_boxes.items():
        box_x, box_y, box_w, box_h = [int(x) for x in bounding_box]
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(
            bbox_mask,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (255, 255, 255),
            -1,
        )
        bbox_mask = cv2.warpAffine(bbox_mask, M, (w, h))
        boxes[label] = bbox_mask
    # return both the rotated image and the mask
    return rotated_image, mask, boxes


def add_random_shadow(img):
    # Get the image size
    h, w = img.shape[:2]

    # Create a black image with the same size as the input image
    mask = np.zeros((h, w), dtype=np.uint8)

    # Generate and draw multiple random wavy lines
    num_lines = random.randint(50, 500)
    for i in range(num_lines):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = random.randint(0, w)
        y2 = random.randint(0, h)
        num_waves = random.randint(5, 10)
        wave_amplitude = random.randint(5, 25)
        points = []
        for i in range(num_waves + 1):
            x = x1 + i * (x2 - x1) / num_waves
            y = y1 + i * (y2 - y1) / num_waves
            y += wave_amplitude * np.sin(x * np.pi * 2 / num_waves)
            points.append((x, y))
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(mask, [points], False, (255, 255, 255))

    # Apply a Gaussian blur to the mask image
    ksize = (15, 15)
    sigma = 10.0
    mask = cv2.GaussianBlur(mask, ksize, sigma)

    # Convert the shadow mask to a 3-channel image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Merge the shadow mask with the input image, using a random value for alpha
    alpha = random.uniform(0.5, 0.6)
    beta = 1.0 - alpha
    gamma = 0.0
    dst = cv2.addWeighted(img, alpha, mask, beta, gamma)

    return dst


def skew_image(img, mask=None):
    # Get the image size
    h, w = img.shape[:2]

    # Generate a random skew angle
    angle = random.uniform(-5, 5)

    # Calculate the transformation matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    # Skew the image
    skew_img = cv2.warpAffine(img, M, (w, h))

    # Skew the mask if provided
    if mask is not None:
        skew_mask = cv2.warpAffine(mask, M, (w, h))
    else:
        skew_mask = None

    return skew_img, skew_mask


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


def resize_image(
    foreground,
    output_shape=(640, 640),
    bounding_boxes=dict(),
):
    final_boxes = dict()
    for label, corners in bounding_boxes.items():
        mask = np.zeros((foreground.shape[0], foreground.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [corners], (255, 255, 255))
        mask_scaled = ia.imresize_single_image(
            mask,
            (
                int(output_shape[0]),
                int(output_shape[1]),
            ),
        )
        # # Find the contours of the bounding box
        # _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # Extract the corner coordinates from the contours
        # scaled_corners = np.squeeze(contours[0])
        scaled_corners = bitmask_to_bounding_box(mask_scaled)
        final_boxes[label] = scaled_corners
    foreground_scaled = ia.imresize_single_image(
        foreground,
        (
            int(output_shape[0]),
            int(output_shape[1]),
        ),
    )
    return foreground_scaled, final_boxes


def generate_augmented_dataset(
    output_path, input_path, annotations_path, images_per_background=10
):
    # Load annotations from the provided path
    annotations = load_annotations(annotations_path)
    # Create a dictionary of annotations where the key is the image name
    annotated_ids = {x["data"]["image"].split("/")[-1]: x for x in annotations}
    new_annotations = dict()
    for card_id, annotation in annotated_ids.items():
        # Read the image using OpenCV
        # img = cv2.imread(f"{input_path}/{card_id}")
        # Iterate through each bounding box in the JSON
        notes = annotation["annotations"][-1]["result"]
        boxes = {
            x["value"]["rectanglelabels"][0]: (
                # Get the bounding box coordinates
                x["value"]["x"] / 100.0 * x["original_width"],
                x["value"]["y"] / 100.0 * x["original_height"],
                x["value"]["width"] / 100.0 * x["original_width"],
                x["value"]["height"] / 100.0 * x["original_height"],
                # Get the rotation value and multiply by -1 to change the direction
                -1 * x["value"]["rotation"],
            )
            for x in notes
        }
        bboxes = dict()
        for label, box in boxes.items():
            x, y, w, h, rotation = box
            # Get the center of the bounding box
            cx = x  # + w/2
            cy = y  # + h/2
            # Create a rotation matrix
            M = cv2.getRotationMatrix2D((cx, cy), rotation, 1)
            # Rotate the bounding box
            corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            corners = np.expand_dims(corners, axis=1)
            corners = cv2.transform(corners, M)
            corners = np.squeeze(corners)
            # Draw the rotated bounding box on the image
            # cv2.polylines(img, [np.int32(corners)], True, (0, 255, 0), 2)
            bboxes[label] = np.int32(corners)
            # Draw label for the bounding box
            # cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        new_annotations[card_id] = bboxes
        # # Show the image with the rotated bounding boxes
        # cv2.imshow(card_id, img)
        # k = cv2.waitKey()
        # cv2.destroyAllWindows()
        # if k==27:    # Esc key to stop in window focus, SIGINT in the shell
        #     break
    final_annotations = dict()
    for card_id, bboxes in new_annotations.items():
        # Extraxt bounding boxes from annotations to be transformed
        img = cv2.imread(f"{input_path}/{card_id}")
        res, scaled_bboxes = resize_image(
            img,
            (1024, 1024),
            bounding_boxes=bboxes,
        )
        # Save the image to a file
        out_name = f"{card_id.replace('.jpg', '')}_1024x1024.jpg"
        out_path = f"{output_path}/{out_name}"
        cv2.imwrite(out_path, res)
        final_annotations[out_name] = {
            "bbox": {
                x: [[int(i) for i in t] for t in y] for x, y in scaled_bboxes.items()
            },
            "card_id": card_id,
            "file": out_name,
        }
    breakpoint()
    with open(f"{output_path}/annotations.json", "w") as f:
        json.dump(final_annotations, f)
    print(f"Augmented data generated in {output_path}")


output_path = "/Users/kiansheik/code/mydata/processed_pics"
input_path = "/Users/kiansheik/code/mydata/media/upload/1"
annotations_path = (
    "/Users/kiansheik/code/mydata/pics/project-1-at-2023-01-20-18-36-f8af7d8c.json"
)

generate_augmented_dataset(
    output_path, input_path, annotations_path, images_per_background=7
)
