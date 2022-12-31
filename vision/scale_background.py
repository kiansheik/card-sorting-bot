import json
import random
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import imutils
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_annotations(json_path):
    with open(json_path) as f:
        a = json.load(f)
    annotations = [
        x
        for x in a
        if pd.to_datetime(x["updated_at"])
        >= pd.Timestamp("2022-12-24 21:14:36.599512+0000", tz="UTC")
    ]
    return annotations


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


def overlay_image(
    foreground,
    background,
    output_shape=(640, 640),
    scale_factor=0.5,
    x=0.5,
    y=0.5,
    angle=0.5,
    bounding_boxes=dict(),
    draw_boxes=False,
):
    foreground_scaled, mask, box_masks = rotate_bound(
        foreground, 360 * angle, bounding_boxes
    )
    # Resize background image so all training data is the same
    background = cv2.resize(background, output_shape)
    # Calculate the scale factor to fit the foreground image inside the background image
    scale_factor_x = background.shape[1] / foreground_scaled.shape[1]
    scale_factor_y = background.shape[0] / foreground_scaled.shape[0]
    scale_factor = min(scale_factor_x, scale_factor_y) * scale_factor

    # Scale the foreground image
    foreground_scaled = ia.imresize_single_image(
        foreground_scaled,
        (
            int(foreground.shape[0] * scale_factor),
            int(foreground.shape[1] * scale_factor),
        ),
    )
    # Scale the mask
    mask = ia.imresize_single_image(
        mask,
        (
            int(foreground.shape[0] * scale_factor),
            int(foreground.shape[1] * scale_factor),
        ),
    )
    x = int(x * (background.shape[0] - foreground_scaled.shape[0]))
    y = int(y * (background.shape[1] - foreground_scaled.shape[1]))

    # Overlay the scaleed foreground image on the background image
    mask_inv = cv2.bitwise_not(mask)
    final_boxes = dict()
    for label, box_mask in box_masks.items():
        box_mask = ia.imresize_single_image(
            box_mask,
            (
                int(foreground.shape[0] * scale_factor),
                int(foreground.shape[1] * scale_factor),
            ),
        )
        bg_mask = np.zeros(background.shape[:2], dtype=np.uint8)
        bg_mask[
            x : foreground_scaled.shape[0] + x, y : foreground_scaled.shape[1] + y
        ] = box_mask
        bounding_box = bitmask_to_bounding_box(bg_mask)
        final_boxes[label] = bounding_box.tolist()
    masked_foreground = cv2.bitwise_and(foreground_scaled, foreground_scaled, mask=mask)
    masked_background = cv2.bitwise_and(
        background[
            x : foreground_scaled.shape[0] + x, y : foreground_scaled.shape[1] + y
        ],
        background[
            x : foreground_scaled.shape[0] + x, y : foreground_scaled.shape[1] + y
        ],
        mask=mask_inv,
    )
    background[
        x : foreground_scaled.shape[0] + x, y : foreground_scaled.shape[1] + y
    ] = cv2.add(masked_foreground, masked_background)
    # Convert the bitmask to a bounding box
    # Draw the bounding box on the image as a green rectangle
    if draw_boxes:
        for bounding_box in final_boxes.values():
            cv2.drawContours(background, [bounding_box], -1, (0, 255, 0), 2)
    return background, final_boxes


def generate_augmented_dataset(
    output_path, input_path, background_path, annotations_path, images_per_background=10
):
    annotations = load_annotations(annotations_path)
    annotated_ids = {
        x["data"]["image"].split("/")[-1].split(".")[0]: x for x in annotations
    }
    new_annotations = dict()
    bg_map = {
        bp.split("/")[-1].replace(".", "_"): cv2.imread(bp)
        for bp in get_file_paths(background_path)
    }
    tq = tqdm(
        desc="Augmenting files...",
        total=len(bg_map) * len(annotated_ids) * images_per_background,
    )
    for card_id, annotation in annotated_ids.items():
        card = cv2.imread(f"{input_path}/{card_id}.jpg")
        for background_name, background in bg_map.items():
            for i in range(images_per_background):
                # Extraxt bounding boxes from annotations to be transformed
                notes = annotation["annotations"][-1]["result"]
                boxes = {
                    x["value"]["rectanglelabels"][0]: (
                        x["value"]["x"] / 100.0 * x["original_width"],
                        x["value"]["y"] / 100.0 * x["original_height"],
                        x["value"]["width"] / 100.0 * x["original_width"],
                        x["value"]["height"] / 100.0 * x["original_height"],
                    )
                    for x in notes
                }
                res, bboxes = overlay_image(
                    card,
                    background,
                    (640, 640),
                    scale_factor=random.uniform(0.75, 1),
                    x=random.random(),
                    y=random.random(),
                    angle=random.uniform(0, 1),
                    bounding_boxes=boxes,
                )
                # Save the image to a file
                out_name = f"{card_id}_{background_name}_{i}.jpg"
                out_path = f"{output_path}/{out_name}"
                cv2.imwrite(out_path, res)
                new_annotations[out_name] = {
                    "bbox": bboxes,
                    "card_id": card_id,
                    "file": out_name,
                }
                tq.update(1)
    with open(f"{output_path}/annotations.json", "w") as f:
        json.dump(new_annotations, f)
    print(f"Augmented data generated in {output_path}")


output_path = "/Users/kiansheik/code/card-sorting-bot/vision/augmented_data"
input_path = "/Users/kiansheik/code/card-sorting-bot/vision/data"
background_path = "/Users/kiansheik/code/card-sorting-bot/vision/backgrounds"
annotations_path = "/Users/kiansheik/code/card-sorting-bot/vision/annotations/project-3-at-2022-12-25-00-09-f9c60779.json"

generate_augmented_dataset(
    output_path, input_path, background_path, annotations_path, images_per_background=5
)
