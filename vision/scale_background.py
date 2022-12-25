import json
import random
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import imutils
import numpy as np
import pandas as pd
from PIL import Image


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
    boxes = []
    for i, bounding_box in enumerate(bounding_boxes):
        box_x, box_y, box_w, box_h = bounding_box
        # Convert the bounding box coordinates into a 2D point array
        points = np.array(
            [
                [box_x, box_y],
                [box_x + box_w, box_y],
                [box_x + box_w, box_y + box_h],
                [box_x, box_y + box_h],
            ]
        )
        # apply the transformation to the bounding box coordinates
        points_transformed = cv2.warpAffine(points, M, (w, h))
        box = [
            int(x)
            for x in [
                points_transformed[:, 0].min(),
                points_transformed[:, 1].min(),
                points_transformed[:, 0].max() - points_transformed[:, 0].min(),
                points_transformed[:, 1].max() - points_transformed[:, 1].min(),
            ]
        ]
        boxes.append(box)
    # return both the rotated image and the mask
    return rotated_image, mask, boxes


def overlay_image(
    foreground, background, scale_factor=0.5, x=0.5, y=0.5, angle=0.5, bounding_boxes=[]
):
    foreground_scaled, mask, boxes = rotate_bound(
        foreground, 360 * angle, bounding_boxes
    )
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

    # Overlay the scaled foreground image on the background image
    mask_inv = cv2.bitwise_not(mask)
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
    return background


background = cv2.imread("/Users/kiansheik/Downloads/table_bg.jpeg")
annotations = load_annotations(
    "/Users/kiansheik/code/card-sorting-bot/vision/annotations/project-3-at-2022-12-25-00-09-f9c60779.json"
)
annotated_ids = {
    x["data"]["image"].split("/")[-1].split(".")[0]: x for x in annotations
}

with open("/Users/kiansheik/Downloads/oracle-cards-20221222220256.json") as f:
    all_cards = json.load(f)

for card in all_cards:
    if card["id"] in annotated_ids:
        # Extraxt bounding boxes from annotations to be transformed
        notes = annotated_ids[card["id"]]["annotations"][-1]["result"]
        boxes = [
            (
                x["value"]["x"],
                x["value"]["y"],
                x["value"]["width"],
                x["value"]["height"],
            )
            for x in notes
        ]

        res = overlay_image(
            cv2.imread(f"vision/data/{card['id']}.jpg"),
            background,
            scale_factor=random.uniform(0.75, 1),
            x=random.random(),
            y=random.random(),
            angle=random.uniform(0, 1),
            bounding_boxes=boxes,
        )
        # Display the augmented image using OpenCV
        cv2.imshow("Augmented Image", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
