import json
import os
import random
import time
import urllib.request

import cv2
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


def download_image(url, file_path, overwrite=False):
    # Open the URL and download the image data
    if not overwrite and os.path.exists(file_path):
        print(f"{file_path} already downloaded...")
        return False
    with urllib.request.urlopen(url) as url_handle:
        image_data = url_handle.read()

    # Write the image data to the specified file
    with open(file_path, "wb") as f:
        f.write(image_data)
    return True


with open("/Users/kiansheik/Downloads/oracle-cards-20221222220256.json") as f:
    all_cards = json.load(f)

# for card in unique_artwork:


def overlay_image(
    foreground_image_path, annotation, background_image_path, overwrite=False
):
    """
    Overlays the foreground image on top of the background image at a random position.
    The images must be the same type and have the same size.
    """
    path_parts = foreground_image_path.split(".")
    output_image_path = f"{'/'.join(foreground_image_path.split('/')[:-1])}/{path_parts[-2].split('/')[-1]}_{background_image_path.split('/')[-1]}"
    if not overwrite and os.path.exists(output_image_path):
        return False
    background_image = cv2.imread(background_image_path)
    foreground_image = cv2.imread(foreground_image_path)

    # Get the size of the foreground image
    fg_height, fg_width, _ = foreground_image.shape

    # Calculate the ratio of the foreground image
    ratio = fg_width / fg_height

    # Get the size of the background image
    bg_height, bg_width, _ = background_image.shape

    # Choose a random size for the foreground image
    max_height = bg_height
    max_width = bg_width
    if max_height * ratio > max_width:
        new_height = random.randint(max_height / 4, max_height * 0.7)
        new_width = new_height * ratio
    else:
        new_width = random.randint(max_width / 4, max_width * 0.7)
        new_height = new_width / ratio

    # Resize the foreground image
    foreground_image = cv2.resize(foreground_image, (int(new_width), int(new_height)))
    print(bg_width, new_width, bg_height, new_height)
    # Choose random coordinates for the top left corner of the foreground image
    x = random.randint(0, bg_width - int(new_width))
    y = random.randint(0, bg_height - int(new_height))

    # Overlay the foreground image on top of the background image
    background_image[y : y + int(new_height), x : x + int(new_width)] = foreground_image

    # Save the resulting image
    cv2.imwrite(output_image_path, background_image)
    return True


def overlay_image_annotate(
    foreground_image_path, annotation, background_image_path, overwrite=False
):
    # Load the base image
    base_image = cv2.imread("base_image.jpg")

    # Extract the annotation data
    annotations = data["annotations"][0]["result"]

    # Get the dimensions of the base image
    base_image_height, base_image_width, _ = base_image.shape

    # Generate random values for the size and angle of rotation
    scale = random.uniform(0.5, 1.0)
    angle = random.uniform(-45, 45)

    # Compute the size of the scaled and rotated base image
    scaled_base_image_height = int(base_image_height * scale)
    scaled_base_image_width = int(base_image_width * scale)

    # Scale the base image
    scaled_base_image = cv2.resize(
        base_image, (scaled_base_image_width, scaled_base_image_height)
    )

    # Compute a rotation matrix for rotating the base image
    rotation_matrix = cv2.getRotationMatrix2D(
        (scaled_base_image_width / 2, scaled_base_image_height / 2), angle, 1.0
    )

    # Rotate the base image
    rotated_base_image = cv2.warpAffine(
        scaled_base_image,
        rotation_matrix,
        (scaled_base_image_width, scaled_base_image_height),
    )

    # Loop through the annotations
    for annotation in annotations:
        # Extract the annotation data
        x = annotation["value"]["x"]
        y = annotation["value"]["y"]
        width = annotation["value"]["width"]
        height = annotation["value"]["height"]
        label = annotation["value"]["rectanglelabels"][0]

        # Scale the annotation data
        x = int(x * scale)
        y = int(y * scale)
        width = int(width * scale)
        height = int(height * scale)

        # Rotate the annotation data
        points = np.array(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
        )
        points = cv2.transform(points.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)
        x, y = points[0]
        width = points[2][0] - points[0][0]
        height = points[2][1] - points[0][1]


def transform_and_rotate_image(
    image_path,
    annotation,
    background_image_path,
    output_image_path,
    output_annotation_path,
):
    # Load image and annotation
    image = cv2.imread(image_path)
    with open(annotation_path) as f:
        annotation = json.load(f)

    # Get image rotation from annotation
    image_rotation = annotation["annotations"][0]["result"][0]["image_rotation"]
    # Rotate image
    image = rotate_image(image, image_rotation)

    # Load background image
    background_image = cv2.imread(background_image_path)

    # Overlay image on top of background image
    background_image[: image.shape[0], : image.shape[1]] = image

    # Save output image
    cv2.imwrite(output_image_path, background_image)

    # Update annotation with new image dimensions and rotation
    annotation["annotations"][0]["result"][0][
        "original_width"
    ] = background_image.shape[1]
    annotation["annotations"][0]["result"][0][
        "original_height"
    ] = background_image.shape[0]
    annotation["annotations"][0]["result"][0]["image_rotation"] = 0

    # Save output annotation
    with open(output_annotation_path, "w") as f:
        json.dump(annotation, f)


def rotate_image(image, angle):
    # Get image dimensions
    (h, w) = image.shape[:2]
    # Calculate center of image
    center = (w / 2, h / 2)
    # Perform rotation using OpenCV's warpAffine function
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


annotations = load_annotations(
    "/Users/kiansheik/code/card-sorting-bot/vision/annotations/project-3-at-2022-12-25-00-09-f9c60779.json"
)
annotated_ids = {
    x["data"]["image"].split("/")[-1].split(".")[0]: x for x in annotations
}
breakpoint()

for card in all_cards:
    if card["id"] in annotated_ids:
        try:
            dl = download_image(
                card["image_uris"]["border_crop"], f"data/{card['id']}.jpg"
            )
            overlay_image(
                f"data/{card['id']}.jpg",
                card["id"],
                "/Users/kiansheik/Downloads/table_bg.jpeg",
            )
            if dl:
                time.sleep(1)
        except Exception as e:
            print(f"failed: ({e})")
