import imgaug as ia
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import random
import imutils


def rotate_bound(image, angle):
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
    return rotated_image, mask


def overlay_image(image, background, scale_factor=0.5, x=0.5, y=0.5, angle=0.5):
    # Convert the foreground and background images to numpy arrays
    foreground = image  # ia.imresize_single_image(np.array(image), (64, 64))
    background = (
        background  # ia.imresize_single_image(np.array(background), (512, 512))
    )

    # Calculate the scale factor to fit the foreground image inside the background image
    scale_factor_x = background.shape[1] / foreground.shape[1]
    scale_factor_y = background.shape[0] / foreground.shape[0]
    scale_factor = min(scale_factor_x, scale_factor_y) * scale_factor

    # # Scale the foreground image
    # foreground_scaled = ia.imresize_single_image(
    #     foreground,
    #     (
    #         int(foreground.shape[0] * scale_factor),
    #         int(foreground.shape[1] * scale_factor),
    #     ),
    # )
    foreground_scaled, mask = rotate_bound(foreground, 360 * angle)
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

    # Convert the image back to a PIL image and save it
    result = Image.fromarray(background)
    return background


img = cv2.imread(
    "/Users/kiansheik/code/card-sorting-bot/vision/data/68593c6d-8d1b-4c36-84a1-f1144669825e.jpg"
)
background = cv2.imread("/Users/kiansheik/Downloads/table_bg.jpeg")

for i in range(10):
    res = overlay_image(
        img,
        background,
        scale_factor=random.uniform(0.5, 1),
        x=random.random(),
        y=random.random(),
        angle=random.uniform(-0.29, -0.21),
    )
    # Display the augmented image using OpenCV
    cv2.imshow("Augmented Image", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
