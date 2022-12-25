import imgaug as ia
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
from random import random
import imutils


def img_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 20, 100)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # ensure at least one contour was found
    if len(cnts) > 0:
        # grab the largest contour, then draw a mask for the pill
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # compute its bounding box of pill, then extract the ROI,
        # and apply the mask
        (x, y, w, h) = cv2.boundingRect(c)
        imageROI = image[y : y + h, x : x + w]
        maskROI = mask[y : y + h, x : x + w]
        imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)
    return mask


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    rotated_image = cv2.warpAffine(image, M, (nW, nH))
    # create a blank mask with the same size as the rotated image
    mask = np.zeros((nH, nW), dtype=np.uint8)
    # draw a filled rectangle on the mask with the same dimensions as the original image
    cv2.rectangle(
        mask,
        (cX - w // 2, cY - h // 2),
        (cX + w // 2, cY + h // 2),
        (255, 255, 255),
        -1,
    )
    # apply the rotation transformation to the mask
    mask = cv2.warpAffine(mask, M, (nW, nH))
    # return both the rotated image and the mask
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

    # Scale the foreground image
    foreground_scaled = ia.imresize_single_image(
        foreground,
        (
            int(foreground.shape[0] * scale_factor),
            int(foreground.shape[1] * scale_factor),
        ),
    )
    foreground_scaled, mask = rotate_bound(foreground_scaled, 360 * angle)
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
        img, background, scale_factor=random(), x=random(), y=random(), angle=random()
    )
    # Display the augmented image using OpenCV
    cv2.imshow("Augmented Image", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
