import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap
import numpy as np
import random

# Load the image
image = cv2.imread(
    "/Users/kiansheik/code/card-sorting-bot/vision/data/68593c6d-8d1b-4c36-84a1-f1144669825e.jpg",
    cv2.IMREAD_UNCHANGED,
)
# Get the original dimensions of the image
height, width = image.shape[:2]

# Calculate the new dimensions of the image
new_height = height * 2
new_width = width * 2

# Calculate the padding required to keep the image centered
top_pad = int((new_height - height) / 2)
bottom_pad = new_height - height - top_pad
left_pad = int((new_width - width) / 2)
right_pad = new_width - width - left_pad

# Create a transparent image with the new dimensions
result = np.zeros((new_height, new_width, 3), dtype=np.uint8)

foreground_image = result
background_image = cv2.imread("/Users/kiansheik/Downloads/table_bg.jpeg")

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


# # Copy the original image into the center of the transparent image
# result[top_pad:top_pad+height, left_pad:left_pad+width] = image

# # Define the rotation and scaling transforms
# rotate_transform = iaa.Affine(rotate=(-45, 45))
# scale_transform = iaa.Affine(scale=(1, 1))

# # Combine the transforms into a sequence
# seq = iaa.Sequential([rotate_transform, scale_transform])

# # Apply the transforms to the image
# augmented_im = seq(image=result)

# Convert the augmented image back to a PIL Image
# augmented_im = Image.fromarray(augmented_image)

# Display the augmented image using OpenCV
cv2.imshow("Augmented Image", background_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
