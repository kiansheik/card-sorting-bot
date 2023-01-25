import cv2
import numpy as np

# Define the aruco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

# Define the size of the tag
size = 0.03  # Meters

# Define the number of tags to generate
num_tags = 50

# Create an empty image to draw the tags on
img = 255 * np.ones((1000, 1000), dtype=np.uint8)

# Iterate through the number of tags
for i in range(num_tags):
    # Generate the tag
    marker = cv2.aruco.drawMarker(aruco_dict, i, int(size * 1000))
    # Define the position of the tag
    position = ((i % 10) * 100, (i // 10) * 100)
    # Paste the tag onto the image
    img[
        position[1] : position[1] + marker.shape[0],
        position[0] : position[0] + marker.shape[1],
    ] = marker

# Save the image
cv2.imwrite("aruco_tags.png", img)
