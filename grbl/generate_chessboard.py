import cv2
import numpy as np

# Define the dimensions of the chessboard
rows = 63 // 5
cols = 88 // 5

# Create an empty black image
chessboard_image = np.zeros((rows * 50, cols * 50, 3), np.uint8)

# Draw the chessboard squares
for row in range(rows):
    for col in range(cols):
        color = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
        cv2.rectangle(
            chessboard_image,
            (col * 50, row * 50),
            ((col + 1) * 50, (row + 1) * 50),
            color,
            -1,
        )


cv2.imwrite("chessboard.jpg", chessboard_image)
