import cv2
import numpy as np
import os


# Load image, create mask, and draw white circle on mask
image = cv2.imread('images/story.jpeg')
mask = np.zeros(image.shape, dtype=np.uint8)
mask = cv2.circle(mask, (260, 300), 225, (255,255,255), -1) 

# Mask input image with binary mask
result = cv2.bitwise_and(image, mask)
# Color background white
result[mask==0] = 255 # Optional

# cv2.imwrite('image', image)
cv2.imwrite('mask.jpg', mask)
cv2.imwrite('result.jpg', result)
# cv2.waitKey()
