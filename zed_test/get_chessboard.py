import cv2
from PIL import Image
import numpy as np
import os

img = np.array(Image.open(os.path.join(os.path.dirname(__file__), "Image.png")))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

ret, corners = cv2.findChessboardCorners(gray_img, (6, 5), None)
if ret == True:
    corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1),
                                       criteria)
else:
    exit(1)

corners = np.squeeze(corners)

font = cv2.FONT_HERSHEY_SIMPLEX

for idx, point in enumerate(corners):
    img = cv2.circle(img, tuple(point), 10, (int(255*idx/30), int(255*idx/30), int(255*idx/30)), -1)
    img = cv2.putText(img, f"{idx:02d}", tuple(point), font, 0.4, (255, 0, 0))

Image.fromarray(img).save(os.path.join(os.path.dirname(__file__), "Image_chessboard.png"))