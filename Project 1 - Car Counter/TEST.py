from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort


img = cv2.imread("r.png")
resize_img = cv2.resize(img, (600, 400))
cv2.imshow("r", resize_img)
cv2.waitKey(0)
