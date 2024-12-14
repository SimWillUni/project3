import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import gc

''' STEP 1: Object Masking'''

image = cv2.imread("data/motherboard_image.JPEG")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # underscore used to ignore the threshold value output

# Detect edges with Canny
edges = cv2.Canny(thresholded_image, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # underscore used since the variable will not be used later

# Make a minimum contour size
min_area = 1000  # Minimum area threshold
filtered_contours = []
for i in contours:
    if cv2.contourArea(i) > min_area:
        filtered_contours.append(i)