import re
import cv2
import numpy as np

car = 0
plate = 1
truck = 2


def read_pate(img):
    kernel = np.ones((3, 3))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 150, 200)
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThres = cv2.erode(imgDial, kernel, iterations=3)
    contours, _ = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    src = order_points(box).astype(np.float32)
    height = img.shape[0]
    width = img.shape[1]
    # Destination points
    dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])
    dst = order_points(dst).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image
    img_shape = (width, height)
    warped = cv2.warpPerspective(img, M, img_shape, flags=cv2.INTER_CUBIC)
    warped = cv2.resize(warped, (96, 24), interpolation=cv2.INTER_CUBIC)
    return warped


def datafilter(org_string):
    pattern = r'([^A-Z0-9])'
    return re.sub(pattern, '', org_string)


def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts, axis=0)

    # Step 2: Move coordinate system to centre of object
    shifted = pts - center

    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta, axis=0)
    return pts[ind]
