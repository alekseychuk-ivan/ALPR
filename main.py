from ultralytics import YOLO
from pathlib import Path
import cv2
from paddleocr import PaddleOCR
import os

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from scipy.stats import mode
from skimage.filters import threshold_otsu, sobel, scharr
from skimage import io
from skimage import util
from matplotlib import pyplot as plt


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


def getContours(img, orig):  # Change - pass the original image too
    biggest = np.array([])
    maxArea = 0
    imgContour = orig.copy()  # Make a copy of the original image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        if area > 20: # 500 20
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*4*peri, True) # 5
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                index = i  # Also save index to contour

    warped = None  # Stores the warped license plate image
    if index is not None:  # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

        src = np.squeeze(biggest).astype(np.float32)  # Source points
        height = orig.shape[0]
        width = orig.shape[1]
        # Destination points
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        src = order_points(src)
        dst = order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image
        img_shape = (width, height)
        warped = cv2.warpPerspective(orig, M, img_shape, flags=cv2.INTER_LINEAR)

    return biggest, imgContour, warped  # Change - also return drawn image


def binarizeImage(RGB_image):
    image = rgb2gray(RGB_image)
    threshold = threshold_otsu(image)
    bina_image = image < threshold
    io.imshow(bina_image)
    return bina_image


def findEdges(bina_image):
    image_edges = sobel(bina_image)
    return image_edges


def findTiltAngle(image_edges):
    h, theta, d = hough_line(image_edges)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    angle = np.rad2deg(mode(angles)[0][0])
    image_edges = util.invert(image_edges)
    if (angle < 0):

        r_angle = 90 - abs(angle)

    else:

        r_angle = angle - 90

    origin = np.array((0, image_edges.shape[1]))

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)

    return r_angle


def rotateImage(RGB_image, angle):
    fixed_image = rotate(RGB_image, angle)
    return fixed_image


def generalPipeline(img):
    # image = io.imread(img)
    bina_image = binarizeImage(img)
    image_edges = findEdges(bina_image)
    angle = findTiltAngle(image_edges)
    return rotateImage(img, angle)


weight = Path('detect/weights/model.pt')
imagefolder = Path('test')
outfolder = Path('out')
# image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite('resize.jpg', image)
yolo = YOLO(model=weight)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

for p, _, files in os.walk(imagefolder):
    for file in files:
        image = cv2.imread(os.path.join(p, file))
        imcopy = image.copy()
        outputs = yolo.predict(source=image, save=False)
        for output in outputs:
            for out in output:
                if out.boxes.cls == 1:
                    xyxy = out.boxes.xyxy[0]
                    x, y, w, h = map(int, xyxy)
                    im = image[y:h, x:w, :]
                    # cv2.imwrite(f'crop{file}', im)
                    # kernel = np.ones((3, 3))
                    # imgGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    # imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
                    # imgCanny = cv2.Canny(imgBlur, 150, 200)
                    # imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
                    # imgThres = cv2.erode(imgDial, kernel, iterations=2)
                    # biggest, imgContour, warped = getContours(imgThres, im)  # Change
                    #
                    # titles = ['Original', 'Blur', 'Canny', 'Dilate', 'Threshold', 'Contours',
                    #           'Warped']  # Change - also show warped image
                    # images = [image[..., ::-1], imgBlur, imgCanny, imgDial, imgThres, imgContour, warped]  # Change
                    #
                    # # Change - Also show contour drawn image + warped image
                    # for i in range(5):
                    #     plt.subplot(3, 3, i + 1)
                    #     plt.imshow(images[i], cmap='gray')
                    #     plt.title(titles[i])
                    #
                    # plt.subplot(3, 3, 6)
                    # plt.imshow(images[-2])
                    # plt.title(titles[-2])
                    #
                    # plt.subplot(3, 3, 8)
                    # plt.imshow(images[-1])
                    # plt.title(titles[-1])
                    #
                    # plt.show
                    im = cv2.resize(im, (94, 24), interpolation=cv2.INTER_CUBIC)
                    result = ocr.ocr(im, det=False, cls=False)
                    for idx in range(len(result)):
                        res = result[idx]
                        for line in res:
                            print(line)
                            imcopy = cv2.rectangle(img=imcopy, pt1=(x, y), pt2=(w, h), color=(127, 127, 255),
                                                   thickness=3)
                            imcopy = cv2.putText(imcopy, line[0], org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                 fontScale=0.8, color=(127, 127, 255), thickness=2, )

            cv2.imwrite(f'{os.path.join(outfolder, file)}', imcopy)
