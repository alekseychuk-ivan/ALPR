from ultralytics import YOLO
from pathlib import Path
from paddleocr import PaddleOCR
import os
from function.processing import *
from shapely.geometry import Polygon, Point
import matplotlib.path as mplPath
import numpy as np

weight = Path('detect/model.pt')
imagefolder = Path('test')
outfolder = Path('out')
yolo = YOLO(model=weight)
file = f'D:\\Python\\ALPR\\test\\20230831_666_0.jpg'
# ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir='./detect/det_dir/',
#                 rec_model_dir='./detect/rec_dir/',
#                 cls_model_dir='./detect/cls_dir/')

filename = Path(file).stem
path = Path(file).parent

image = cv2.imread(file)
imcopy = image.copy()
outputs = yolo.predict(source=image, save=False)
carsdct, platesdct, car_i, plate_i = dict(), dict(), 0, 0

for output in outputs:
    for out in output:
        x, y, w, h = map(int, out.boxes.xyxy[0])
        if out.boxes.cls == plate:
            platesdct[f'plate_{plate_i}'] = [[x, y, w, h],  int(out.boxes.conf), int(out.boxes.cls)]
            plate_i += 1
        elif out.boxes.cls == car or out.boxes.cls == truck:
            carsdct[f'car_{car_i}'] = [[x, y, w, h], int(out.boxes.conf)]
            car_i += 1


for plate in platesdct:
    xp, yp, wp, hp = platesdct[plate][0]
    for car in carsdct:
        xc, yc, wc, hc = carsdct[car][0]
        if xc <= xp and yc <= yp and wp <= wc and hp <= hc:
            with open(f"{os.path.join(path, filename)}.txt", "a") as my_file:
                my_file.write()



#
# for car in carsdct:
#     # x, y, w, h = map(int, carsdct[car][0])
#     x, y, w, h = carsdct[car][0]
#     imcopy = cv2.rectangle(img=imcopy, pt1=(x, y), pt2=(w, h), color=(255, 0, 255),
#                            thickness=3)


# cv2.imshow('image', imcopy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for p, _, files in os.walk(imagefolder):
#     for file in files:
#         image = cv2.imread(os.path.join(p, file))
#         imcopy = image.copy()
#
#         outputs = yolo.predict(source=image, save=False)
#         platelst = list()
#         for output in outputs:
#             for out in output:
#                 if out.boxes.cls == car or out.boxes.cls == truck:
#                     xyxy = out.boxes.xyxy[0]
#                     x, y, w, h = map(int, xyxy)
#                     carimage = image[y:h, x:w, :]
#
#                     caroutputs = yolo.predict(source=carimage, save=False)
#                     for caroutput in caroutputs:
#                         for carout in caroutput:
#                             if carout.boxes.cls == plate:
#                                 xyxy = carout.boxes.xyxy[0]
#                                 x, y, w, h = map(int, xyxy)
#                                 carplate = carimage[y:h, x:w, :]
#                                 carplate = cv2.resize(carplate, (94, 24), interpolation=cv2.INTER_CUBIC)
#                                 numplate = ocr.ocr(carplate, det=False, cls=False)
#                                 text = datafilter(numplate[0][0][0])
#                                 # if len(text) < 5:
#                                 #         numplate = ocr.ocr(read_pate(carimage[y - 1: h, x - 1: w, :]), det=False, cls=False)
#                                 #         text = datafilter(numplate[0][0][0])
#                                 if len(text) > 3:
#                                     platelst.append((text, f'{"car" if out.boxes.cls == car else "truck"}'))
#         print(platelst)

