from ultralytics import YOLO
from pathlib import Path
from paddleocr import PaddleOCR
import os
from function.processing import *


weight = Path('detect/model.pt')
imagefolder = Path('test')
outfolder = Path('out')
yolo = YOLO(model=weight)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# for p, _, files in os.walk(imagefolder):
#     for file in files:
#         image = cv2.imread(os.path.join(p, file))
#         imcopy = image.copy()
#         outputs = yolo.predict(source=image, save=False)
#         for output in outputs:
#             for out in output:
#                 if out.boxes.cls == 1:
#                     xyxy = out.boxes.xyxy[0]
#                     x, y, w, h = map(int, xyxy)
#                     im = image[y:h, x:w, :]
#                     im = cv2.resize(im, (94, 24), interpolation=cv2.INTER_CUBIC)
#                     result = ocr.ocr(im, det=False, cls=False)
#                     text = datafilter(result[0][0][0])
#                     if len(text) < 5:
#                         result = ocr.ocr(read_pate(image[y - 1: h, x - 1: w, :]), det=False, cls=False)
#                         text = datafilter(result[0][0][0])
#
#                     imcopy = cv2.rectangle(img=imcopy, pt1=(x, y), pt2=(w, h), color=(255, 0, 255),
#                                            thickness=3)
#                     imcopy = cv2.putText(imcopy, text, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                                          fontScale=0.8, color=(0, 255, 255), thickness=2, )
#
#             cv2.imwrite(f'{os.path.join(outfolder, file)}', imcopy)


for p, _, files in os.walk(imagefolder):
    for file in files:
        image = cv2.imread(os.path.join(p, file))
        imcopy = image.copy()

        # cv2.imshow('window_name', image)
        # cv2.waitKey(0)  # waits until a key is pressed
        # cv2.destroyAllWindows()  # destroys the window showing image

        outputs = yolo.predict(source=image, save=False)
        platelst = list()
        for output in outputs:
            for out in output:
                if out.boxes.cls == car or out.boxes.cls == truck:
                    xyxy = out.boxes.xyxy[0]
                    x, y, w, h = map(int, xyxy)
                    carimage = image[y:h, x:w, :]

                    caroutputs = yolo.predict(source=carimage, save=False)
                    for caroutput in caroutputs:
                        for carout in caroutput:
                            if carout.boxes.cls == plate:
                                xyxy = carout.boxes.xyxy[0]
                                x, y, w, h = map(int, xyxy)
                                carplate = carimage[y:h, x:w, :]
                                carplate = cv2.resize(carplate, (94, 24), interpolation=cv2.INTER_CUBIC)
                                numplate = ocr.ocr(carplate, det=False, cls=False)
                                text = datafilter(numplate[0][0][0])
                                # if len(text) < 5:
                                #         numplate = ocr.ocr(read_pate(carimage[y - 1: h, x - 1: w, :]), det=False, cls=False)
                                #         text = datafilter(numplate[0][0][0])
                                if len(text) > 3:
                                    platelst.append((text, f'{"car" if out.boxes.cls == car else "truck"}'))
        print(platelst)

