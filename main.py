from ultralytics import YOLO
from pathlib import Path
import cv2
from paddleocr import PaddleOCR


weight = Path('detect/weights/model.pt')
image = cv2.imread('test/BYD_Yuan_facelift_004.jpg')
imcopy = image.copy()
# image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('resize.jpg', image)
yolo = YOLO(model=weight)
ocr = PaddleOCR(use_angle_cls=True, lang='en')
outputs = yolo.predict(source=image, save=False)
for output in outputs:
    for out in output:
        if out.boxes.cls == 1:
            xyxy = out.boxes.xyxy[0]
            x, y, w, h = map(int, xyxy)
            im = image[y:h, x:w, :]
            im = cv2.resize(im, (94, 24), interpolation=cv2.INTER_CUBIC)
            result = ocr.ocr(im, det=False, cls=False)
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    print(line)
                    imcopy = cv2.rectangle(img=imcopy, pt1=(x, y), pt2=(w, h), color=(127, 127, 255), thickness=3)
                    imcopy = cv2.putText(imcopy, line[0], org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.8, color=(127, 127, 255), thickness=2, )

    cv2.imwrite('result.jpg', imcopy)
