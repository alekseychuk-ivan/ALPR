from enum import Enum
from typing import Dict, List
import torch
from pydantic import BaseModel, constr
from starlette.responses import JSONResponse
from ultralytics import YOLO
from pathlib import Path
from paddleocr import PaddleOCR
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, StreamingResponse
from function.processing import *

weights = Path('detect/model.pt')
yolo = YOLO(model=weights)
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir='./detect/det_dir/',
                rec_model_dir='./detect/rec_dir/',
                cls_model_dir='./detect/cls_dir/')

app = FastAPI(
    title="Custom ALPR",
    description="""Obtain object value out of image
    and return image and json result""",
    version="0.0.1",
)


class CarType(Enum):
    car = 'car'
    truck = 'truck'


class CarPlate(BaseModel):
    type: CarType
    plate: str
    # plate_conf: None | float


ConStrType = constr(min_length=1)  # constr(regex=r'^[A-Z0-9]{4}-[A-Z0-9]{6}$')
CarDict = Dict[ConStrType, CarPlate]


class CarModel(BaseModel):
    __root__: CarDict


# origins = [
#     "http://localhost",
#     "http://localhost:8000",
#     "*"
# ]
# app.add_middleware(
#      CORSMiddleware,
#      allow_origins=origins,
#      allow_credentials=True,
#      allow_methods=["*"],
#      allow_headers=["*"],
# )


@app.get("/")
def read_root() -> JSONResponse:
    return JSONResponse({"message": " Hello World"})


# @app.get('/notify/v1/health')
# def get_health():
#     return dict(msg='OK')


@app.get('/notify/cuda')
def check_cuda() -> JSONResponse:
    return JSONResponse(dict(device='cuda' if torch.cuda.is_available() else 'cpu'))


# @app.post("/detectPIL")
# async def detect(file: UploadFile = File(...)):
#     # since = time.time()
#     contents = await file.read()
#     pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
#     pil_image = pil_image.resize((224, 224))
#     filtered_image = io.BytesIO()
#     pil_image.save(filtered_image, "JPEG")
#     filtered_image.seek(0)
#     # print(time.time() - since)
#     # print(type(filtered_image))
#
#     return StreamingResponse(content=filtered_image, media_type="image/jpeg")


@app.post("/detectCV")
async def detect(file: UploadFile = File(...)) -> StreamingResponse:
    # since = time.time()
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    # res, im_png = cv2.imencode(".jpeg", image)
    imcopy = image.copy()
    outputs = yolo.predict(source=image, save=False)
    for output in outputs:
        for out in output:
            if out.boxes.cls == plate:
                xyxy = out.boxes.xyxy[0]
                x, y, w, h = map(int, xyxy)
                im = image[y:h, x:w, :]
                im = cv2.resize(im, (94, 24), interpolation=cv2.INTER_CUBIC)
                result = ocr.ocr(im, det=False, cls=False)
                text = datafilter(result[0][0][0])
                print(result)
                if len(text) == 0:
                    continue
                if len(text) < 5:
                    result = ocr.ocr(read_pate(image[y - 1: h, x - 1: w, :]), det=False, cls=False)
                    text = datafilter(result[0][0][0])

                imcopy = cv2.rectangle(img=imcopy, pt1=(x, y), pt2=(w, h), color=(255, 0, 255),
                                       thickness=3)
                imcopy = cv2.putText(imcopy, text, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=0.8, color=(0, 255, 255), thickness=2, )
    #
    # # print(time.time() - since)
    # # print(type(io.BytesIO(im_png.tobytes())))
    res, im_png = cv2.imencode(".jpeg", imcopy)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")


@app.post("/platecar", response_model=List[CarDict])
async def detectplatecar(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    outputs = yolo.predict(source=image, save=False)
    platelst = list()
    i = 0
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
                            if len(text) > 3:
                                platelst.append({f'Object_{i}':
                                                {'type': f'{"car" if out.boxes.cls == car else "truck"}',
                                                 'plate': text}})
                                i += 1
                                # platelst[f'Object_{i}'] = [(text, f'{"car" if out.boxes.cls == car else "truck"}')]
    return platelst


@app.post("/plate")
async def detectplate(file: UploadFile = File(...)):
    # since = time.time()
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    imcopy = image.copy()
    outputs = yolo.predict(source=image, save=False)
    platelst = list()
    for output in outputs:
        for out in output:
            if out.boxes.cls == plate:
                xyxy = out.boxes.xyxy[0]
                x, y, w, h = map(int, xyxy)
                im = image[y:h, x:w, :]
                im = cv2.resize(im, (94, 24), interpolation=cv2.INTER_CUBIC)
                result = ocr.ocr(im, det=False, cls=False)
                text = datafilter(result[0][0][0])
                if len(text) < 5:
                    result = ocr.ocr(read_pate(image[y - 1: h, x - 1: w, :]), det=False, cls=False)
                    text = datafilter(result[0][0][0])

                platelst.append(text)

    return platelst

# @app.post("/detectPILjson")
# async def detect(file: UploadFile = File(...)):
#     # since = time.time()
#     contents = await file.read()
#     pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
#     pil_image = pil_image.resize((224, 224))
#     filtered_image = io.BytesIO()
#     pil_image.save(filtered_image, "JPEG")
#     filtered_image.seek(0)
#     # print(time.time() - since)
#     print(type(filtered_image.getvalue()))
#
#     return Response(content=filtered_image.getvalue(), headers=dict([('1', '2')]),  media_type="image/jpeg")


# @app.post("/detectCV")
# async def detect(file: UploadFile = File(...)):
#     # since = time.time()
#     contents = await file.read()
#     nparr = np.fromstring(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#     res, im_png = cv2.imencode(".jpeg", img)
#
#     # print(time.time() - since)
#     print(type(im_png.tobytes()))
#     return Response(im_png.tobytes(), media_type="image/jpeg")


# if __name__ == '__main__':
#     import uvicorn
#
#     app_str = 'app:app'
#     uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)
