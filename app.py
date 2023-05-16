import torch
from ultralytics import YOLO
from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, StreamingResponse
import cv2
import numpy as np

weights = weight = Path('detect/weights/model.pt')
model = YOLO(model=weights)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

app = FastAPI(
    title="Custom ALPR",
    description="""Obtain object value out of image
    and return image and json result""",
    version="0.0.1",
)

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


@app.get('/notify/v1/health')
def get_health():
    return dict(msg='OK')


@app.get('/notify/cuda')
def check_cuda():
    return dict(device='cuda' if torch.cuda.is_available() else 'cpu')


@app.post("/detectPIL")
async def detect(file: UploadFile = File(...)):
    # since = time.time()
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    pil_image = pil_image.resize((224, 224))
    filtered_image = io.BytesIO()
    pil_image.save(filtered_image, "JPEG")
    filtered_image.seek(0)
    # print(time.time() - since)
    # print(type(filtered_image))

    return StreamingResponse(content=filtered_image, media_type="image/jpeg")


@app.post("/detectCV")
async def detect(file: UploadFile = File(...)):
    # since = time.time()
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    res, im_png = cv2.imencode(".jpeg", img)

    # print(time.time() - since)
    # print(type(io.BytesIO(im_png.tobytes())))
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

@app.post("/detectPILjson")
async def detect(file: UploadFile = File(...)):
    # since = time.time()
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    pil_image = pil_image.resize((224, 224))
    filtered_image = io.BytesIO()
    pil_image.save(filtered_image, "JPEG")
    filtered_image.seek(0)
    # print(time.time() - since)
    print(type(filtered_image.getvalue()))

    return Response(content=filtered_image.getvalue(), headers=dict([('1', '2')]),  media_type="image/jpeg")
#
#
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


def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize((
        int(input_image.width * resize_factor),
        int(input_image.height * resize_factor)
    ))
    return resized_image


if __name__ == '__main__':
    import uvicorn

    app_str = 'app:app'
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)
