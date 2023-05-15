from ultralytics import YOLO
from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image
import io
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

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


@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    return {"result": detect_res}

@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")


def get_image_from_bytes(binary_image, max_size=1024):
  input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
  width, height = input_image.size
  resize_factor = min(max_size / width, max_size / height)
  resized_image = input_image.resize((
    int(input_image.width * resize_factor),
    int(input_image.height * resize_factor)
  ))
  return resized_image

# @app.get("/")
# async def home(request: Request):
#   ''' Returns barebones HTML form allowing the user to select a file and model '''
#
#   html_content = '''
# <form method="post" enctype="multipart/form-data">
#   <div>
#     <label>Upload Image</label>
#     <input name="file" type="file" multiple>
#     <div>
#       <label>Select YOLO Model</label>
#       <select name="model_name">
#         <option>yolov5s</option>
#         <option>yolov5m</option>
#         <option>yolov5l</option>
#         <option>yolov5x</option>
#       </select>
#     </div>
#   </div>
#   <button type="submit">Submit</button>
# </form>
# '''
#   return HTMLResponse(content=html_content, status_code=200)
#
#
# @app.post("/")
# async def process_home_form(file: UploadFile = File(...),
#                             model_name: str = Form(...)):
#   '''
#   '''
#
#   # model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, force_reload=False)
#
#   # This is how you decode + process image with PIL
#   img = Image.open(BytesIO(await file.read()))
#   results = model(img)
#
#   # This is how you decode + process image with OpenCV + numpy
#   # results = model(cv2.cvtColor(cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR))
#
#   json_results = results_to_json(results, model)
#   return json_results
#
#
# def results_to_json(results, model):
#   ''' Helper function for process_home_form()'''
#   return [
#     [
#       {
#         "class": int(pred[5]),
#         "class_name": model.model.names[int(pred[5])],
#         "bbox": [int(x) for x in pred[:4].tolist()],  # convert bbox results to int from float
#         "confidence": float(pred[4]),
#       }
#       for pred in result
#     ]
#     for result in results.xyxy
#   ]


if __name__ == '__main__':
  import uvicorn

  app_str = 'app:app'
  uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)