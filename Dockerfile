FROM python:3.10-slim-buster

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

#RUN pip install --no-cache
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY /app /code/app
COPY /function /code/function
COPY /detect /code/detect

#EXPOSE 80
# http://127.0.0.1:8000
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]