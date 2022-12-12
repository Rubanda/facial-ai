from imp import load_module
import shutil
from typing import Union
from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
from enum import Enum
import cv2
import numpy as np
from keras.models import load_model

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()


# root path '/'

@app.post("/upload-file/")
async def root(file: UploadFile = File(...)):
    if not file:
        return {"message": "No upload file sent"}
    else:
        with open("file.jpg", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"message":file.filename}


@app.post("/predict/")
async def create_upload_file():
    print(cv2.imread('file.jpg'))
    img1 = cv2.imread('file.jpg')
    print('hhhhhh',img1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    print('cascade:: ',cascade)
    faces = cascade.detectMultiScale(gray, 1.1, 3)
    print('firs image', cascade.detectMultiScale(gray, 1.1, 3))

    for x, y, w, h in faces:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = img1[y:y + h, x:x + w]
        print('crooped', cropped)

        cv2.imwrite('after.jpg', img1)
    # print('cv2 image', img1)

    try:
        cv2.imwrite('cropped.jpg', cropped)
    except:
        pass

    #######

    try:
        image = cv2.imread('cropped.jpg', 0)

    except:
        image = cv2.imread('file.jpg', 0)

    print('=====',image)

    image = cv2.resize(image, (48, 48))
    # print('--------------------------', image)

    image = image / 255.0

    image = np.reshape(image, (1, 48, 48, 1))
    # print('type ------------>',image)

    model = load_model('model.h5')

    prediction = model.predict(image)

    label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]
    return {"filename": final_prediction}
