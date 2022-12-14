from imp import load_module
import shutil
from typing import Union
from fastapi import FastAPI, Form, File, UploadFile,Response
from pydantic import BaseModel
from enum import Enum
import cv2
import numpy as np
from keras.models import load_model
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import base64

class Data(BaseModel):
    image: str 


app = FastAPI()

origins = [

    "http://localhost",
    "http://localhost:3000",
    'https://facial-ai-client.vercel.app/'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# root path '/'

@app.post("/upload-file/")
async def root(file: UploadFile = File(...)):
    print('jjj',file)
    if not file:
        return {"message": "No upload file sent"}
    
    else:
        with open("file.jpg", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"message":file.filename}

@app.post("/upload/")
async def upload(file: str = Form(...)):
    # print('file',file.split(',')[1])
    new_file = file.split(',')[1]
    img_recovered = base64.b64decode((new_file))
    
    if not file:
        return {"message": "No upload file sent"}
    
    else:
        with open("file.jpg", "wb") as f:
            f.write(img_recovered)
    print('data')
    return {"message": "picture saved"}
@app.get("/image/")
async def get_image():
    with open("file.jpg", "rb") as image:
        return Response(content=image, media_type="image/jpeg")
@app.get("/predict/")
async def create_upload_file():
    print('------------',cv2.imread('file.jpg') is None)
    if( cv2.imread('file.jpg') is not None):
        
        img1 = cv2.imread('file.jpg')
        print('hhhhhh',img1)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        # print('cascade:: ',cascade)
        faces = cascade.detectMultiScale(gray, 1.1, 3)
        # print('firs image', cascade.detectMultiScale(gray, 1.1, 3))

        for x, y, w, h in faces:
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = img1[y:y + h, x:x + w]
            print('crooped', cropped)

            cv2.imwrite('after.jpg', img1)
        print('cv2 image', img1)

        try:
            cv2.imwrite('cropped.jpg', cropped)
        except:
            pass

        #######

        try:
            image = cv2.imread('cropped.jpg', 0)

        except:
            return {'error': 'Failed to open the uploaded Image'}
            # image = cv2.imread('file.jpg', 0)

        print('=====',image)
        if( image is not None):
            image = cv2.resize(image, (48, 48))
            # print('--------------------------', image)

            image = image / 255.0

            image = np.reshape(image, (1, 48, 48, 1))
            # print('type ------------>',image)

            model = load_model('model.h5')

            prediction = model.predict(image)

            label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
            prediction = np.argmax(prediction)
            # print('predict',prediction)
          
            final_prediction = label_map[prediction]
            files =  ('file.jpg', open('file.jpg', 'rb'), "image/jpeg")
            # os.remove('cropped.jpg')
            # os.remove('file.jpg')
            # os.remove('after.jpg')
            return {"result": final_prediction,'file':files}
        else:
            return {'error':'Check if you have Uploaded a correct Image'}
    else:
                return {'error': 'uploaded Image'}