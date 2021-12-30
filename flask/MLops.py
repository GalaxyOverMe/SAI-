from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from codes.detection import Img_localize, get_edge

# image import
import cv2

# Keras import
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils import
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# 이미지 분류를 위한 딕셔너리
classfication = {
    0 : 'T-shirt/top',
    1 : 'Trouser',
    2 : 'Pullover',
    3 : 'Dress',
    4 : 'Coat',
    5 : 'Sandal',
    6 : 'Shirt',
    7 : 'Sneaker',
    8 : 'Bag',
    9 : 'Ankel boot'
}


app = Flask(__name__)

# 경로 지정
UPLOAD_FOLDER = os.path.join(os.getcwd(),'flask/static/img/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 학습을 위한 모델 불러오기 
MODEL_PATH = os.path.join(os.getcwd(),'classification_model.h5')
model = load_model(MODEL_PATH)
model.make_predict_function()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('MLops_index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        num_result = request.form['Num']
        # Save the file to ./uploads
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)

        #img_localize사용해서 사진 분할 이후 전처리
        preds = []
        detect_imgs = Img_localize(file_path, int(num_result), "cluster")
        detect_imgs = np.array(detect_imgs, np.float32)
        detect_imgs /= 255.
        detect_imgs = np.expand_dims(detect_imgs, axis=-1)
        
        #preds에 결과 값 저장
        pred = model.predict(detect_imgs)
        pred_class = pred.argmax(axis=1)
        preds = []


        for i in range(int(num_result)):
            preds.append(classfication[pred_class[i]])

        #string형태로 결과를 리턴        
        result = ''
        for x in range(int(num_result)):
            result += preds[x] + ','
        
        return result[:-1]
    return None


if __name__ == '__main__':
    app.run(debug=True)