from flask import Flask, jsonify, request, render_template, current_app, redirect, url_for
from io import BytesIO
import base64
import json
from PIL import Image
import google.generativeai as genai
import cv2
from keras import models
from keras import preprocessing
import tensorflow
import numpy as np

app = Flask(__name__)
genai.configure(api_key="AIzaSyBkdAFRqUSFfhOe4ldTl6UBaCB0idFw3lg")
generator = genai.GenerativeModel("gemini-1.5-flash")

# x = open("BrightSide/app/fer3.json", "r")
model = models.model_from_json(open("BrightSide/app/fer3.json", "r").read())  
#load weights  
model.load_weights('BrightSide/app/fer3.weights.h5') 
face_haar_cascade = cv2.CascadeClassifier("BrightSide/app/haarcascade_frontalface_default.xml")  

@app.route('/')
def root():

    return render_template("index.html")
    # return redirect("/meme")

@app.route('/switch_page_to_home')
def switch_page_to_home():
    return redirect(url_for("home"))

@app.route('/switch_page_to_meme')
def switch_page_to_meme():
    return redirect(url_for("meme"))

@app.route('/switch_page_to_happy')
def switch_page_to_happy():
    return redirect(url_for("happy"))

@app.route('/switch_page_to_ssad')
def switch_page_to_ssad():
    return redirect(url_for("ssad"))

@app.route('/switch_page_to_megasad')
def switch_page_to_megasad():
    return redirect(url_for("megasad"))

@app.route('/meme')
def meme():
    return render_template("meme.html")

@app.route('/happy')
def happy():
    return render_template("happy.html")

@app.route('/ssad')
def ssad():
    return render_template("ssad.html")

@app.route('/megasad')
def megasad():
    return render_template("megasad.html")

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/generate', methods = ['POST'])
def generateText():
    print("test generate")
    # data = request.form()
    data = request.get_json()
    print("after")
    # text = data['data']
    # data = request.data.decode('utf-8')
    text = data['data']
    print(text)
    output = generator.generate_content(text)
    return jsonify(output.text)

@app.route('/predict', methods=['POST'])
def predictions_endpoint():
    if request.method == 'POST':

        image_base64 = request.data

        encodings = image_base64.split(b',')[-1]    # removes data:image/png;base64 from encoding i think
        # print(encodings)
        image_bytes = base64.decodebytes(encodings) 
        # print(image_bytes)
        pil_image = Image.open(BytesIO(image_bytes))

        white = (255,255,255)
        background = Image.new(pil_image.mode[:-1], pil_image.size, white)
        background.paste(pil_image, pil_image.split()[-1])  # make transparent background of png white for conversion to jpg
        pil_image = background
        
        pil_image = pil_image.convert('RGB')    # originally rgba now rgb because png to jpg
        """
        JEFF AND TUAN: pil_image is pure image RGB jpg in PIL Image format,
        I need you to evaluate on model and set value variable below to 0 if neutral, 1 if happy, 2 if sad, 3 for mega negative emotions
        believe in frontend magic
        """
        pil_image.save("before resize.jpg")
        pil_image = pil_image.crop((100, 25, pil_image.width-100, pil_image.height-25))
        pil_image.save("after resize.jpg")

        # pil_image = pil_image.resize((48, 48)) 

        test_img = np.array(pil_image)
        test_img = test_img[:, :, ::-1].copy()
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  
        # gray_img = pil_image.convert("L")
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  
        cv2.imwrite("BrightSide/app/cv_img1.jpg", test_img) 
        cv2.imwrite("BrightSide/app/cv_img2.jpg", gray_img) 
        value = 1   # number representation of emotion returned by function

        # cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
        # roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(gray_img,(48,48))  
        img_pixels = preprocessing.image.img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  

        predictions = model.predict(img_pixels)  

        #find max indexed array  
        max_index = np.argmax(predictions[0])  

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
        # predicted_emotion = emotions[max_index]
        if(max_index == 6):
            # value = 0
            value = 1      # supposed to be neutral but lets switch to happy
        elif max_index == 3 or max_index == 5:
            value = 1
        elif max_index == 4:
            value = 2
        else:
            value = 3
        # if max_index == 3 or max_index == 6:
        #     value = 1
        # else:
        #     value = 2

        print("value is " + str(value))
        return jsonify(value)

        # for (x,y,w,h) in faces_detected:  
        #     cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
        #     roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
        #     roi_gray=cv2.resize(roi_gray,(48,48))  
        #     img_pixels = preprocessing.image.img_to_array(roi_gray)  
        #     img_pixels = np.expand_dims(img_pixels, axis = 0)  
        #     img_pixels /= 255  

        #     predictions = model.predict(img_pixels)  

        #     #find max indexed array  
        #     max_index = np.argmax(predictions[0])  

        #     emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
        #     # predicted_emotion = emotions[max_index]
        #     if(max_index == 6):
        #         value = 0
        #     elif max_index == 3 or max_index == 5:
        #         value = 1
        #     elif max_index == 4:
        #         value = 2
        #     else:
        #         value = 3

        #     return jsonify(value)

        return jsonify(value)
    
if __name__ == "__main__":



    # genai.configure(api_key="AIzaSyBkdAFRqUSFfhOe4ldTl6UBaCB0idFw3lg")
    # model = genai.GenerativeModel("gemini-1.5-flash")
    # response = model.generate_content("cheer me up, i had a bad day in one sentence")
    # print(response.text)

    host = "127.0.0.1"
    port_number = 8080 

    # app.run(debug=True)
    app.run(host, port_number)