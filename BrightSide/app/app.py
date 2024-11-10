from flask import Flask, jsonify, request, render_template, current_app, redirect, url_for
from io import BytesIO
import base64
import json
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)
genai.configure(api_key="AIzaSyBkdAFRqUSFfhOe4ldTl6UBaCB0idFw3lg")
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/')
def root():
    # return render_template("index.html",user_image ='default.jpg')
    print("open app")
    # return render_template("meme.html")
    # return render_template("meme.html")
    # return render_template("ssad.html")

    return render_template("index.html")
    # return redirect("/meme")

@app.route('/switch_page_to_home')
def switch_page_to_home():
    return redirect(url_for("home"))

@app.route('/switch_page_to_meme')
def switch_page_to_meme():
    return redirect(url_for("meme"))

@app.route('/switch_page_to_ssad')
def switch_page_to_ssad():
    return redirect(url_for("ssad"))

@app.route('/switch_page_to_megasad')
def switch_page_to_megasad():
    return redirect(url_for("megasad"))

@app.route('/meme')
def meme():
    return render_template("meme.html")

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
    output = model.generate_content(text)
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

        pil_image = pil_image.crop((175, 75, pil_image.width-175, pil_image.height-75))
        pil_image.save("before resize.jpg")

        pil_image = pil_image.resize((48, 48)) 

        pil_image.save("test2.jpg")

        value = 2   # number representation of emotion returned by function

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