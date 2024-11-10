from flask import Flask, jsonify, request, render_template, current_app, redirect
from io import BytesIO
import base64
import json
from PIL import Image

app = Flask(__name__)

@app.route('/')
def root():
    # return render_template("index.html",user_image ='default.jpg')
    print("open app")
    # return render_template("meme.html")
    # return render_template("meme.html")

    return render_template("index.html")

# @app.route('/switch')
@app.route('/meme', methods=['GET', 'POST'])
def switch_page():
    print("asdkbjfslkgkdfbgdkgnkdfg")
    # return redirect("meme.html")

    return render_template("meme.html")

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

        value = 5   # number representation of emotion returned by function

        return jsonify(value)
    
if __name__ == "__main__":
    
    host = "127.0.0.1"
    port_number = 8080 

    # app.run(debug=True)
    app.run(host, port_number)