from flask import Flask, jsonify, request, render_template, current_app

app = Flask(__name__)

@app.route('/')
def root():
    # return render_template("index.html",user_image ='default.jpg')
    return render_template("index.html")

if __name__ == "__main__":
    
    host = "127.0.0.1"
    port_number = 8080 

    app.run(host, port_number)