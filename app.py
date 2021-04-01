import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask
import json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

UPLOAD_FOLDER = 'C:/Users/adaml/PycharmProjects/flaskProject2/static'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


from flask import Flask, render_template, request, redirect, flash, url_for
import main
import urllib.request
from app import app
from werkzeug.utils import secure_filename
from main import getPrediction
import os

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label = getPrediction(filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(img_path)
            return render_template('index.html', resultat = label, affich_image = url_for('static', filename = filename))
       #     return redirect('/')

if __name__ == "__main__":
    app.run()