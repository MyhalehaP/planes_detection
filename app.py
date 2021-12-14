import json
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import finder
import cv2
from numpyencoder import NumpyEncoder
import base64

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
UPLOAD_FOLDER = os.path.dirname(__file__)

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return "hello there"


@app.route('/detect_plane', methods=['GET', 'POST'])
def detect():

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            path_to_file = os.path.join('images/', filename)
            file.save(path_to_file)


            img = cv2.imread(path_to_file, cv2.IMREAD_COLOR)

            processed = finder.find_plane(image=img)
            processed = cv2.imencode('.jpg', processed)[1]
            output_img = str(base64.b64encode(processed))
            output_img = output_img[2:]
            output_img = output_img[:len(output_img)-2]

            return jsonify(image=output_img)
    return "lol"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
