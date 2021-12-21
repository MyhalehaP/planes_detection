import json
import os
from flask import Flask, request, jsonify, send_file, Response
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

            flag = False
            if request.form.get('mode') == 'zoom':
                flag = True

            processed= finder.find_plane(image=img, zoom=flag)

            processed = cv2.imencode('.png', processed)[1]

            output_img= base64.b64encode(processed)



            return Response(response=output_img,content_type='image/png')
    return "lol"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
