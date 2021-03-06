import sys
sys.path.append("/home/dks/workspace/hackathon/price_estimate/src")



from flask import Flask, redirect, render_template, request, session, url_for, Response
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from googleapiclient import discovery

import os
import cv2
# from skimage import io
import json
import urllib.request
import numpy as np
import pathlib

import glob
import os
import numpy as np
import face_recognition
known_face_encodings = []
known_face_names = []

def load_image(path):
    for img in glob.glob(path + '/*.jpg'):
        name = os.path.basename(img).split('.')[0]
        print('User loading ...', name)
        known_face_names.append(name)
        image = face_recognition.load_image_file(img)
        image_encode = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(image_encode)

def identify(path):
    input_img = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(input_img)
    face_encodings = face_recognition.face_encodings(input_img, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        print('face distance: ', face_distances)
        # print('face matches: ', matches)
        if face_distances[best_match_index] <= 0.5:
            name = known_face_names[best_match_index]

        face_names.append(name)

    print(face_names)
    return face_names[0]



ENGINE_IMG_SIZE = 299
DETECTION_THRESH = 0.5
item_list = \
[{'name': 'None', 'price': 5},
 {'name': 'person', 'price': 5},
 {'name': 'bicycle', 'price': 5},
 {'name': 'car', 'price': 5},
 {'name': 'motorcycle', 'price': 5},
 {'name': 'airplane', 'price': 5},
 {'name': 'bus', 'price': 5},
 {'name': 'train', 'price': 5},
 {'name': 'truck', 'price': 5},
 {'name': 'boat', 'price': 5},
 {'name': 'traffic light', 'price': 5},
 {'name': 'fire hydrant', 'price': 5},
 {'name': 'street sign', 'price': 5},
 {'name': 'stop sign', 'price': 5},
 {'name': 'parking meter', 'price': 5},
 {'name': 'bench', 'price': 5},
 {'name': 'bird', 'price': 5},
 {'name': 'cat', 'price': 5},
 {'name': 'dog', 'price': 5},
 {'name': 'horse', 'price': 5},
 {'name': 'sheep', 'price': 5},
 {'name': 'cow', 'price': 5},
 {'name': 'elephant', 'price': 5},
 {'name': 'bear', 'price': 5},
 {'name': 'zebra', 'price': 5},
 {'name': 'giraffe', 'price': 5},
 {'name': 'hat', 'price': 5},
 {'name': 'backpack', 'price': 5},
 {'name': 'umbrella', 'price': 5},
 {'name': 'shoe', 'price': 5},
 {'name': 'eye glasses', 'price': 5},
 {'name': 'handbag', 'price': 5},
 {'name': 'tie', 'price': 5},
 {'name': 'suitcase', 'price': 5},
 {'name': 'frisbee', 'price': 5},
 {'name': 'skis', 'price': 5},
 {'name': 'snowboard', 'price': 5},
 {'name': 'sports ball', 'price': 5},
 {'name': 'kite', 'price': 5},
 {'name': 'baseball bat', 'price': 5},
 {'name': 'baseball glove', 'price': 5},
 {'name': 'skateboard', 'price': 5},
 {'name': 'surfboard', 'price': 5},
 {'name': 'tennis racket', 'price': 5},
 {'name': 'bottle', 'price': 5},
 {'name': 'plate', 'price': 5},
 {'name': 'wine glass', 'price': 5},
 {'name': 'cup', 'price': 5},
 {'name': 'fork', 'price': 5},
 {'name': 'knife', 'price': 5},
 {'name': 'spoon', 'price': 5},
 {'name': 'bowl', 'price': 5},
 {'name': 'banana', 'price': 5},
 {'name': 'apple', 'price': 5},
 {'name': 'sandwich', 'price': 5},
 {'name': 'orange', 'price': 5},
 {'name': 'broccoli', 'price': 5},
 {'name': 'carrot', 'price': 5},
 {'name': 'hot dog', 'price': 5},
 {'name': 'pizza', 'price': 5},
 {'name': 'donut', 'price': 5},
 {'name': 'cake', 'price': 5},
 {'name': 'chair', 'price': 5},
 {'name': 'couch', 'price': 5},
 {'name': 'potted plant', 'price': 5},
 {'name': 'bed', 'price': 5},
 {'name': 'mirror', 'price': 5},
 {'name': 'dining table', 'price': 5},
 {'name': 'window', 'price': 5},
 {'name': 'desk', 'price': 5},
 {'name': 'toilet', 'price': 5},
 {'name': 'door', 'price': 5},
 {'name': 'tv', 'price': 5},
 {'name': 'laptop', 'price': 5},
 {'name': 'mouse', 'price': 5},
 {'name': 'remote', 'price': 5},
 {'name': 'keyboard', 'price': 5},
 {'name': 'cell phone', 'price': 5},
 {'name': 'microwave', 'price': 5},
 {'name': 'oven', 'price': 5},
 {'name': 'toaster', 'price': 5},
 {'name': 'sink', 'price': 5},
 {'name': 'refrigerator', 'price': 5},
 {'name': 'blender', 'price': 5},
 {'name': 'book', 'price': 5},
 {'name': 'clock', 'price': 5},
 {'name': 'vase', 'price': 5},
 {'name': 'scissors', 'price': 5},
 {'name': 'teddy bear', 'price': 5},
 {'name': 'hair drier', 'price': 5},
 {'name': 'toothbrush', 'price': 5},
 {'name': 'hair brush', 'price': 5}]

load_image('./known_face')

app = Flask(__name__)
dropzone = Dropzone(app)

app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

# Call google api envs
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './hackathon-kkb-b0dac966ae6a.json'
PROJECT = "hackathon-kkb"
MODEL   = "hackathon_predict"
INPUT_NODE = "inputs"
VERSION = "version_2"

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


@app.route('/', methods=['GET', 'POST'])
def index():
    def predict_json(project, model, instances, version=None):
        """Send json data to a deployed model for prediction.

        Args:
            project (str): project where the Cloud ML Engine Model is deployed.
            model (str): model name.
            instances ([Mapping[str: Any]]): Keys should be the names of Tensors
                your deployed model expects as inputs. Values should be datatypes
                convertible to Tensors, or (potentially nested) lists of datatypes
                convertible to tensors.
            version: str, version of the model to target.
        Returns:
            Mapping[str: any]: dictionary of prediction results defined by the
                model.
        """
        # Create the ML Engine service object.
        # To authenticate set the environment variable
        # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
        # CREDENTIALS = app_engine.Credentials()
        # service = discovery.build('ml', 'v1', credentials=CREDENTIALS)
        service = discovery.build('ml', 'v1')
        name = 'projects/{}/models/{}'.format(project, model)
        if version is not None:
            name += '/versions/{}'.format(version)
        response = service.projects().predict(
            name=name,
            body={'instances': instances}
        ).execute()
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['predictions']

    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']

    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            # cv2.imwrite()

            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename
            )

            # append image urls
            file_urls.append(photos.url(filename))
            # img = io.imread(file_urls[0])
            # print(img)
            req = urllib.request.urlopen(file_urls[0])
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
            # w, h , _ = image.shape
            img = cv2.resize(img, (299,299))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_json = {
                'inputs': img.tolist()
            }
            # call gcp api
            print('aaaaaaaaaaaa')
            res = predict_json(PROJECT, MODEL, img_json, VERSION)
            print('bbbbbbbbbbb')
            res = res[0]

            item_price, item_amount, item_price_total, total = estimate(res)
            session['item_amount'] = item_amount
            session['item_price_total'] = item_price_total
            session['item_price'] = item_price
            session['total'] = total

            break

        session['file_urls'] = file_urls
        return "uploading..."
        # return render_template('capture_face.html')
    # return dropzone template on GET request
    return render_template('index.html')



@app.route('/results')
def results():

    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))

    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)

    item_price, item_amount, item_price_total, total = session['item_price'], session['item_amount'], session['item_price_total'], session['total']
    # items = [{'coca': 1, 'pepsi': 2, 'aquafina': 5}]

    return render_template('results.html', file_urls=file_urls, item_amount=item_amount, item_price=item_price, item_price_total=item_price_total, total=total)


def draw_box(image, boxes):
    width = (image.shape[0] / ENGINE_IMG_SIZE)
    height = (image.shape[1] / ENGINE_IMG_SIZE)

    for box in boxes:
        (top_left) = ((int)(box[0] * width), (int)(box[1] * height))
        (bottom_right) = ((int)(box[2] * width) + top_left[0], (int)(box[3] * height) + top_left[1])
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)

    return image


def estimate(res):
    num_of_detections = int(res['num_detections'])
    detection_classes = res['detection_classes']
    detection_scores = res['detection_scores']

    total_price = 0.0
    total_item_price = 0.0
    purchases_amount = dict()
    purchases_item_price = dict()
    item_price = dict()

    for i in range(num_of_detections):
        if detection_scores[i] < DETECTION_THRESH:
            continue

        purchases_amount[item_list[int(detection_classes[i])]['name']] = 0
        item_price[item_list[int(detection_classes[i])]['name']] = item_list[int(detection_classes[i])]['price']
        purchases_item_price[item_list[int(detection_classes[i])]['name']] = 0.0

    for i in range(num_of_detections):
        if detection_scores[i] < DETECTION_THRESH:
            continue

        purchases_amount[item_list[int(detection_classes[i])]['name']] += 1
        purchases_item_price[item_list[int(detection_classes[i])]['name']] += item_list[int(detection_classes[i])]['price']
        total_price += item_list[int(detection_classes[i])]['price']

    return item_price, purchases_amount, purchases_item_price, total_price


def test_json(json_path):
    with open(json_path) as f:
        res = json.load(f)
        purchases = estimate(res)

        print(purchases)


def gen():
    import time
    cap = cv2.VideoCapture(0)
    capturing = True
    start = time.time()
    while capturing:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break

        cv2.imwrite('demo.jpg', frame)
        if time.time() - start >= 3.0:
            cap.release()
            capturing = False
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')
@app.route('/get_user')
def get_user():
    name = identify('demo.jpg')
    return Response(name)


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
