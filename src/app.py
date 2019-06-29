from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from googleapiclient import discovery

import os
import cv2
from skimage import io
import json

ENGINE_IMG_SIZE = 299

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
            image = cv2.imread(file_urls[0])
            w,h, _ = image.shape
            img = cv2.resize(img, (299,299))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_json = {
                'inputs': image.tolist()
            }
            # call gcp api
            res = predict_json(PROJECT, MODEL, img_json, VERSION)
            res = res[0]

        session['file_urls'] = file_urls
        return "uploading..."
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

    items = [{'coca': 1, 'pepsi': 2, 'aquafina': 5}]
    
    return render_template('results.html', file_urls=file_urls, items=items)


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
    
        
if __name__ == '__main__':
    app.run()