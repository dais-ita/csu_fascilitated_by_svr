from flask import Flask, Response ,send_file , send_from_directory, request
import cv2
import numpy as np
import os

import json

import base64
from PIL import Image
# from StringIO import StringIO
from io import BytesIO as StringIO


from pi_object_detection import PiObjectDetector
from yolo_object_detection import YoloObjectDetector

app = Flask(__name__)

def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def encIMG64(image,convert_colour = False):
    if(convert_colour):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    retval, img_buf = cv2.imencode('.jpg', image)
    
    return base64.b64encode(img_buf)


@app.route("/object_detector/image", methods=['POST', 'GET'])
def DetectObjects():
    if request.method == 'POST':
        if 'image' in request.files:
            input_image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)

            detections = detector.DetectObjectsFromArray(input_image,confidence_threshold=0.5,as_dict = True)

            json_data = json.dumps({'boxes': detections})

            return json_data

        if 'image' in request.form.keys():
            input_image = readb64(request.form["image"])
            
            detections = detector.DetectObjectsFromArray(input_image,confidence_threshold=0.5,as_dict = True)

            # json_data = json.dumps({'boxes': detections})
            
            return str(detections)


        return 'error'
    return '''
    <!doctype html>
    <title>Upload Image File to Detect Objects</title>
    <h1>Upload Image File to Detect Objects</h1>
    <form method=post enctype=multipart/form-data>
    <p><input type=file name=image>
    <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    # prototxt_path = "MobileNetSSD_deploy.prototxt.txt"
    # model_path = "MobileNetSSD_deploy.caffemodel"
    
    # detector = PiObjectDetector(prototxt_path,model_path)
    
    net_config_path = "yolov3.cfg"
    net_weights_path = "yolov3.weights"

    detector = YoloObjectDetector(net_config_path,net_weights_path)

    print('Starting the API')
    app.run(host='0.0.0.0', port=5200)