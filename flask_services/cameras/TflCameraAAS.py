from flask import Flask, Response ,send_file , send_from_directory, request, current_app
import cv2
import numpy as np
import os

import json

import base64
from PIL import Image
from StringIO import StringIO

import urllib

import sys


app = Flask(__name__)
camera_id = ""

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


@app.route("/tfl_camera/image", methods=['POST', 'GET'])
def FetchCameraImage():
    global camera_id
    print(camera_id)
    
    api_url = 'http://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.{camera_id}.jpg'

    camera_url = api_url.format(camera_id=camera_id)
    
    resp = urllib.urlopen(camera_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    image_dict = {"image":encIMG64(image)}

    return json.dumps(image_dict)



if __name__ == "__main__":
    if(len(sys.argv) > 1):
        camera_id = sys.argv[1]
        print("")
        print("Creating service for camera with id: "+camera_id)
        print("")
    else:
        camera_id = "02158"
    

    print('Starting the API')
    app.run(host='0.0.0.0', port=5001)