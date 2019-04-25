# csu_fascilitated_by_svr
BPP4 and BPP5 AFM 2018 Demo showing coalition situational understanding fascilitated by symbolic vector represenation service orchistration


#flask services
To use the pipeline, you will need to start each of the flask services. Navigate to each of the flask service folders (e.g. csu_fascilitated_by_svr/flask_services/cameras/) and run the as-a-service ("AAS") python file. 

Example:
python TflCameraAAS.py

This will start the flask service for that part of the pipeline. 


For object detection, ensure opencv contrib is installed (pip install opencv-contrib-python)

There is a text file containing a node-red flow that chains together the flask services for testing. For this, ensure that node-red dashboard is installed

Currently the object detector will return non-car objects (as well as cars), we need to have a discussion RE where the list would be filtered/how the service chain knows it needs to be filtered etc...
