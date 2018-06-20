# csu_fascilitated_by_svr
BPP4 and BPP5 AFM 2018 Demo showing coalition situational understanding fascilitated by symbolic vector represenation service orchistration


#flask services

for object detection, ensure opencv contrib is installed (pip install opencv-contrib-python)

there is a text file containing a node-red flow that chains together the flask services for testing. For this, ensure that node-red dashboard is installed

currently the object detector will return non-car objects (as well as cars), we need to have a discussion RE where the list would be filtered/how the service chain knows it needs to be filtered etc...
