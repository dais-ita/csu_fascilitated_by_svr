import numpy as np
import cv2

class YoloObjectDetector(object):
    """docstring for YoloObjectDetector"""
    def __init__(self, config_path,weights_path):
        super(YoloObjectDetector, self).__init__()
        self.config_path = config_path
        self.weights_path = weights_path
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.InitialiseClassesAndColours()


    def InitialiseClassesAndColours(self):
        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        self.CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def GetOutputLayers(self,net):
    
        layer_names = self.net.getLayerNames()
        
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return output_layers


    def DetectObjectsFromPath(self,image_path,confidence_threshold=-1,as_dict = False):
        img = cv2.imread(image_path)
        return self.DetectObjectsFromArray(img,confidence_threshold,as_dict)


    def DetectObjectsFromArray(self,image,confidence_threshold=-1,as_dict = False):
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        self.net.setInput(blob)

        outs = self.net.forward(self.GetOutputLayers(self.net))

        class_ids = []
        confidences = []
        boxes = []
        if(confidence_threshold > 0):
            conf_threshold = confidence_threshold
        else:
            conf_threshold = 0.0

        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        detections = list(zip(class_ids,confidences,boxes))

        if(as_dict):
            return self.DetectionsToDicts(detections)
        else:
            return detections 


    def DetectionsToDicts(self,detections):
        detection_dicts_list = []
        if(len(detections) == 0):
            return detection_dicts_list
        
        for detection in detections:
            detection_dicts_list.append(self.DetectionToDict(detection))

        return detection_dicts_list


    def DetectionToDict(self,detection):
        idx = int(detection[0])
        class_name = self.CLASSES[idx]
        confidence = float(detection[1])
        box = detection[2]
        (startX, startY, width, height) = box
        
        return {"label":class_name,"confidence":confidence,"startX":int(startX),"startY":int(startY),"endX":int(startX+width),"endY":int(startY+height)}


    def DrawDetections(self, input_image,detections,confidence_threshold=-1):
        for detection in detections:
            label = detection["label"]

            color = self.COLORS[self.CLASSES.index(label)]

            cv2.rectangle(input_image, (detection["startX"],detection["startY"]), (detection["endX"],detection["endY"]), color, 2)

            cv2.putText(input_image, label, (detection["startX"]-10,detection["startY"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return input_image


if __name__ == '__main__':
    net_config_path = "yolov3.cfg"
    net_weights_path = "yolov3.weights"

    confidence_threshold = 0.6

    yolo_detector = YoloObjectDetector(net_config_path,net_weights_path)

    image_path = "tfl.jpg"

    detections = yolo_detector.DetectObjectsFromPath(image_path)

    print(detections)

    
    input_image = cv2.imread(image_path)

    detection_image = yolo_detector.DrawDetections(input_image,detections,confidence_threshold=confidence_threshold)

    cv2.imshow('detected objects', detection_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()