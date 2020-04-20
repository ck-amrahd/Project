import numpy as np
import cv2


class Detector:
    def __init__(self, cls_file, cfg_file, weight_file, confidence_threshold, nms_threshold, input_width,
                 input_height):
        self.cls_file = cls_file
        self.cfg_file = cfg_file
        self.weight_file = weight_file
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_width = input_width
        self.input_height = input_height

        self.model = cv2.dnn.readNetFromDarknet(self.cfg_file, self.weight_file)

        self.class_names = []
        with open(cls_file, 'r') as f:
            for line in f:
                class_name = line.strip()
                self.class_names.append(class_name)

        self.out_layers = self.model.getUnconnectedOutLayersNames()

    def detect(self, img_path):
        img = cv2.imread(img_path)
        img_blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.input_width, self.input_height),
                                         swapRB=True, crop=False)
        self.model.setInput(img_blob)
        outputs = self.model.forward(self.out_layers)

        height, width, _ = img.shape
        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    width = int(detection[2] * width)
                    height = int(detection[3] * height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indces = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        objects = []
        for i in indces:
            i = i[0]
            class_id = int(class_ids[i])
            objects.append(self.class_names[class_id])
        return objects
