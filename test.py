import pickle

adj = pickle.load(open('data/coco_adj.pkl', 'rb'))

breakpoint()

"""
from detector import Detector

cls_file = 'coco.names'
cfg_file = 'yolov3.cfg'
weight_file = 'yolov3.weights'
confidence_threshold = 0.5
nms_threshold = 0.4
input_width = 416
input_height = 416

detector = Detector(cls_file, cfg_file, weight_file, confidence_threshold, nms_threshold, input_width, input_height)

img_path = 'images/person.jpg'
objects = detector.detect(img_path)
print(f'objects: {objects}')
"""