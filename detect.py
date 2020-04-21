from detector import Detector
from dataset import CocoDataset
from torch.utils.data import DataLoader
from utils import get_class_id_from_objects
import pickle

num_classes = 80

train_pickle = 'train.pickle'
val_pickle = 'val.pickle'

cls_file = 'data/coco.names'
cfg_file = 'data/yolov3.cfg'
weight_file = 'data/yolov3.weights'

confidence_threshold = 0.5
nms_threshold = 0.4
input_width = 416
input_height = 416

train_path = '/home/user/Data/coco2014/train2014'
train_ann_file = '/home/user/Data/coco2014/annotations/instances_train2014.json'

val_path = '/home/user/Data/coco2014/val2014'
val_ann_file = '/home/user/Data/coco2014/annotations/instances_val2014.json'

detector = Detector(cls_file, cfg_file, weight_file, confidence_threshold, nms_threshold, input_width, input_height)

train_dataset = CocoDataset(train_path, train_ann_file, num_classes)
val_dataset = CocoDataset(val_path, val_ann_file, num_classes)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

train_detections = {}
val_detections = {}

print('Running...')

for img_path, label in train_loader:
    img_path = img_path[0]
    img_name = img_path.rsplit('/', 1)[1]
    objects = detector.detect(img_path)
    class_ids = get_class_id_from_objects(objects)
    class_ids = list(set(class_ids))
    train_detections[img_name] = class_ids

for img_path, label in val_loader:
    img_path = img_path[0]
    img_name = img_path.rsplit('/', 1)[1]
    objects = detector.detect(img_path)
    class_ids = get_class_id_from_objects(objects)
    class_ids = list(set(class_ids))
    val_detections[img_name] = class_ids

with open(train_pickle, 'wb') as train_pickle_file:
    pickle.dump(train_detections, train_pickle_file)

with open(val_pickle, 'wb') as val_pickle_file:
    pickle.dump(val_detections, val_pickle_file)


print('Done')
