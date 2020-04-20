import torch
from gcn import GCN
from detector import Detector
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from dataset import CocoDataset
from torch.utils.data import DataLoader
from utils import get_class_id_from_objects
import pickle
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 80
num_epochs = 5
batch_size = 1
results_folder = 'Results'

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
adj = pickle.load(open('data/coco_adj.pkl', 'rb'))
adj = np.float32(adj['adj'])    # numpy ndarray
adj_tensor = torch.from_numpy(adj)

detector = Detector(cls_file, cfg_file, weight_file, confidence_threshold, nms_threshold, input_width, input_height)
model = GCN(adj_tensor, num_classes, 1024, 1)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

checkpoint_path = results_folder + '/' + 'model.pth'

train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

train_dataset = CocoDataset(train_path, train_ann_file, num_classes)
val_dataset = CocoDataset(val_path, val_ann_file, num_classes)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    model.train()
    for img_path, label in train_loader:
        img_path = img_path[0]
        objects = detector.detect(img_path)
        class_ids = get_class_id_from_objects(objects)
        input_vector = torch.zeros(num_classes, num_classes)
        for class_id in class_ids:
            input_vector[class_id] = 1.

        optimizer.zero_grad()
        output = model(input_vector)
        output = output.T
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    # evaulate on val dataset
    val_acc = 0
    val_images = 0
    model.eval()
    with torch.no_grad():
        for img_path, label in val_loader:
            img_path = img_path[0]
            objects = detector.detect(img_path)
            class_ids = get_class_id_from_objects(objects)
            input_vector = torch.zeros(num_classes, num_classes)
            for class_id in class_ids:
                input_vector[class_id] = 1.

            output = model(input_vector)
            if torch.sigmoid(output.T) == label:
                val_acc += 1

            val_images += 1
    val_acc = (val_acc / val_images) * 100.0
    print(f'Epoch: {epoch + 1}/{num_epochs}, Acc: {val_acc}')

"""
x = list(range(num_epochs))
plt.subplot(121)
plt.plot(x, train_acc_list, label='train_acc_label')
plt.plot(x, val_acc_list, label='test_acc_label')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(x, train_loss_list, label='train_loss_label')
plt.plot(x, val_loss_list, label='test_loss_label')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
"""