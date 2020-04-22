# calculate the accuracy of just the output from yolo model
# and the output from yolo+gcn

import torch
from gcn import GCN
import torch.optim as optim
import torch.nn as nn
from dataset import CocoDataset
from torch.utils.data import DataLoader
import pickle
from utils import generate_adjacency_matrix
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'Results/model.pth'
num_classes = 80
train_path = '/home/user/Data/coco2014/train2014'
train_ann_file = '/home/user/Data/coco2014/annotations/instances_train2014.json'
val_path = '/home/user/Data/coco2014/val2014'
val_ann_file = '/home/user/Data/coco2014/annotations/instances_val2014.json'
adj = pickle.load(open('data/coco_adj.pkl', 'rb'))
train_pickle_file = 'train.pickle'
val_pickle_file = 'val.pickle'

adj = np.float32(generate_adjacency_matrix(adj))
adj_tensor = torch.from_numpy(adj)

model = GCN(adj_tensor, num_classes, 80, num_classes)
model.load_state_dict(torch.load(model_path))

train_dataset = CocoDataset(train_path, train_ann_file, num_classes)
val_dataset = CocoDataset(val_path, val_ann_file, num_classes)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

train_detections = pickle.load(open(train_pickle_file, 'rb'))
val_detections = pickle.load(open(val_pickle_file, 'rb'))

total_train_images = len(train_loader)
total_val_images = len(val_loader)

model.eval()
with torch.no_grad():

    correct_yolo = 0
    correct_gcn = 0
    total_instances = 0

    for img_path, label in train_loader:
        img_path = img_path[0]
        img_name = img_path.rsplit('/', 1)[1]
        class_ids = train_detections[img_name]
        input_vector = torch.zeros((1, num_classes))
        for class_id in class_ids:
            input_vector[0, class_id] = 1

        output = model(input_vector)
        predictions = torch.sigmoid(output)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        label = label.T
        predictions = predictions.T
        for idx in range(0, len(label)):
            if label[idx].item() == 1:
                if predictions[idx].item() == 1:
                    correct_gcn += 1

                total_instances += 1

    train_acc_gcn = (correct_gcn / total_instances) * 100.0

    print(f'Train Acc: {round(train_acc_gcn, 2)}')
    print(f'Train total_instances: {int(total_instances)} correct_instalces: {int(train_acc_gcn)}')

    correct_yolo = 0
    correct_gcn = 0
    total_instances = 0
    for img_path, label in val_loader:
        img_path = img_path[0]
        img_name = img_path.rsplit('/', 1)[1]
        class_ids = val_detections[img_name]
        input_vector = torch.zeros((1, num_classes))
        for class_id in class_ids:
            input_vector[0, class_id] = 1

        output = model(input_vector)
        loss = criterion(output, label)

        predictions = torch.sigmoid(output)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        label = label.T
        predictions = predictions.T
        for idx in range(0, len(label)):
            if label[idx].item() == 1:
                if predictions[idx].item() == 1:
                    correct_gcn += 1

                total_instances += 1

    val_acc_gcn = (correct_gcn / total_instances) * 100.0

    print(f'Val Acc: {round(val_acc_gcn, 2)}')
    print(f'Train total_instances: {int(total_instances)} correct_instalces: {int(correct_gcn)}')
