# calculate the accuracy of just the output from yolo model
# and the output from yolo+gcn

import torch
from gcn import GCN
from dataset import CocoDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'Results/model.pth'
num_classes = 80
train_path = '/home/user/Data/coco2014/train2014'
train_ann_file = '/home/user/Data/coco2014/annotations/instances_train2014.json'
val_path = '/home/user/Data/coco2014/val2014'
val_ann_file = '/home/user/Data/coco2014/annotations/instances_val2014.json'
train_pickle_file = 'train.pickle'
val_pickle_file = 'val.pickle'

adj = pickle.load(open('adj.pickle', 'rb'))
adj = np.float32(adj / np.max(adj) + np.identity(num_classes))
adj_tensor = torch.from_numpy(adj)

model = GCN(adj_tensor, num_classes, 80, num_classes)
model.load_state_dict(torch.load(model_path))

train_dataset = CocoDataset(train_path, train_ann_file, num_classes)
val_dataset = CocoDataset(val_path, val_ann_file, num_classes)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

train_detections = pickle.load(open(train_pickle_file, 'rb'))
val_detections = pickle.load(open(val_pickle_file, 'rb'))

total_train_images = len(train_loader)
total_val_images = len(val_loader)

model.eval()

print('Running...')
print('\n')

with torch.no_grad():
    correct_yolo = 0
    correct_gcn = 0
    total_instances = 0

    f1_yolo = []
    f1_gcn = []

    for img_path, label in train_loader:
        img_path = img_path[0]
        img_name = img_path.rsplit('/', 1)[1]
        class_ids = train_detections[img_name]

        y_true = np.array(label).squeeze()
        y_yolo = np.zeros(num_classes)
        y_gcn = np.zeros(num_classes)

        for class_id in class_ids:
            if class_id is not None:
                if label.T[class_id] == 1:
                    correct_yolo += 1
                    y_yolo[class_id] = 1

        input_vector = torch.zeros((1, num_classes))
        for class_id in class_ids:
            input_vector[0, class_id] = 1

        output = model(input_vector)
        predictions = torch.sigmoid(output)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        # concatenate output from yolo and gcn
        for class_id in class_ids:
            if class_id is not None:
                predictions.T[class_id] = 1
                y_gcn[class_id] = 1

        label = label.T
        predictions = predictions.T
        for idx in range(0, len(label)):
            if label[idx].item() == 1:
                if predictions[idx].item() == 1:
                    correct_gcn += 1
                    y_gcn[idx] = 1

                total_instances += 1

        f1_yolo_sample = f1_score(y_true, y_yolo, zero_division=1)
        f1_yolo.append(f1_yolo_sample)
        f1_gcn_sample = f1_score(y_true, y_gcn, zero_division=1)
        f1_gcn.append(f1_gcn_sample)

    gcn_acc = (correct_gcn / total_instances) * 100.0
    yolo_acc = (correct_yolo / total_instances) * 100.0

    f1_yolo = np.mean(f1_yolo)
    f1_gcn = np.mean(f1_gcn)

    print('Training set')
    print(f'total_instances: {int(total_instances)} ')
    print(f'yolo: {int(correct_yolo)}')
    print(f'gcn: {int(correct_gcn)}')

    print(f'yolo acc: {round(yolo_acc, 2)}')
    print(f'gcn acc: {round(gcn_acc, 2)}')
    print(f'improvement: {correct_gcn - correct_yolo}')

    print(f'f1 yolo: {np.round(f1_yolo, 2)}')
    print(f'f1 gcn: {np.round(f1_gcn, 2)}')

    correct_yolo = 0
    correct_gcn = 0
    total_instances = 0

    f1_yolo = []
    f1_gcn = []

    for img_path, label in val_loader:
        img_path = img_path[0]
        img_name = img_path.rsplit('/', 1)[1]
        class_ids = val_detections[img_name]

        y_true = np.array(label).squeeze()
        y_yolo = np.zeros(num_classes)
        y_gcn = np.zeros(num_classes)

        for class_id in class_ids:
            if class_id is not None:
                if label.T[class_id] == 1:
                    correct_yolo += 1
                    y_yolo[class_id] = 1

        input_vector = torch.zeros((1, num_classes))
        for class_id in class_ids:
            input_vector[0, class_id] = 1

        output = model(input_vector)
        predictions = torch.sigmoid(output)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        # concatenate output from yolo and gcn
        for class_id in class_ids:
            if class_id is not None:
                predictions.T[class_id] = 1
                y_gcn[class_id] = 1

        label = label.T
        predictions = predictions.T
        for idx in range(0, len(label)):
            if label[idx].item() == 1:
                if predictions[idx].item() == 1:
                    correct_gcn += 1
                    y_gcn[idx] = 1

                total_instances += 1

        f1_yolo_sample = f1_score(y_true, y_yolo, zero_division=1)
        f1_yolo.append(f1_yolo_sample)
        f1_gcn_sample = f1_score(y_true, y_gcn, zero_division=1)
        f1_gcn.append(f1_gcn_sample)

    gcn_acc = (correct_gcn / total_instances) * 100.0
    yolo_acc = (correct_gcn / total_instances) * 100.0

    f1_yolo = np.mean(f1_yolo)
    f1_gcn = np.mean(f1_gcn)

    print('\n')
    print('Validation set')
    print(f'total_instances: {int(total_instances)} ')
    print(f'yolo: {int(correct_yolo)}')
    print(f'gcn: {int(correct_gcn)}')

    print(f'yolo acc: {round(yolo_acc, 2)}')
    print(f'gcn acc: {round(gcn_acc, 2)}')
    print(f'improvement: {correct_gcn - correct_yolo}')

    print(f'f1 yolo: {np.round(f1_yolo, 2)}')
    print(f'f1 gcn: {np.round(f1_gcn, 2)}')
