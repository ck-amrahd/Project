import torch
from gcn import GCN
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from dataset import CocoDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 80
num_epochs = 3
results_folder = 'Results'

train_path = '/home/user/Data/coco2014/train2014'
train_ann_file = '/home/user/Data/coco2014/annotations/instances_train2014.json'
val_path = '/home/user/Data/coco2014/val2014'
val_ann_file = '/home/user/Data/coco2014/annotations/instances_val2014.json'
adj = pickle.load(open('data/coco_adj.pkl', 'rb'))
train_pickle_file = 'train.pickle'
val_pickle_file = 'val.pickle'

adj = np.float32(adj['adj'])  # numpy ndarray
adj_tensor = torch.from_numpy(adj)

model = GCN(adj_tensor, num_classes, 1024, 1)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

checkpoint_path = results_folder + '/' + 'model.pth'

train_dataset = CocoDataset(train_path, train_ann_file, num_classes)
val_dataset = CocoDataset(val_path, val_ann_file, num_classes)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

train_detections = pickle.load(open(train_pickle_file, 'rb'))
val_detections = pickle.load(open(val_pickle_file, 'rb'))

print('Training...')
print('-' * 100)

total_train_images = len(train_loader)
total_val_images = len(val_loader)

train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    model.train()

    train_loss = 0.0
    train_correct = 0.0

    for img_path, label in train_loader:
        img_path = img_path[0]
        img_name = img_path.rsplit('/', 1)[1]
        class_ids = train_detections[img_name]
        input_vector = torch.zeros(num_classes, num_classes)
        for class_id in class_ids:
            input_vector[class_id] = 1.

        optimizer.zero_grad()
        output = model(input_vector)
        loss = criterion(output.T, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = torch.sigmoid(output.T)
        predictions[predictions >= 0.5] = 1.
        predictions[predictions < 0.5] = 0.

        if (predictions == label).sum().item() == len(label):
            train_correct += 1

    train_loss = train_loss / total_train_images
    train_loss_list.append(train_loss)
    train_acc = (train_correct / total_train_images) * 100.0
    train_acc_list.append(train_acc)

    print(f'Epoch: {epoch + 1} / {num_epochs}')
    print(f'Train Loss: {round(train_loss, 2)} Acc: {round(train_acc, 2)}')

    # evaulate on val dataset
    val_loss = 0.0
    val_correct = 0.0
    model.eval()
    with torch.no_grad():
        for img_path, label in val_loader:
            img_path = img_path[0]
            img_name = img_path.rsplit('/', 1)[1]
            class_ids = val_detections[img_name]
            input_vector = torch.zeros(num_classes, num_classes)
            for class_id in class_ids:
                input_vector[class_id] = 1.

            output = model(input_vector)
            loss = criterion(output.T, label)
            predictions = torch.sigmoid(output.T)
            predictions[predictions >= 0.5] = 1.
            predictions[predictions < 0.5] = 0.

            # predict correct if 3 of the labels are correct
            if (predictions == label).sum().item() == len(label):
                val_correct += 1

    val_loss = val_loss / total_val_images
    val_loss_list.append(val_loss)
    val_acc = (val_correct / total_val_images) * 100.0
    val_acc_list.append(val_acc)

    print(f'Val Loss: {round(val_loss, 2)}, Acc: {round(val_acc, 2)}')
    print('-' * 50)

torch.save(model.state_dict(), results_folder + '/' + 'model.pth')

x = list(range(num_epochs))
plt.subplot(121)
plt.plot(x, train_acc_list, label='train_acc')
plt.plot(x, val_acc_list, label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(x, train_loss_list, label='train_loss')
plt.plot(x, val_loss_list, label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
