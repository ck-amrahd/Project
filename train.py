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
num_epochs = 10
results_folder = 'Results'

train_path = '/home/user/Data/coco2014/train2014'
train_ann_file = '/home/user/Data/coco2014/annotations/instances_train2014.json'
val_path = '/home/user/Data/coco2014/val2014'
val_ann_file = '/home/user/Data/coco2014/annotations/instances_val2014.json'

adj = pickle.load(open('adj.pickle', 'rb'))
adj = np.float32(adj / np.max(adj) + np.identity(num_classes))

train_pickle_file = 'train.pickle'
val_pickle_file = 'val.pickle'

adj_tensor = torch.from_numpy(adj)

model = GCN(adj_tensor, num_classes, 80, num_classes)

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

total_train_images = len(train_loader)
total_val_images = len(val_loader)

print('\n')
print(f'Total train images: {total_train_images}')
print(f'Total validation images: {total_val_images}')
print('\n')

print('Training...')
print('-' * 100)

train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    model.train()

    train_loss = 0.0
    # train_correct = 0.0

    total_instances = 0.0
    correct_instances = 0.0

    for img_path, label in train_loader:
        img_path = img_path[0]
        img_name = img_path.rsplit('/', 1)[1]
        class_ids = train_detections[img_name]
        input_vector = torch.zeros((1, num_classes))
        for class_id in class_ids:
            input_vector[0, class_id] = 1

        optimizer.zero_grad()
        output = model(input_vector)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = torch.sigmoid(output)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        # if (predictions == label).sum().item() == len(label):
        #    train_correct += 1

        # if it's able to detect one instance, then gets a value of 1
        label = label.T
        predictions = predictions.T
        for idx in range(0, len(label)):
            if label[idx].item() == 1:
                if predictions[idx].item() == 1:
                    correct_instances += 1

                total_instances += 1

    train_loss = train_loss / total_train_images
    train_loss_list.append(train_loss)
    train_acc = (correct_instances / total_instances) * 100.0
    train_acc_list.append(train_acc)

    print(f'Epoch: {epoch + 1} / {num_epochs}')
    print(f'Train Loss: {round(train_loss, 2)} Acc: {round(train_acc, 2)}')
    print(f'Train total_instances: {int(total_instances)} correct_instalces: {int(correct_instances)}')

    # evaulate on val dataset
    val_loss = 0.0
    # val_correct = 0.0
    correct_instances = 0.0
    total_instances = 0.0
    model.eval()
    with torch.no_grad():
        for img_path, label in val_loader:
            img_path = img_path[0]
            img_name = img_path.rsplit('/', 1)[1]
            class_ids = val_detections[img_name]
            input_vector = torch.zeros((1, num_classes))
            for class_id in class_ids:
                input_vector[0, class_id] = 1

            output = model(input_vector)
            loss = criterion(output, label)
            val_loss += loss.item()

            predictions = torch.sigmoid(output)
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0

            # if (predictions == label).sum().item() == torch.sum(label).item():
            #   val_correct += 1

            # if it's able to detect one instance, then gets a value of 1
            label = label.T
            predictions = predictions.T
            for idx in range(0, len(label)):
                if label[idx].item() == 1:
                    if predictions[idx].item() == 1:
                        correct_instances += 1

                    total_instances += 1

    val_loss = val_loss / total_val_images
    val_loss_list.append(val_loss)
    val_acc = (correct_instances / total_instances) * 100.0
    val_acc_list.append(val_acc)

    print(f'Val total_instances: {int(total_instances)} correct_instalces: {int(correct_instances)}')
    print(f'Val Loss: {round(val_loss, 2)}, Acc: {round(val_acc, 2)}')
    print('-' * 100)

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
