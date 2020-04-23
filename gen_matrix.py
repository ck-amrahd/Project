# code to generate the co-occurrence matrix
from dataset import CocoDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np

num_classes = 80
train_path = '/home/user/Data/coco2014/train2014'
train_ann_file = '/home/user/Data/coco2014/annotations/instances_train2014.json'
adj_file = 'adj.pickle'

train_dataset = CocoDataset(train_path, train_ann_file, num_classes)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
total_train_images = len(train_loader)

adj = np.zeros((num_classes, num_classes))

print('Generating Matrix...')

for _, label in train_loader:
    class_ids = []
    label = label.squeeze()
    label = label.tolist()
    for idx, item in enumerate(label):
        if item == 1:
            class_ids.append(idx)

    for i in range(0, len(class_ids) - 1):
        for j in range(i + 1, len(class_ids)):
            row = class_ids[i]
            column = class_ids[j]

            adj[row, column] += 1
            adj[column, row] += 1

print(adj)

with open(adj_file, 'wb') as write_file:
    pickle.dump(adj, write_file)

print('Done')
