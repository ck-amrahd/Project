# test file for what's inside pickle files.

import pickle

# adj = pickle.load(open('data/coco_adj.pkl', 'rb'))
# adj is a dict with two key - value pairs
# adj['nums'] and adj['adj']
# adj['nums'] - ndarray of shape (80)
# adj['adj'] - ndarray of shape (80, 80) is the adjancency matrix

# print(adj['nums'])
# print(adj['adj'])

# test the train.pickle and val.pickle file
# train.pickle is a dictionary with image_name as the key and class_ids [objects] present in that image as the value
train_detections = pickle.load(open('train.pickle', 'rb'))
print(len(train_detections))    # should output 82783 - number of training images

val_detections = pickle.load(open('val.pickle', 'rb'))
print(len(val_detections))      # should pring 40504 - number of validation images
