import numpy as np
import torch
from pycocotools.coco import COCO

train_ann_file = '/home/user/Data/coco2014/annotations/instances_train2014.json'
coco = COCO(train_ann_file)
cats = coco.loadCats(coco.getCatIds())


class Category:
    def __init__(self, class_index, super_category, cat_id, nm):
        self.class_id = class_index
        self.supercategory = super_category
        self.category_id = cat_id
        self.class_name = nm


categories = []
for idx, item in enumerate(cats):
    supercategory = item['supercategory']
    category_id = item['id']
    name = item['name']
    category = Category(idx, supercategory, category_id, name)
    categories.append(category)


def get_class_id_from_category_id(cat_id):
    for cat in categories:
        if cat.category_id == cat_id:
            return cat.class_id
        else:
            continue


def get_category_id_from_class_id(class_id):
    for cat in categories:
        if cat.class_id == class_id:
            return cat.category_id
        else:
            continue


def get_class_name_from_category_id(cat_id):
    for cat in categories:
        if cat.category_id == cat_id:
            return cat.class_name
        else:
            continue


def get_class_name_from_class_id(class_id):
    return categories[class_id].class_name


def get_class_id_from_class_name(class_name):
    for cat in categories:
        if cat.class_name == class_name:
            return cat.class_id
        else:
            continue


def get_classes_from_labels(labels):
    class_names = []
    label_size = labels.shape[0]
    for label in range(label_size):
        if int(labels[label]) == 1:
            class_name = get_class_name_from_class_id(label)
            class_names.append(class_name)

    return class_names


def get_class_id_from_objects(objects):
    class_ids = []
    for obj in objects:
        class_id = get_class_id_from_class_name(obj)
        class_ids.append(class_id)

    return class_ids


def prepare_image(np_img):
    np_img = np.float32(np_img)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # convert to [C, H, W] from [H, W, C]
    np_img = np_img.transpose(2, 0, 1)
    channels, width, height = np_img.shape
    for channel in range(channels):
        np_img[channel] /= 255
        np_img[channel] -= mean[channel]
        np_img[channel] /= std[channel]

    return np_img


def tensor_to_image(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img.astype('float32')
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    # img = np.clip(img, 0, 1)
    return img

