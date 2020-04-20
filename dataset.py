from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torch


class CocoDataset(Dataset):
    def __init__(self, folder_path, ann_file, num_labels, transform=None):
        self.folder_path = folder_path
        self.annFile = ann_file
        self.transform = transform
        self.num_labels = num_labels

        self.coco = COCO(self.annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.categoryIdToClassIndex = {}
        self.classIndexToCategoryId = {}

        categoryid_list = self.coco.getCatIds()

        for idx, item in enumerate(categoryid_list):
            self.categoryIdToClassIndex[item] = idx
            self.classIndexToCategoryId[idx] = item

    def get_class_id(self, category_id):
        return self.categoryIdToClassIndex[category_id]

    def get_category_id(self, class_id):
        return self.classIndexToCategoryId[class_id]

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.folder_path + '/' + img_info['file_name']
        # img = Image.open(img_path).convert('RGB')
        # if self.transform is not None:
        #    img = self.transform(img)

        label = torch.zeros(self.num_labels)
        for idx in range(len(anns)):
            category_id = anns[idx]['category_id']
            class_id = self.get_class_id(category_id)
            label[class_id] = 1

        # return img, label
        return img_path, label

    def __len__(self):
        return len(self.ids)
