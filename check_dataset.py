from dataset import CocoDataset
from torchvision import transforms
import torch
from utils import get_classes_from_labels
from utils import tensor_to_image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

train_path = '/home/user/Data/coco2014/train2014'
train_ann_file = '/home/user/Data/coco2014/annotations/instances_train2014.json'
val_path = '/home/user/Data/coco2014/val2014'
val_ann_file = '/home/user/Data/coco2014/annotations/instances_val2014.json'
coco = COCO(train_ann_file)

num_labels = 80

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = CocoDataset(train_path, train_ann_file, transform, num_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
images, labels = next(iter(train_loader))

print(f'images.shape: {images.shape}')
print(labels.shape)
img = tensor_to_image(images[0])
class_names = get_classes_from_labels(labels[0])
print(class_names)

plt.imshow(img)
plt.show()
