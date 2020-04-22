from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt

train_path = '/home/user/Data/coco2014/train2014'
train_ann_file = '/home/user/Data/coco2014/annotations/instances_train2014.json'
val_path = '/home/user/Data/coco2014/val2014'
val_ann_file = '/home/user/Data/coco2014/annotations/instances_val2014.json'

coco = COCO(train_ann_file)

categories = coco.loadCats(coco.getCatIds())
names = [cat['name'] for cat in categories]
supercategory = set([cat['supercategory'] for cat in categories])
catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard', 'car'])
# print(f'catIds: {catIds}')
imgIds = coco.getImgIds(catIds=catIds)
# print(f'imgIds: {imgIds}')

# segmentation plot
imgObj = coco.loadImgs(imgIds)
img_path = val_path + '/' + imgObj[0]['file_name']
img_id = imgObj[0]['id']
# print(f'img_path: {img_path}')
# print(f'img_id: {img_id}')

img = io.imread(img_path)
annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
# print(f'annIds: {annIds}')
anns = coco.loadAnns(annIds)
# print(f'anns: {anns}')
plt.imshow(img)
plt.axis('off')

coco.showAnns(anns)
plt.show()


coco = COCO(train_ann_file)
train_images = list(sorted(coco.imgs.keys()))
imageId = train_images[1000]
annIds = coco.getAnnIds(imageId)
cats = coco.loadCats(coco.getCatIds())
cocoNames = [cat['name'] for cat in cats]
# print('traffic light' in cocoNames)

# print(annIds)
anns = coco.loadAnns(annIds)

categories = set()
for idx in range(len(anns)):
    categoryId = anns[idx]['category_id']
    category = coco.loadCats(categoryId)[0]['name']
    categories.add(category)

print(categories)
img_info = coco.loadImgs(imageId)
img_path = train_path + '/' + img_info[0]['file_name']
img = io.imread(img_path)
plt.imshow(img)
coco.showAnns(anns)
plt.show()
