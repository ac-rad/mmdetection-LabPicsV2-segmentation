import torch
from PIL import Image
from torchvision import datasets
import json
import os
import numpy as np
from tqdm import tqdm
import mmcv
import torch

from skimage import measure

class LabPicsHelper(datasets.VisionDataset):
    def __init__(self, root, source, classes=None, subclasses=None, train=True, load_subclasses=False):
        self.root = root
        self.source = source
        self.annotations = []
        self.classes = classes
        self.subclass = subclasses
        self.train = train
        self.load_subclasses = load_subclasses
        # self.datapath = self.root + ("/Train" if train else "/Eval")
        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(self.root):
            self.annotations.append(self.root+"/"+AnnDir)
        print("Total=" + str(len(self.annotations)))
        print("done making file list")

    def __getitem__(self, idx):
        data_path = self.annotations[idx]
        data = json.load(open(data_path + '/Data.json', 'r'))
        img_size = (Image.open(data_path + "/Image.jpg").convert("RGB")).size
        img_path = os.path.join(data_path, "Image.jpg")
        img = {"img_path": img_path, "img_size": img_size}
        labels = []
        sub_class = []
        masks = []
        boxes = []

        def _create_item(data_i, type):
            if data_i[type]:
                labels.append(self.classes[data_i[type][0]])
            else:
                print("Empty", type)
                print(data_i)
                print()
                return
            if self.load_subclasses:
                sub_label = np.zeros(len(self.subclass) + 1)
                for sub_cls in data_i[type]:
                    if sub_cls in self.subclass:
                        sub_label[self.subclass[sub_cls]] = 1
                sub_class.append(sub_label)
            mask = Image.open(data_path + data_i["MaskFilePath"])
            mask = np.array(mask)
            if len(mask.shape) == 3:
                mask = mask[:, :, -1]
            foreGround = mask > 0
            masks.append(foreGround)
            pos = np.where(foreGround)
            try:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            except:
                print(pos)
                print(data_path)

        if "Vessel" in self.source:
            for item in data["Vessels"].keys():
                _create_item(data["Vessels"][item], "VesselType_ClassNames")

        if "Material" in self.source:
            for item in data["MaterialsAndParts"].keys():
                if not (data["MaterialsAndParts"][item]["IsPart"]):
                     _create_item(data["MaterialsAndParts"][item], "MaterialType_ClassNames")

        valid_boxes = []
        valid_labels = []
        valid_masks = []
        for i, box in enumerate(boxes):
            if box[3] > box[1] and box[2] > box[0]:
                valid_boxes.append(box)
                valid_labels.append(labels[i])
                valid_masks.append(masks[i])
    
        if valid_boxes:
            valid_boxes = np.array(valid_boxes, dtype=np.float32)
            valid_labels = np.array(valid_labels, dtype=np.int64)
            valid_masks = np.stack(valid_masks, axis=0)
            valid_masks = valid_masks.astype(np.float32)
        else:
            valid_boxes = np.zeros((0, 4), dtype=np.float32)
            valid_labels = np.array([], dtype=np.int64)
            valid_masks = np.zeros((0, img_size[1], img_size[0]), dtype=np.float32)

        target = {
            "boxes": valid_boxes,
            "labels": valid_labels,
            "masks": valid_masks,
            "image_id": torch.tensor([idx]),
            "sub_cls": sub_class,
        }

        return img, target

    def __len__(self):
        return len(self.annotations)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    # check if the mask is truely binary
    assert(np.array_equal(np.unique(binary_mask), [0, 1]))
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = [np.subtract(contour, 1) for contour in contours] # fix for ValueError
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def create_coco_style_json(train=True):
    # labpics_classes = {"Vessel": 1, "Liquid": 2, "Cork": 0, "Solid": 2, "Part": 0, "Foam": 2, "Gel": 2, "Label": 0, "Vapor":2, "Other Material":2}
    labpics_classes = {"Vessel": 1, "Syringe": 1, "Pippete": 1, "Tube": 1, "IVBag": 1, "DripChamber": 1, "IVBottle": 1,
     "Beaker": 1, "RoundFlask": 1, "Cylinder": 1, "SeparatoryFunnel": 1, "Funnel": 1, "Burete": 1,
     "ChromatographyColumn": 1, "Condenser": 1, "Bottle": 1, "Jar": 1, "Connector": 1, "Flask": 1,
     "Cup": 1, "Bowl": 1, "Erlenmeyer": 1, "Vial": 1, "Dish": 1, "HeatingVessel": 1, "Transparent": 0,
     "SemiTrans": 0, "Opaque": 0, "Cork": 0, "Label": 0, "Part": 0, "Spike": 0, "Valve": 0, "DisturbeView": 0,
     "Liquid": 2, "Foam": 2, "Suspension": 2, "Solid": 2, "Filled": 2, "Powder": 2, "Urine": 2, "Blood": 2,
     "MaterialOnSurface": 0, "MaterialScattered": 0, "PropertiesMaterialInsideImmersed": 0,
     "PropertiesMaterialInFront": 0, "Gel": 2, "Granular": 2, "SolidLargChunk": 2, "Vapor": 2,
     "Other Material": 2, "VesselInsideVessel": 0, "VesselLinked": 0, "PartInsideVessel": 0,
     "SolidIncludingParts": 0, "MagneticStirer": 0, "Thermometer": 0, "Spatula": 0, "Holder": 0,
     "Filter": 0, "PipeTubeStraw": 0}

    dataset_root = "/home/alexliu/Dev/LabPicV2_Dataset/Chemistry"
    if train:
        dataset_root = os.path.join(dataset_root, "Train")
    else:
        dataset_root = os.path.join(dataset_root, "Eval")

    lp_helper_dataset = LabPicsHelper(dataset_root, ["Vessel", "Material"], classes=labpics_classes)

    dataset = {
    'images': [],
    'annotations': [],
    'categories': []
    }
    category_names = ['Vessel', 'Material']
    category_ids = [1, 2]

    for i, category_name in enumerate(category_names):
        dataset['categories'].append({
            'id': category_ids[i],
            'name': category_name,
        })

    for i in tqdm(range(len(lp_helper_dataset)//3), mininterval=1):
        img, ann = lp_helper_dataset.__getitem__(i)
        img_path = img['img_path']
        img_width, img_height = img['img_size']
        img_id = len(dataset['images']) + 1
        dataset['images'].append({
            'id': img_id,
            'file_name': img_path,
            'width': img_width,
            'height': img_height,
        })
        for j, box in enumerate(ann['boxes']):
            xmin, ymin, xmax, ymax = box
            polygons = binary_mask_to_polygon(ann['masks'][j])
            dataset['annotations'].append({
                'id': len(dataset['annotations']) + 1,
                'image_id': img_id,
                'category_id': ann['labels'][j],
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'area': (xmax - xmin) * (ymax - ymin),
                'segmentation': polygons,
                'iscrowd': 0,
            })

    print("Finished creating {} dataset".format("train" if train else "val"))
    file_name = 'train.json' if train else 'val_reduced.json'
    mmcv.dump(dataset, os.path.join(dataset_root, file_name))


def filter_large_images(data_json_path):
    with open(data_json_path, 'r') as f:
        dataset = json.load(f)
    image_ids = set()
    total_removed = 0
    total_original = len(dataset['images'])
    for img in dataset['images']:
        if img['width'] * img['height'] > 2500 * 2500:
            image_ids.add(img['id'])
            print("Removing image {}".format(img['file_name']))
            dataset['images'].remove(img)
            total_removed += 1
    for ann in dataset['annotations']:
        if ann['image_id'] in image_ids:
            dataset['annotations'].remove(ann)
    new_file_name = data_json_path.split('.')[0] + '_filtered.json'
    mmcv.dump(dataset, new_file_name)
    print("Finished filtering large images, removed {} images out of {}".format(total_removed, total_original))

if __name__ == "__main__":
    # create_coco_style_json(train=False)
    filter_large_images("/home/alexliu/Dev/LabPicV2_Dataset/Chemistry/Eval/val.json")