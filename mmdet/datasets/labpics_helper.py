import torch
from PIL import Image
from torchvision import datasets
import json
import os
import numpy as np

class LabPicsHelper(datasets.VisionDataset):
    def __init__(self, root, source, classes=None, subclasses=None, train=True, load_subclasses=False):
        self.root = root
        self.source = source
        self.annotations = []
        self.classes = classes
        self.subclass = subclasses
        self.train = train
        self.load_subclasses = load_subclasses
        self.datapath = self.root + ("/Train" if train else "/Eval")
        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(self.datapath):
            self.annotations.append(self.datapath+"/"+AnnDir)
        print("Total=" + str(len(self.annotations)))
        print("done making file list")

    def __getitem__(self, idx):
        data_path = self.annotations[idx]
        data = json.load(open(data_path + '/Data.json', 'r'))
        img_size = (Image.open(data_path + "/Image.jpg").convert("RGB")).size
        img_path = os.path.join(data_path, "Image.jpg")
        img = {"img_path": img_path, "img_size": img_size}
        num_objs = 0
        labels = []
        sub_class = []
        masks = []
        boxes = []

        def _create_item(data_i, type):
            try:
                labels.append(self.classes[data_i[type][0]])
            except:
                print(data_i)
                print(type)
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
            num_objs += len(data["Vessels"])
            for item in data["Vessels"].keys():
                _create_item(data["Vessels"][item], "VesselType_ClassNames")

        if "Material" in self.source:
            num_objs += len(data["MaterialsAndParts"])
            for item in data["MaterialsAndParts"].keys():
                if not (data["MaterialsAndParts"][item]["IsPart"] or data["MaterialsAndParts"][item]["IsOnSurface"] or data["MaterialsAndParts"][item]['IsScattered'] or data["MaterialsAndParts"][item]['IsFullSegmentableMaterialPhase']):
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

if __name__ == "__main__":
    pass
    # data_dir = "/home/alexliu/Dev/LabPicV2_Dataset"
    # dataset = LabPicsHelper(os.path.join(data_dir, "Chemistry"), ["Vessel, Material"], classes=labpics_classes)
    
    # # Loop through all images in dataset, create a list of dicts called data_infos, where each dict
    # # contains the image path, width, height, and annotations. dataset is a torch.utils.data.Dataset
    # # object, so we can use the __getitem__ method to get the image and annotations.
    # data_infos = []    
    # for i in range(len(dataset)):
    #     # Add a progress bar for loading the annotations
    #     if i % 100 == 0:
    #         print(f"Loading annotations: {i}/{len(dataset)}")
    #     img, ann = dataset.__getitem__(i)
    #     img_path = img['img_path']
    #     img_width, img_height = img['img_size']
    #     ann_dict = {
    #         'filename': img_path,
    #         'width': img_width,
    #         'height': img_height,
    #         'ann': {
    #             'bboxes': ann['boxes'],
    #             'labels': ann['labels'],
    #         }
    #     }
    #     data_infos.append(ann_dict)
    
    # # Save the data_infos list to a json file
    # with open(os.path.join(data_dir, 'data_infos.json'), 'w') as f:
    #     json.dump(data_infos, f)