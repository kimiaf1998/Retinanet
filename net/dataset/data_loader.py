from __future__ import print_function, division

import os
import random

import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from net.utility.constants import DATASET_MODEL, DATASET_PATH, NUMBER_OF_CLASSES


class CocoDataset(Dataset):
    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = DATASET_PATH
        self.transform = transform
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + DATASET_MODEL + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', DATASET_MODEL, image_info['file_name'])
        img = skimage.io.imread(path)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get GTB annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return NUMBER_OF_CLASSES


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)
    max_width = np.array(widths).max()
    max_height = np.array(heights).max()
    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]
