# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import time
from io import BytesIO
import base64
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000000
from tqdm import tqdm
import glob
import os
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


train_transform = A.Compose([
    # reszie
    A.Resize(256, 256),
    #
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5)
    ]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
import os
import cv2
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
import glob
from torch.utils.data import Dataset

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class RSCDataset(Dataset):
    def __init__(self, imgs_dir, transform=None):
        generate = glob.glob(os.path.join(imgs_dir, '*.tif'))
        self.file_list = [item for item in list(generate)][:300]
        self.label_list = [item.replace('tif', 'png') for item in self.file_list][:300]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            img_trans = img_nd.transpose(2, 0, 1)
        except:
            print(img_nd.shape)
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        img_file = self.file_list[i]
        mask_file = self.label_list[i]
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        # if self.transform:
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        return {
            'image': image,
            'label': mask.long()
        }


import segmentation_models_pytorch as smp


def load_model(model_path):
    from codes.Mymodels.segmentation_models import segmentation_models
    model_entity = dict(
        model_name='segmentation_models_pytorch',
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=10,
    )
    model = segmentation_models(model_entity)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path)
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    # model_1 = model_1.to(device)
    # model_1.load_state_dict(checkpoint['b4_state_dict']['state_dict'])

    model.eval()
    return model


#
def cal_metrics(pred_label, gt):
    def _generate_matrix(gt_image, pre_image, num_class=10):
        mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        # print(mask)
        label = num_class * gt_image[mask].astype('int') + pre_image[mask]
        # print(label.shape)
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=num_class ** 2)
        confusion_matrix = count.reshape(num_class, num_class)  # 21 * 21(for pascal)
        # print(confusion_matrix.shape)
        return confusion_matrix

    def _Class_IOU(confusion_matrix):
        MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        return MIoU

    confusion_matrix = _generate_matrix(gt.astype(np.int8), pred_label.astype(np.int8))
    miou = _Class_IOU(confusion_matrix)
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return miou, acc


#
if __name__ == "__main__":
    iou = IOUMetric(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/home/xjz/Desktop/Coding/PycharmProjects/competition/tianchi/LishuiYaogan_Sementation/model_weights/2021_0224_20_2441_segmentation_models_pytorch/fold_1/checkpoints/best.ckpt'
    model = load_model(model_path)

    data_dir = "/home/xjz/Desktop/Coding/DL_Data/LishuiYaogan/train_images_256_256"


    valid_data = RSCDataset(data_dir, transform=val_transform)
    valid_loader = DataLoader(dataset=valid_data, batch_size=16, shuffle=False, num_workers=1)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_samples in tqdm(enumerate(valid_loader)):
            data, target = batch_samples['image'], batch_samples['label']
            print(data.shape, target.shape)
            data = Variable(data.to(device))
            pred = model(data)
            pred = pred.cpu().data.numpy()
            target = target.numpy()
            pred = np.argmax(pred, axis=1)
            iou.add_batch(pred, target)

    #
    acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
    print(iu)
    print(mean_iu)
