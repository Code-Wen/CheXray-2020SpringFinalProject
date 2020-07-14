#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:22:24 2020

@author: cwen
"""

import os
import numpy as np
import time
import sys
import csv
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision as tv
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import random


class HeatmapGenerator ():

    def __init__(self, model, device, num_classes=14, crop_size=224):

        self.model = model
        self.device = device
        self.model.eval()

        # Initialize the weights
        self.weights = list(self.model.densenet121.features.parameters())[-2]

        # Initialize the image transform
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize((crop_size, crop_size)))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transformSequence = transforms.Compose(transformList)

    # --------------------------------------------------------------------------------

    def generate(self, input_path, output_path, crop_size=224):
        """
        input_path - path for the input image.
        output_path - path for the output.
        crop_size - default 224.
        """
        # Load image, transform, convert
        with torch.no_grad():

            image_data = Image.open(input_path).convert('RGB')
            image_data = self.transformSequence(image_data)
            image_data = image_data.unsqueeze_(0)

            # Class names
            class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                           'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                           'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

            if self.device.type == 'cuda':
                image_data = image_data.cuda()
            l = self.model(image_data)
            output = self.model.densenet121.features(image_data)
            # label = class_names[torch.max(l, 1)[1]]
            # Generate heatmap
            heatmap = None
            for i in range(0, len(self.weights)):
                map = output[0, i, :, :]
                if i == 0:
                    heatmap = self.weights[i] * map
                else:
                    heatmap += self.weights[i] * map
                heatmap_np = heatmap.cpu().data.numpy()

        # Blend original and heatmap

        img_original = cv2.imread(input_path, 1)
        img_original = cv2.resize(img_original, (crop_size, crop_size))

        cam = heatmap_np / np.max(heatmap_np)
        cam = cv2.resize(cam, (crop_size, crop_size))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

        img = cv2.addWeighted(img_original, 1, heatmap, 0.35, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        # plt.title(label)
        axs[0].imshow(img_original)
        axs[1].imshow(img)
        plt.plot()
        plt.axis('off')
        plt.savefig(output_path)
        plt.show()
