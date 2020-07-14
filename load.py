from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from PIL import Image

# https://www.med-ed.virginia.edu/courses/rad/cxr/technique3chest.html
# "AP views are less useful"

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Implements torch.utils.data.Dataset
class CheXpertTorchDataset(Dataset):
    def __init__(self, labels_csv, transform = None):
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        self.image_paths = []
        self.patient_ids = []
        self.studies = []
        self.clinical_info = [] # list of tuples (age, sex)
        self.labels = [] # list of lists
        for index, row in self.labels_df.iterrows():
            image_path = row['Path']
            patient_id = row['Path'].split("/")[2]
            study = row['Path'].split("/")[3]
            clin_info = (row['Age'], row['Sex'])
            # Column index 5 and onwards are the labels. [No Finding ... Support Devices]
            label = row[5:]
            # Treat positive (1) and uncertain (-1) as positive
            # Treat blank as negative
            for i in range(len(label)):
                label[i] = float(label[i])
                if not math.isnan(label[i]):
                    labelFloat = float(label[i])
                    if labelFloat == -1.0:
                        # The value we assign here can be parameterized.
                        label[i] = 1.0
                    else:
                        # labelFloat is 0.0 or 1.0
                        label[i] = labelFloat
                else:
                    # label[i] is undefined (blank), treat as negative.
                    label[i] = 0.0
            self.image_paths.append(image_path)
            self.patient_ids.append(patient_id)
            self.studies.append(study)
            self.clinical_info.append(clin_info)
            self.labels.append(label)

    # Implements torch.utils.data.Dataset
    def __len__(self):
        return len(self.labels_df)

    # Implements torch.utils.data.Dataset
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # image = io.imread(image_path)
        # Load image and convert it to rgb format
        image = Image.open(image_path)
        image = image.convert('RGB')
        # Transform image if transformers are defined.
        if self.transform:
            image = self.transform(image)
        patient_id = self.patient_ids[idx]
        study = self.studies[idx]
        clin_info = self.clinical_info[idx]
        # label is a list of labels corresponding to 14 classes.
        label = self.labels[idx]
        sample = {'image': image, 'patient_id': patient_id, 'study': study, 'clinical_info': clin_info, 'labels': torch.FloatTensor(label)}
        return sample


def data_loader(label_path, batch_size):
    # Takes path to label csv (e.g. train_frontal_pa.csv, valid.csv) and returns a torch.DataLoader.

    # From torchvision.models doc:
    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    # where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and
    # then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    # Define a transformer to load images into required format
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_transforms = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.Resize(224),  # Resize to n*n, n = 224 for now
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean, std)
    ])

    # Load datasets using the tranforms. Now each image is a tensor of size [3,224,224]
    dataset = CheXpertTorchDataset(label_path, image_transforms)

    # Fill in batch size, shuffler, num workers, etc.
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
