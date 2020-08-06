# Preprocess RadOnc Labels
# Split the data into training and validation set

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
import fnmatch
import re


input_dir = './data/RadOnc/NN_Ventricles/'
output_dir = './data/RadOnc/labels/'
files = os.listdir(input_dir)
files = fnmatch.filter(files, '*ventricle*.nii.gz')

for file_name in files:
    if not os.path.isdir(file_name):
        label_name = os.path.join(input_dir, file_name)
        label = nibabel.load(label_name)
        data = label.get_fdata()
        header = label.header
        affine = label.affine

        header_new = header.copy()
        data_new = np.zeros(np.shape(data))
        index = [52, 51, 4]  # Left/Right/3rd Ventricle

        for i, j in enumerate(index):
            data_new[data == j] = 1
            # data_new[data == j] = i + 1

        label_new = nibabel.nifti1.Nifti1Image(data_new, affine, header)
        label_name_new = os.path.join(output_dir, file_name)
        label_new.to_filename(label_name_new)

random.shuffle(files)
train_labels = files[:40]
test_labels = files[40:]

with open('./data/train.txt', "w") as f:
    for train_label in train_labels:
        name = re.sub('_ventricle_mask.nii.gz', '', train_label)
        train_data = name + '_T1-crop-resampled.nii'
        f.write('RadOnc/images/' + train_data + ' ')
        f.write('RadOnc/labels/' + train_label + '\n')

f.close()

with open('./data/val.txt', "w") as f:
    for test_label in test_labels:
        name = re.sub('_ventricle_mask.nii.gz', '', test_label)
        test_data = name + '_T1-crop-resampled.nii'
        f.write('RadOnc/images/' + test_data + ' ')
        f.write('RadOnc/labels/' + test_label + '\n')

f.close()