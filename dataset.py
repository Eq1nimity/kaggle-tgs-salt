import gc
import glob
import os
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import models



class TGSSaltDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 file_list,
                 is_test=False,
                 divide=False,
                 image_size=(128, 128)):

        self.root_path = root_path
        self.file_list = file_list
        self.is_test = is_test

        self.divide = divide
        self.image_size = image_size

        self.orig_image_size = (101, 101)
        self.padding_pixels = None

        """
        root_path: folder specifying files location
        file_list: list of images IDs
        is_test: whether train or test data is used (contains masks or not)

        divide: whether to divide by 255
        image_size: output image size, should be divisible by 32

        orig_image_size: original images size
        padding_pixels: placeholder for list of padding dimensions
        """

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        # Get image path
        image_folder = os.path.join(self.root_path, 'images')
        image_path = os.path.join(image_folder, file_id + '.png')

        # Get mask path
        mask_folder = os.path.join(self.root_path, 'masks')
        mask_path = os.path.join(mask_folder, file_id + '.png')

        # Load image
        image = self.__load_image(image_path)
        if not self.is_test:
            # Load mask for training or evaluation
            mask = self.__load_image(mask_path, mask=True)
            if self.divide:
                image = image / 255.
                mask = mask / 255.
            # Transform into torch float Tensors of shape (CxHxW).
            image = torch.from_numpy(
                image).float().permute([2, 0, 1])
            mask = torch.from_numpy(
                np.expand_dims(mask, axis=-1)).float().permute([2, 0, 1])
            return image, mask

        if self.is_test:
            if self.divide:
                image = image / 255.
            image = torch.from_numpy(image).float().permute([2, 0, 1])
            return (image,)

    def set_padding(self):

        """
        Compute padding borders for images based on original and specified image size.
        """

        pad_floor = np.floor(
            (np.asarray(self.image_size) - np.asarray(self.orig_image_size)) / 2)
        pad_ceil = np.ceil((np.asarray(self.image_size) -
                            np.asarray(self.orig_image_size)) / 2)

        self.padding_pixels = np.asarray(
            (pad_floor[0], pad_ceil[0], pad_floor[1], pad_ceil[1])).astype(np.int32)

        return

    def __pad_image(self, img):

        """
        Pad images according to border set in set_padding.
        Original image is centered.
        """

        y_min_pad, y_max_pad, x_min_pad, x_max_pad = self.padding_pixels[
                                                         0], self.padding_pixels[1], self.padding_pixels[2], \
                                                     self.padding_pixels[3]

        img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad,
                                 x_min_pad, x_max_pad,
                                 cv2.BORDER_REPLICATE)

        assert img.shape[:2] == self.image_size, '\
        Image after padding must have the same shape as input image.'

        return img

    def __load_image(self, path, mask=False):

        """
        Helper function for loading image.
        If mask is loaded, it is loaded in grayscale (, 0) parameter.
        """

        if mask:
            img = cv2.imread(str(path), 0)
        else:
            img = cv2.imread(str(path))


        height, width = img.shape[0], img.shape[1]

        img = self.__pad_image(img)

        return img


    def return_padding_borders(self):
        """
        Return padding borders to easily crop the images.
        """
        return self.padding_pixels