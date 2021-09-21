import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel
from PIL import Image
from modules.utils.logger import log
import logging
import torch
import cv2
from modules.utils.general import Modalities
from random import randrange
import volumentations as V
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.getLogger('PIL').setLevel(logging.WARNING)


def get_augmentation_volume(patch_size):
    return V.Compose([
        V.RandomCrop(patch_size),
        # V.Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        # V.RandomCropFromBorders(crop_value=0.1, p=0.5),
        # V.ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        # Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
        # Flip(0, p=0.5),
        # Flip(1, p=0.5),
        # Flip(2, p=0.5),
        # RandomRotate90((1, 2), p=0.5),
        # GaussianNoise(var_limit=(0, 5), p=0.2),
        # RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
    ], p=1.0)


def get_augmentation_image(patch_size):
    return A.Compose([
        A.RandomCrop(height=patch_size[1], width=patch_size[2]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.3),
        A.CLAHE(p=0.3),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
        # A.Affine(shear=30, rotate=0, p=0.2),
        # ToTensorV2(),
    ], p=1.0)


class MultiModelDataset(Dataset):

    def __init__(self, root_dir, gt_csv_file, modalities_to_load, phase, sets):

        self.__gt_dataframe = pd.read_csv(gt_csv_file, header=0)
        print("Processing {} data".format(len(self.__gt_dataframe.index)))
        self.__root_dir = root_dir
        self.__input_D = sets.input_struct_D
        self.__input_H = sets.input_struct_H
        self.__input_W = sets.input_struct_W
        self.__input_C = sets.input_struct_C
        self.__input_fundus_H = sets.input_fundus_H
        self.__input_fundus_W = sets.input_fundus_W
        self.__generate_dummy_data = sets.dummy_data
        self.__phase = phase
        self.__task = sets.task
        self.__modalities_to_load = modalities_to_load
        self.__augmentation = get_augmentation_volume((sets.input_struct_D, sets.input_struct_H, sets.input_struct_W))
        self.__augmentation_fundus = get_augmentation_image(
            (sets.input_fundus_C, sets.input_fundus_H, sets.input_fundus_W))
        self.__data_augmentation = sets.augment_data
        self.__group_classes = sets.group_classes

    def __len__(self):
        return len(self.__gt_dataframe.index)

    def __getitem__(self, idx):

        if 'classif' in self.__task:
            if self.__generate_dummy_data:
                img_oct_array = torch.rand(self.__input_C, self.__input_D, self.__input_H, self.__input_W)
            else:
                ith_info = '{:0>4d}'.format(self.__gt_dataframe.iloc[idx]['data'])

                if self.__modalities_to_load.value % Modalities.OCT.value == 0:
                    img_name = os.path.join(self.__root_dir, ith_info, f'{ith_info}.nii.gz')
                    if not os.path.isfile(img_name):
                        log.info(img_name)
                    assert os.path.isfile(img_name)
                    img_oct_array = nibabel.load(img_name)

                    # data processing
                    if self.__phase == 'train' and self.__data_augmentation:
                        new_data = img_oct_array.get_data()
                        data = {'image': new_data}
                        aug_data = self.__augmentation(**data)
                        img_array = aug_data['image']
                        img_oct_array = np.expand_dims(img_array, axis=0)
                    else:
                        img_oct_array = self.__nii2tensorarray__(img_oct_array)

                    # Todo: check this normalisation strategy
                    # img_oct_array = (img_oct_array - img_oct_array.mean()) / img_oct_array.std()

                if self.__modalities_to_load.value % Modalities.Fundus.value == 0:
                    fundus_img_name = os.path.join(self.__root_dir, ith_info, f'{ith_info}.png')
                    if not os.path.isfile(fundus_img_name):
                        log.info(fundus_img_name)
                    fundus_img = Image.open(fundus_img_name).convert("RGB")
                    fundus_img = np.asarray(fundus_img)

                    if self.__phase == 'train' and self.__data_augmentation:
                        data = {'image': fundus_img}
                        aug_data = self.__augmentation_fundus(**data)
                        fundus_img = aug_data['image']

                    # data = Image.fromarray(fundus_img)
                    # rand_n = randrange(0, 1000000)
                    # data.save(f'./weights/train_{rand_n}.png')

                    fundus_img = fundus_img.reshape(
                        [fundus_img.shape[2], fundus_img.shape[1], fundus_img.shape[0]]).astype("float32")

            if self.__group_classes:
                labels = self.__gt_dataframe.loc[idx, ['non']]
                labels = np.float32(labels)
            else:
                labels = self.__gt_dataframe.loc[idx, ['non', 'early', 'mid_advanced']]
                labels = np.argmax(np.float32(labels))

            if self.__modalities_to_load == Modalities.OCT_FUNDUS:
                return (img_oct_array, fundus_img), labels, ith_info
            if self.__modalities_to_load == Modalities.OCT:
                return img_oct_array, labels, ith_info
            if self.__modalities_to_load == Modalities.Fundus:
                return fundus_img, labels, ith_info

        elif 'segmentation' in self.__task:
            print('asdas')
        elif 'fovea' in self.__task:
            print('asdas')

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape

        new_data = np.reshape(data.get_data(), [1, z, y, x])

        new_data = new_data.astype("float32")

        return new_data
