import math
import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
from modules.utils.logger import log
import torch

# https://github.com/jcreinhold/niftidataset/blob/master/niftidataset/dataset.py
# class NiftiDataset2(Dataset):
#     """
#     create a dataset class in PyTorch for reading NIfTI files
#     Args:
#         source_fns (List[str]): list of paths to source images
#         target_fns (List[str]): list of paths to target images
#         transform (Callable): transform to apply to both source and target images
#         preload (bool): load all data when initializing the dataset
#     """
#
#     def __init__(self, source_fns:List[str], target_fns:List[str], transform:Optional[Callable]=None, preload:bool=False):
#         self.source_fns, self.target_fns = source_fns, target_fns
#         self.transform = transform
#         self.preload = preload
#         if len(self.source_fns) != len(self.target_fns) or len(self.source_fns) == 0:
#             raise ValueError(f'Number of source and target images must be equal and non-zero')
#         if preload:
#             self.imgs = [(nib.load(s).get_data(), nib.load(t).get_data())
#                          for s, t in zip(self.source_fns, self.target_fns)]
#
#     @classmethod
#     def setup_from_dir(cls, source_dir:str, target_dir:str, transform:Optional[Callable]=None, preload:bool=False):
#         source_fns, target_fns = glob_imgs(source_dir), glob_imgs(target_dir)
#         return cls(source_fns, target_fns, transform, preload)
#
#     def __len__(self):
#         return len(self.source_fns)
#
#     def __getitem__(self, idx:int):
#         if not self.preload:
#             src_fn, tgt_fn = self.source_fns[idx], self.target_fns[idx]
#             sample = (nib.load(src_fn).get_fdata(dtype=np.float32), nib.load(tgt_fn).get_fdata(dtype=np.float32))
#         else:
#             sample = self.imgs[idx]
#         if self.transform is not None:
#             sample = self.transform(sample)
#         return sample

class NIFTIDataset(Dataset):

    def __init__(self, root_dir, gt_csv_file, sets):

        self.__gt_dataframe = pd.read_csv(gt_csv_file, header=0)
        print("Processing {} data".format(len(self.__gt_dataframe.index)))
        self.__root_dir = root_dir
        self.__input_D = sets.input_D
        self.__input_H = sets.input_H
        self.__input_W = sets.input_W
        self.__input_C = sets.input_C
        self.__generate_dummy_data = sets.dummy_data
        self.__phase = sets.phase
        self.__task = sets.task
        self.__volumes_dir = sets.volumes_dir

    def __len__(self):
        return len(self.__gt_dataframe.index)

    def __getitem__(self, idx):

        if 'classif' in self.__task:
            if self.__generate_dummy_data:
                img_array = torch.rand(self.__input_C, self.__input_D, self.__input_H, self.__input_W)
            else:
                ith_info = self.__gt_dataframe.iloc[idx]['urls']
                img_name = os.path.join(self.__root_dir, self.__volumes_dir, ith_info)
                if not os.path.isfile(img_name):
                    log.info (img_name)
                assert os.path.isfile(img_name)
                img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
                assert img is not None

                # data processing
                img_array = self.__data_process_classification__(img)
                #Todo: check this: Wenpei added this line of code but not sure why!!
                # img_array = img_array - img_array.mean()

                # 1 tensor array
                img_array = self.__nii2tensorarray__(img_array)

            #Todo: check this normalisation strategy
            img_array = (img_array - img_array.mean()) / img_array.std()

            labels = self.__gt_dataframe.iloc[idx]['normal']
            labels = np.float32(labels)

            return img_array, labels

        else:
            ith_info = self.__gt_dataframe.iloc[idx]['urls']
            img_name = os.path.join(self.__root_dir, ith_info)
            label_name = os.path.join(self.__root_dir, ith_info[1])
            assert os.path.isfile(img_name)
            assert os.path.isfile(label_name)
            img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
            assert img is not None
            mask = nibabel.load(label_name)
            assert mask is not None

            # data processing
            img_array, mask_array = self.__training_data_process__(img, mask)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            mask_array = self.__nii2tensorarray__(mask_array)

            assert img_array.shape == mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)

            return img_array, mask_array


    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape

        new_data = np.reshape(data, [1, z, y, x])

        new_data = new_data.astype("float32")

        return new_data

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = 0
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.__input_D * 1.0 / depth, self.__input_H * 1.0 / height, self.__input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __data_process_classification__(self, data):

        data = data.get_data()

        #Todo: check drop invalid range
        data = self.__drop_invalid_range__(data)
        data = self.__resize_data__(data)
        return data

    def __training_data_process__(self, data, label):
        # crop data according net input size
        data = data.get_data()
        label = label.get_data()

        # drop out the invalid range
        data, label = self.__drop_invalid_range__(data, label)

        # crop data
        data, label = self.__crop_data__(data, label)

        # resize data
        data = self.__resize_data__(data)
        label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data, label

    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data