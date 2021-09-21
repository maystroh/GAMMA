import numpy as np
import skimage
import nibabel as nib
import os
import torch
from PIL import Image
from random import randrange

patch_score_function = {
    'max': torch.max,
    'min': torch.min,
    'median': torch.median,
    'avg': torch.mean
}


def extract_vol_patches(batch_volume, patch_size, overlapped_patches=False):
    batch_size, vol_c, vol_d, vol_h, vol_w = batch_volume.shape
    p_c, p_d, p_h, p_w = patch_size
    if p_h > vol_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the volume.")

    if p_w > vol_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the volume.")

    if p_d > vol_d:
        raise ValueError("z of the patch should be less than the z"
                         " of the volume.")

    if p_c > vol_c:
        raise ValueError("The channels of the patch should be less than the channels"
                         " of the volume.")

    list_vols = []
    for ind in range(batch_size):
        volume = batch_volume[ind]
        list_patches = []
        block_shape = np.array(patch_size)
        npads = np.array(volume.shape) % block_shape
        # print(npads)

        new_shape = (
            ((patch_size[0] - npads[0]) // 2, (patch_size[0] - npads[0]) // 2) if npads[1] != 0 else (0, 0),
            ((patch_size[1] - npads[1]) // 2, (patch_size[1] - npads[1]) // 2) if npads[1] != 0 else (0, 0),
            ((patch_size[2] - npads[2]) // 2, (patch_size[2] - npads[2]) // 2) if npads[2] != 0 else (0, 0),
            ((patch_size[3] - npads[3]) // 2, (patch_size[3] - npads[3]) // 2) if npads[3] != 0 else (0, 0))
        # print(new_shape)
        padded_vol = np.pad(volume.cpu(), new_shape, constant_values=128)
        # print(padded_vol.shape)

        Blocks = skimage.util.view_as_blocks(padded_vol, block_shape=patch_size) if not overlapped_patches \
            else skimage.util.view_as_windows(padded_vol, block_shape=patch_size)

        c_dim, z_dim, h_dim, w_dim, *_ = Blocks.shape
        for ind_c in range(c_dim):
            for ind_z in range(z_dim):
                for ind_h in range(h_dim):
                    for ind_w in range(w_dim):
                        patch = Blocks[ind_c, ind_z, ind_h, ind_w]
                        # img = nib.Nifti1Image(patch.squeeze(), np.eye(4))
                        # nib.save(img, '/home/harddrive/Projects/GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True/0001/{}_{}_{}_{}.nii.gz'.format(ind_c, ind_z, ind_h, ind_w))
                        list_patches.append(patch)

        list_vols.append(list_patches)

    return list_vols


def extract_img_patches(images, patch_size, overlapped_patches=False):
    batch_size, image_c, image_h, image_w = images.shape
    p_c, p_y, p_x = patch_size

    if p_y > image_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_x > image_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    if p_c > image_c:
        raise ValueError("z of the patch should be less than the z"
                         " of the image.")

    patch_size = (p_x, p_y, p_c)
    images = images.reshape(
            [images.shape[0], images.shape[3], images.shape[2], images.shape[1]])
    list_images = []
    for ind in range(batch_size):
        fundus_img = images[ind]

        block_shape = np.array(patch_size)
        npads = np.array(fundus_img.shape) % block_shape

        padded_img = np.pad(fundus_img.cpu(), (
            ((patch_size[0] - npads[0]) // 2, (patch_size[0] - npads[0]) // 2) if npads[0] != 0 else (0, 0),
            ((patch_size[1] - npads[1]) // 2, (patch_size[1] - npads[1]) // 2) if npads[1] != 0 else (0, 0),
            ((patch_size[2] - npads[2]) // 2, (patch_size[2] - npads[2]) // 2) if npads[2] != 0 else (0, 0)),
                            constant_values=128)

        # data = Image.fromarray(padded_img.astype("uint8"))
        # data.save(f'./weights/padded_img_val.png')
        # print(padded_img.shape)

        Blocks = skimage.util.view_as_blocks(padded_img, block_shape=patch_size) if not overlapped_patches else \
            skimage.util.view_as_windows(padded_img, window_shape=patch_size, step=(image_w//10, image_h//10, 3))
        # print(Blocks.shape)

        list_patches = []
        c_dim, h_dim, w_dim, *_ = Blocks.shape
        for ind_c in range(c_dim):
                for ind_h in range(h_dim):
                    for ind_w in range(w_dim):
                        patch = Blocks[ind_c, ind_h, ind_w]
                        # data = Image.fromarray(patch.astype("uint8"))
                        # rand_n = randrange(0, 1000000)
                        # data.save(f'./weights/val_{rand_n}.png')
                        patch_tosave = patch.reshape(
                            [patch.shape[2], patch.shape[1], patch.shape[0]]).astype("float32")
                        list_patches.append(patch_tosave)

        list_images.append(list_patches)
    return list_images
