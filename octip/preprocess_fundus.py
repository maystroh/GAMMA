from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2015-2018 Gwenole Quellec'
__license__ = 'Proprietary'
__version__ = '1.0'

import cv2
import numpy as np
import os
import sys
import timeit


class Engine(object):
    """
    Pre-processes retinal images.
    """

    def __init__(self,
                 output_size=448,
                 verbose=True,
                 resize_only=False,
                 luminance_only=False):
        """
        Pre-processing engine constructor.

        :param output_size: size of images after pre-processing
        :param verbose: activate verbose mode?
        :param resize_only: resize images without normalizing intensity
        :param luminance_only: normalize the luminance only
        """
        self.resize_only = resize_only
        if self.resize_only:
            self.radius = int(round(output_size / 2.))
        else:
            self.radius = int(round((output_size / 448.) * 256.))
        self.output_size = output_size
        if output_size <= 0:
            raise ValueError('A strictly positive output image size is required')
        self.verbose = verbose
        self.resize_only = resize_only
        self.luminance_only = luminance_only

    def __pad_with_zeros(self, img):
        """
        Pads an image with zeros if necessary.

        :param img: input image

        :return: the zero padded image
        :return: the offset of the original image in the zero-padded image
        """
        offset = [0, 0]
        if img.shape[0] < self.output_size:
            delta = self.output_size - img.shape[0]
            top = delta // 2
            bottom = delta - top
            img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            offset[0] = top
        if img.shape[1] < self.output_size:
            delta = self.output_size - img.shape[1]
            left = delta // 2
            right = delta - left
            img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            offset[1] = left
        return img, offset

    def __scale_to_radius(self,  img):
        """
        Scales an image to the desired radius.

        :param img: input image

        :return: the scaled image
        :return: the scale factor
        """
        x = img[img.shape[0] // 2, :, :].sum(1)
        r = (x > x.mean() / 10).sum() / 2
        scale_factor = float(self.radius) / r
        return cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor), scale_factor

    def run(self, input_file):
        """
        Loads and pre-processes one image.

        :param input_file: file containing the image

        :return: the pre-processed image
        :return: the region of the input image that has been preprocessed
        :return: Boolean error code
        """
        start_time = timeit.default_timer()
        error = False
        try:

            # scaling the image to a given radius
            img, scale_factor = self.__scale_to_radius(cv2.imread(input_file))

            # detecting empty rows
            row_scores = np.max(np.sum(img, axis=2), axis=1)
            row_mask = np.expand_dims(row_scores > row_scores.mean() / 10, axis=1)
            row_mask = np.expand_dims(row_mask, axis=2)

            # subtracting the local mean color
            if not self.resize_only:
                if self.luminance_only:
                    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    v_img = ycrcb_img[:, :, 0]
                    v_img = cv2.addWeighted(v_img, 4, cv2.GaussianBlur(v_img, (0, 0), self.radius / 30),
                                            -4, 128)
                    ycrcb_img[:, :, 0] = v_img
                    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
                else:
                    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), self.radius / 30),
                                          -4, 128)

            # computing the mask image
            mask = np.zeros(img.shape)
            cv2.circle(mask, (img.shape[1] // 2, img.shape[0] // 2), self.radius, (1, 1, 1),
                       -1, 8, 0)
            mask, _ = self.__pad_with_zeros(mask * row_mask)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), anchor=(-1, -1),
                             iterations=int(.05 * self.radius))

            # masking the image
            img, offset = self.__pad_with_zeros(img)
            if self.resize_only:
                rgb = img
            else:
                rgb = img * mask + 128 * (1 - mask)

            # cropping parameters
            top = (rgb.shape[0] - self.output_size) // 2
            left = (rgb.shape[1] - self.output_size) // 2
            roi = [int(round(x / scale_factor)) for x in [left - offset[1], top - offset[0],
                                                          self.output_size]]

            # cropping the image
            result = rgb[top: top + self.output_size, left: left + self.output_size].copy()
            np.clip(result, 0, 255, out=result)
            result = result.astype(np.uint8)

        # error management
        except Exception as err:
            if self.verbose:
                print(err)
            result = 128 * np.ones((self.output_size, self.output_size, 3), np.uint8)
            roi = []
            error = True

        # printing a summary
        if self.verbose:
            elapsed = timeit.default_timer() - start_time
            print('Image {} preprocessed {}successfully in {} seconds.'
                  .format(input_file, 'un' if error else '', elapsed))
        return result, roi, error
