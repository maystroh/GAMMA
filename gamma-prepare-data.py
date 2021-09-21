"""
OCTIP script for converting Brest's OCT dataset in compressed NifTI format.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'Proprietary'
__version__ = '1.1'

import cv2
import glob
import nibabel as nib
import numpy as np
import octip
import os
import pandas as pd
import sys
from argparse import ArgumentParser
from collections import defaultdict

labels = ['Drusens', 'AEP', 'DSR', 'DEP', 'SHE', 'AG', 'Logettes', 'AER', 'Exsudats', 'MER', 'TM',
          'TVM', 'Autres Lésions', 'MLA', 'DMLA E', 'DMLA A', 'OMC diabètique', 'OMC', 'IVM',
          'Autres patho']
clean_labels = ['drusen', 'AEP', 'DSR', 'DEP', 'SHE', 'AG', 'logettes', 'AER', 'exsudats', 'MER',
                'TM', 'TVM', 'autres-lesions', 'MLA', 'DMLA-E', 'DMLA-A', 'OMC-diabetique', 'OMC',
                'IVM', 'autres-pathologies']


def resize_images(args,
                  selected_base_names,
                  input_dir,
                  output_dir,
                  change_ext=False):
    """
    Resizes all B-scans in a volume.

    :param args: command-line arguments
    :param selected_base_names: base name of the selected B-scans
    :param input_dir: directory containing the original B-scans
    :param output_dir: directory containing the resized B-scans
    :param change_ext: whether the extension of input image files should be changed to .png

    :return: the list of resized B-scans
    """
    volume = []
    for url, i in zip(selected_base_names, range(args.depth_oct)):
        if url is not None:
            url_ext = os.path.splitext(url.split("/")[-1])[0] + '.png'
            image = cv2.imread(os.path.join(input_dir, url_ext), cv2.IMREAD_GRAYSCALE)
            if image is None:
                url_ext = os.path.splitext(url.split("/")[-1])[0] + '.jpg'
                image = cv2.imread(os.path.join(input_dir, url_ext), cv2.IMREAD_GRAYSCALE)
            if image.shape != (args.height_oct, args.width_oct):
                image = cv2.resize(image, (args.width_oct, args.height_oct))
        else:
            image = np.zeros((args.height_oct, args.width_oct), np.uint8)
        volume.append(image)
        cv2.imwrite(os.path.join(output_dir, '{}.png'.format(i)), image)
    return volume

# --input_dirs=../GAMMA_data/training_data --depth_oct=100 --height_oct=224  --width_oct=224 --height_fundus=356 --width_fundus=356 --segment_retina_layers --gamma_task=classification

def main():
    """
    Preprocess Gamma's dataset.
    """

    # parsing the command line
    parser = ArgumentParser(
        description='Preprocess Gamma\'s dataset.')
    parser.add_argument('--input_dirs', required=True,
                        help='space-delimited list of input directories')
    parser.add_argument('-r', '--retina_model_dir', default='./models/',
                        help='directory containing retina segmentation models')
    parser.add_argument('--depth_oct', type=int, default=256, help='number of images selected per volume')
    parser.add_argument('--height_oct', type=int, default=224, help='image height after resizing')
    parser.add_argument('--width_oct', type=int, default=224, help='image width after resizing')

    parser.add_argument('--height_fundus', type=int, default=900, help='image height after resizing')
    parser.add_argument('--width_fundus', type=int, default=900, help='image width after resizing')
    parser.add_argument('--fundus_resize_only', dest='fundus_resize_only', action='store_true',
                        help='simply resize images')
    parser.set_defaults(fundus_resize_only=False)
    parser.add_argument('--luminance_only', dest='luminance_only', action='store_true',
                        help='normalize the luminance only')
    parser.set_defaults(luminance_only=True)
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='activate verbose log mode')

    parser.add_argument('--normalize_intensities', dest='normalize_intensities',
                        action='store_true',
                        help='if a retina segmentation model is provided, '
                             'intensities should be normalized')
    parser.add_argument('--native_intensities', dest='normalize_intensities',
                        action='store_false',
                        help='even if a retina segmentation model is provided, '
                             'intensities should not be normalized')
    parser.add_argument('--segment_retina_layers', dest='segment_retina_layers',
                        action='store_true',
                        help='Create Niftii files without segmenting the retina layers')
    parser.set_defaults(segment_retina_layers=False)
    parser.add_argument('--gamma_task', required=True,
                        help='classification/fovea/cupDisk')
    parser.set_defaults(normalize_intensities=False)

    if len(sys.argv[1:]) == 0:
        parser.print_usage()
        parser.exit()
    args = parser.parse_args()

    if 'classification' in args.gamma_task:

        # output multi-modal directory
        output_dirs = os.path.join(args.input_dirs,
                                   f"{args.gamma_task}_F_{args.width_fundus}_{args.height_fundus}_OCT_{args.width_oct}_{args.height_oct}_{args.depth_oct}_Seg_{args.segment_retina_layers}")
        if not os.path.exists(output_dirs):
            os.makedirs(output_dirs)

        # preparing retina segmentation if requested
        if args.segment_retina_layers and args.retina_model_dir is not None:
            localizer1 = octip.RetinaLocalizer('FPN', 'efficientnetb6', (384, 384),
                                               model_directory=args.retina_model_dir)
            localizer2 = octip.RetinaLocalizer('FPN', 'efficientnetb7', (320, 320),
                                               model_directory=args.retina_model_dir)
        else:
            localizer1, localizer2 = None, None

        multi_modal_dirs = os.path.join(args.input_dirs, 'multi-modality_images')
        for patient in glob.glob(os.path.join(multi_modal_dirs, '*')):
            print('Processing patient {}...'.format(patient))
            patient_name = patient.split("/")[-1]
            oct_dir = os.path.join(patient, patient_name)

            # selecting images
            bscans = glob.glob(os.path.join(oct_dir, '*.jpg'))

            # resizing images
            patient_image_dir = os.path.join(output_dirs, patient_name)
            if not os.path.exists(patient_image_dir):
                os.makedirs(patient_image_dir)
            oct_images_dir = os.path.join(patient_image_dir, patient_name)
            if not os.path.exists(oct_images_dir):
                os.makedirs(oct_images_dir)

            if args.segment_retina_layers:
                # segmenting and preprocessing images
                segmentation_directory_1 = os.path.join(oct_images_dir, 'segmentation1')
                segmentation_directory_2 = os.path.join(oct_images_dir, 'segmentation2')
                preprocessed_dir = os.path.join(oct_images_dir, 'preprocessed')
                if not os.path.exists(segmentation_directory_1):
                    os.makedirs(segmentation_directory_1)
                if not os.path.exists(segmentation_directory_2):
                    os.makedirs(segmentation_directory_2)
                if not os.path.exists(preprocessed_dir):
                    os.makedirs(preprocessed_dir)
                localizer1(octip.RetinaLocalizationDataset(bscans, 4, localizer1),
                           segmentation_directory_1)
                localizer2(octip.RetinaLocalizationDataset(bscans, 4, localizer2),
                           segmentation_directory_2)
                preprocessor = octip.PreProcessor(
                    args.height_oct, min_height=100,
                    normalize_intensities=args.normalize_intensities)
                preprocessor(bscans, [segmentation_directory_1, segmentation_directory_2],
                             preprocessed_dir)

                # resizing preprocessed images
                volume = resize_images(args, bscans, preprocessed_dir, oct_images_dir, True)
            else:
                # resizing preprocessed images
                volume = resize_images(args, bscans, oct_dir, oct_images_dir, True)

            volume = np.asarray(volume)
            print(volume.shape)
            # saving the volume
            img = nib.Nifti1Image(volume, np.eye(4))
            nib.save(img, os.path.join(output_dirs, patient_name + '/' + patient_name + '.nii.gz'))

            fundus_path = os.path.join(patient, f'{patient_name}.jpg')
            preprocessor = octip.Engine(args.width_fundus, args.verbose, args.fundus_resize_only, args.luminance_only)
            output = os.path.join(patient_image_dir, f'{patient_name}.png')
            image, _, _ = preprocessor.run(fundus_path)
            cv2.imwrite(output, image)
            # processing the image
            image, roi, error = preprocessor.run(fundus_path)
            cv2.imwrite(output, image)
            # if args.verbose and not error:
            #     print('ROI: [x0 = {}, y0 = {}, size = {}]'.format(roi[0], roi[1], roi[2]))

    elif 'fovea' in args.gamma_task:

        # output multi-modal directory
        output_dirs = os.path.join(args.input_dirs,
                                   f"{args.gamma_task}_{args.width_fundus}_{args.height_fundus}")
        if not os.path.exists(output_dirs):
            os.makedirs(output_dirs)

        multi_modal_dirs = os.path.join(args.input_dirs, 'multi-modality_images')

    elif 'cupDisk' in args.gamma_task:
        # output multi-modal directory
        output_dirs = os.path.join(args.input_dirs,
                                   f"{args.gamma_task}_{args.width_fundus}_{args.height_fundus}")
        if not os.path.exists(output_dirs):
            os.makedirs(output_dirs)

        multi_modal_dirs = os.path.join(args.input_dirs, 'multi-modality_images')

        gt_dirs = os.path.join(args.input_dirs, 'Disc_Cup_Mask')


if __name__ == "__main__":
    main()
