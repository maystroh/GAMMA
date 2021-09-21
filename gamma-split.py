"""
OCTIP script for splitting Brest's OCT dataset into train, validation and test subsets.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'Proprietary'
__version__ = '1.1'

import pandas as pd
import random
import sys
import tensorflow as tf
from argparse import ArgumentParser
from collections import defaultdict


def save_ground_truth(output_file_name,
                      selected_patient_ids,
                      patient_ids,
                      data):
    """
    Saves the ground truth for one data split.

    :param output_file_name: output CSV file name
    :param selected_patient_ids: selected patient IDs for this split
    :param patient_ids: patient ID associated with each row in the data matrix
    :param data: data matrix
    """
    columns = list(data.columns)
    with open(output_file_name, 'w') as f:
        f.write(','.join(columns) + '\n')
        for index, val in patient_ids.items():
            if val in selected_patient_ids:
                f.write(','.join([str(x) for x in data.loc[index, :].values]) + '\n')


def main():
    """
    Splits Brest's OCT dataset into train, validation and test subsets.
    """

    # parsing the command line
    parser = ArgumentParser(
        description='Splits Refuge dataset into train, validation and test subsets.')
    parser.add_argument('-r', '--ratios', type=float, nargs='+', default=[.8, .1, .1],
                        help='space-delimited list of train, validation (and test) ratios')
    parser.add_argument('-n', '--num_splits', type=int, default=100000,
                        help='number of tested random splits')
    parser.add_argument('--gamma_task', required=True,
                        help='classification/fovea/cupDisk')
    parser.add_argument('--exclude_data', default=['0019'], nargs='+',
                        help='classification/fovea/cupDisk')
    parser.add_argument('-l', '--labels', nargs='+',
                        default=['non', 'early', 'mid_advanced'],
                        help='space-delimited list of labels that should be split evenly across '
                             'subsets')
    if len(sys.argv[1:]) == 0:
        parser.print_usage()
        parser.exit()
    args = parser.parse_args()

    # train/validation/test ratio verification
    assert 2 <= len(args.ratios) <= 3, \
        'provide at least two ratios (train and validation) and at most three ' \
        '(train, validation and test)'
    if len(args.ratios) == 2:
        train_ratio, validation_ratio = tuple(args.ratios)
        test_ratio = 1. - train_ratio - validation_ratio
    else:
        train_ratio, validation_ratio, test_ratio = tuple(args.ratios)
    assert train_ratio >= 0. and validation_ratio >= 0. and test_ratio >= 0., \
        'all ratios must be positive or zero'
    assert abs(train_ratio + validation_ratio + test_ratio - 1.) < 1.e-9, 'ratios must sum to 1.'

    if 'classification' in args.gamma_task:
        data = pd.read_excel('../GAMMA_data/training_data/glaucoma_grading_training_GT.xlsx', converters={'data': str})
        data = data.drop(data[data.data == '0019'].index)
        patient_ids = data['data']
        all_counts, total_counts = [], []
        for label in args.labels:
            ground_truth_values = data[label]
            counts, total_count = defaultdict(int), 0
            for patient_id, ground_truth_value in zip(patient_ids, ground_truth_values):
                if ground_truth_value:
                    counts[patient_id] += 1
                    total_count += 1
            all_counts.append(counts)
            total_counts.append(total_count)
        num_labels = len(args.labels)
        # evaluating various data splits
        unique_patient_ids = list(set(patient_ids))
        num_patients = len(unique_patient_ids)
        num_train = int(round(num_patients * train_ratio))
        num_validation = int(round(num_patients * validation_ratio))
        best_score, best_split = 1., None
        progress = tf.keras.utils.Progbar(args.num_splits, unit_name='split')
        for step in range(args.num_splits):

            # histogram computation function
            def _compute_histograms(patient_ids):
                counts = []
                for i in range(num_labels):
                    count = 0
                    for patient_id in patient_ids:
                        count += all_counts[i][patient_id]
                    counts.append(count)
                return counts

            # histogram comparison function
            def _compare_histograms(count1, count2, ratio1, ratio2):
                delta_score = 0.
                for i in range(num_labels):
                    h1 = count1[i] / (num_patients * ratio1)
                    h2 = count2[i] / (num_patients * ratio2)
                    average_fraction = (h1 + h2) / 2.
                    relative_delta = (h1 - h2) / average_fraction if average_fraction else 0.
                    delta_score += abs(relative_delta)
                delta_score /= num_labels
                return delta_score

            # new data split
            random.shuffle(unique_patient_ids)
            train_ids = unique_patient_ids[:num_train]
            validation_ids = unique_patient_ids[num_train: num_train + num_validation]
            test_ids = unique_patient_ids[num_train + num_validation:]

            # computing histograms
            train_counts = _compute_histograms(train_ids)
            validation_counts = _compute_histograms(validation_ids)
            test_counts = _compute_histograms(test_ids)

            # comparing histograms
            score1 = _compare_histograms(train_counts, validation_counts, train_ratio, validation_ratio)
            score2 = _compare_histograms(validation_counts, test_counts, validation_ratio, test_ratio)
            score3 = _compare_histograms(test_counts, train_counts, test_ratio, train_ratio)
            score = max(score1, score2, score3)
            if score <= best_score:
                best_score = score
                best_split = train_ids, validation_ids, test_ids, \
                             train_counts, validation_counts, test_counts
            progress.add(1)

        # best split
        train_ids, validation_ids, test_ids, train_counts, validation_counts, test_counts = best_split
        with open('split-summary.yml', 'w') as summary:
            print('Label histograms:\n  train subset -> {}'.format(train_counts))
            print('  validation subset -> {}'.format(validation_counts))
            print('  test subset -> {}'.format(test_counts))
            summary.write('watched_labels: {}\n'.format(args.labels))
            summary.write('train_histogram: {}\n'.format(train_counts))
            summary.write('validation_histogram: {}\n'.format(validation_counts))
            summary.write('test_histogram: {}\n'.format(test_counts))

        # saving the split label files
        save_ground_truth('train_classification.csv', train_ids, patient_ids, data)
        save_ground_truth('validation_classification.csv', validation_ids, patient_ids, data)
        save_ground_truth('test_classification.csv', test_ids, patient_ids, data)


    elif 'fovea' in args.gamma_task or 'cupDisk' in args.gamma_task:

        data = pd.read_excel('../GAMMA_data/training_data/fovea_localization_training_GT.xlsx', converters={'data': str})
        patient_ids = data['data']
        unique_patient_ids = list(set(patient_ids))
        num_patients = len(unique_patient_ids)
        num_train = int(round(num_patients * train_ratio))
        num_validation = int(round(num_patients * validation_ratio))
        # new data split
        random.shuffle(unique_patient_ids)
        train_ids = unique_patient_ids[:num_train]
        validation_ids = unique_patient_ids[num_train: num_train + num_validation]
        test_ids = unique_patient_ids[num_train + num_validation:]

        print('Label histograms:\n  train subset -> {}'.format(len(train_ids)))
        print('  validation subset -> {}'.format(len(validation_ids)))
        print('  test subset -> {}'.format(len(test_ids)))

        if 'fovea' in args.gamma_task:
            # saving the split label files
            save_ground_truth('train_fovea.csv', train_ids, patient_ids, data)
            save_ground_truth('validation_fovea.csv', validation_ids, patient_ids, data)
            save_ground_truth('test_fovea.csv', test_ids, patient_ids, data)

        elif 'cupDisk' in args.gamma_task:
            # saving the split label files
            save_ground_truth('train_segmentation.csv', train_ids, patient_ids, data)
            save_ground_truth('validation_segmentation.csv', validation_ids, patient_ids, data)
            save_ground_truth('test_segmentation.csv', test_ids, patient_ids, data)


if __name__ == "__main__":
    main()
