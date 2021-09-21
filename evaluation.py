from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

labels = ['normal', 'drusen', 'AEP', 'DSR', 'DEP', 'SHE', 'AG', 'logettes', 'AER', 'exsudats', 'MER',
                'TM', 'TVM', 'autres-lesions', 'MLA', 'DMLA-E', 'DMLA-A', 'OMC-diabetique', 'OMC',
                'IVM', 'autres-pathologies']

gt_file = './data/test.csv'

df_gt = pd.read_csv(gt_file, header=0)
df_pred = pd.read_csv('test_predictions.csv', header=0)

labels_classified = labels[:1]
y_true = df_gt[labels_classified].values

num_classes = y_true.shape[1]
for i in range(0, num_classes):
    y_true_class = y_true[:, i:i+1]
    y_score_class = df_pred.iloc[:, i+1:i+2].values
    # print(y_true_class)
    # print(y_score_class)

    print('Class {} : mAz = {} / mAP = {}'.format(i+1, roc_auc_score(y_true_class, y_score_class),average_precision_score(y_true_class, y_score_class)))