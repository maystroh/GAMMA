from sklearn.metrics import roc_auc_score, average_precision_score
from statistics import mean
from modules.utils.logger import log


def compute_metrics(gt_list, pred_list):
    assert gt_list.shape == pred_list.shape, "The shapes are not the same"
    auc_list = []
    map_list = []
    for class_ind in range(gt_list.shape[1]):
        auc_sco = roc_auc_score(gt_list[:, class_ind:class_ind+1], pred_list[:, class_ind:class_ind+1])
        map_sco = average_precision_score(gt_list[:, class_ind:class_ind + 1], pred_list[:, class_ind:class_ind + 1])
        auc_list.append(auc_sco)
        map_list.append(map_sco)
        log.info(f'Class {(class_ind+1)}: Az = {auc_sco} , Ap = {map_sco}')
    log.info(f'Average : Az = {mean(auc_list)} , Ap = {mean(map_list)}')
    return mean(auc_list), mean(map_list)
