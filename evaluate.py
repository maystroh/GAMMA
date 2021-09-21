import torch
from torch.utils.data import DataLoader
import nibabel as nib
import sys
import pandas as pd
import os
from modules.datasets.multimodal_dataset import MultiModelDataset
from modules.utils.logger import log
import numpy as np
from model import generate_model
import settings
from modules.datasets.patchify import extract_vol_patches, extract_img_patches, patch_score_function
import modules.utils.metrics as metrics
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score
from modules.utils.general import Modalities, set_modalities_to_load
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.nn.functional as F

from functools import partial
import collections

# a dictionary that keeps saving the activations as they come
activations = collections.defaultdict(list)


def save_activation(name, mod, inp, out):
    activations[name].append(out.cpu())


def test(data_loader, model, loss_criterion, sets, activations=None):

    GT_list = []
    predictions_list = []
    names_list = []
    losses_list = []
    model = model
    num_labels = 1

    model.eval()  # for testing

    for batch_id, (data, labels, name) in enumerate(data_loader):

        num_labels = labels.shape[0]

        if not sets.no_cuda:
            data = data.cuda()
        labels_task = labels

        if sets.patch_evaluation:
            if sets.modalities_to_load == Modalities.OCT:
                data_patches = extract_vol_patches(batch_volume=data, patch_size=(
                    sets.input_struct_C, sets.input_struct_D, sets.input_struct_H, sets.input_struct_W))
            if sets.modalities_to_load == Modalities.Fundus:
                data_patches = extract_img_patches(images=data, patch_size=(
                    sets.input_fundus_C, sets.input_fundus_H, sets.input_fundus_W), overlapped_patches=True)

            preds = []
            preds_multi_class = []

            with torch.no_grad():
                for batch_patches in data_patches:
                    patches_vol = torch.from_numpy(np.stack(batch_patches, axis=0))
                    patches_vol = patches_vol.cuda()
                    probs = model(patches_vol)
                    if probs.ndim == 1:
                        probs = probs.expand(1, sets.n_classes)

                    if sets.group_classes:
                        predictions = torch.sigmoid(probs)
                        preds.append(patch_score_function[sets.patch_evaluation_func](predictions, dim=0).values)
                    else:
                        if sets.mse_loss:
                            predictions[predictions < 0.5] = 0
                            predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
                            predictions[(predictions >= 1.5)] = 2
                            labels_task = functional.one_hot(labels, num_classes=sets.n_classes)
                        else:
                            predictions = torch.softmax(probs, dim=1)

                        # print(predictions)
                        predictions_argmax = torch.argmax(predictions, dim=1)
                        # TODO:  Affecting maximum class occurence as a predicted class for the image
                        # preds.append(torch.tensor(
                        #     [predictions_argmax.unique(return_counts=True, sorted=True)[0].cpu().tolist()[-1]]
                        #     , device='cuda:0'))
                        print(predictions_argmax)
                        preds.append(torch.max(predictions_argmax))
            # print(preds)
            preds = torch.stack((preds), dim=0)

        else:
            with torch.no_grad():
                preds = model(data)
                if preds.ndim == 1:
                    preds = preds.expand(1, sets.n_classes)
                if sets.mse_loss:
                    preds[preds < 0.5] = 0
                    preds[(preds >= 0.5) & (preds < 1.5)] = 1
                    preds[(preds >= 1.5)] = 2
                    labels_task = functional.one_hot(labels, num_classes=sets.n_classes)
                else:
                    preds = torch.sigmoid(preds) if sets.group_classes else torch.softmax(preds, dim=1)

        # log.info('{} processed -> {} / {}'.format(batch_val_id, preds.detach().cpu().numpy(), labels))
        GT_list.extend(labels.cpu().numpy())
        predictions_list = predictions_list + preds.cpu().numpy().tolist()
        if not sets.patch_evaluation:
            labels_task = labels_task.type(torch.cuda.FloatTensor) if sets.group_classes or sets.mse_loss else labels_task.type(
                torch.cuda.LongTensor)
            batch_loss = loss_criterion(preds.cuda(), labels_task.cuda())
            losses_list.append(batch_loss.item())

        names_list.append(name[0])

        if activations is not None:
            # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
            activations_values = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}
            # just print out the sizes of the saved activations as a sanity check
            for k, v in activations_values.items():
                print(k, v.size())

    # for batch_id, (data, labels, name) in enumerate(data_loader):
    #
    #     num_labels = labels.shape[0]
    #
    #     if not sets.no_cuda:
    #         data = data.cuda()
    #
    #     if sets.patch_evaluation:
    #         if sets.modalities_to_load == Modalities.OCT:
    #             data_patches = extract_vol_patches(batch_volume=data, patch_size=(
    #                 sets.input_struct_C, sets.input_struct_D, sets.input_struct_H, sets.input_struct_W))
    #         if sets.modalities_to_load == Modalities.Fundus:
    #             data_patches = extract_img_patches(images=data, patch_size=(
    #                 sets.input_fundus_C, sets.input_fundus_H, sets.input_fundus_W), overlapped_patches=True)
    #
    #         preds = []
    #         preds_multi_class = []
    #
    #         with torch.no_grad():
    #             for batch_patches in data_patches:
    #                 patches_vol = torch.from_numpy(np.stack(batch_patches, axis=0))
    #                 patches_vol = patches_vol.cuda()
    #                 probs = model(patches_vol)
    #                 if probs.ndim == 1:
    #                     probs = probs.expand(1, sets.n_classes)
    #
    #                 if sets.group_classes:
    #                     predictions = torch.sigmoid(probs)
    #                     preds.append(patch_score_function[sets.patch_evaluation_func](predictions, dim=0).values)
    #                 else:
    #                     predictions = torch.softmax(probs, dim=1)
    #                     predictions_argmax = torch.argmax(predictions, dim=1)
    #                     # TODO:  Affecting maximum class occurence as a predicted class for the image
    #                     preds.append(torch.tensor(
    #                         [predictions_argmax.unique(return_counts=True, sorted=True)[0].cpu().tolist()[-1]]
    #                         , device='cuda:0'))
    #
    #         preds = torch.stack((preds), dim=0)
    #     else:
    #         with torch.no_grad():
    #             preds = model(data)
    #             if preds.ndim == 1:
    #                 preds = preds.expand(1, sets.n_classes)
    #             preds = torch.sigmoid(preds) if sets.group_classes else torch.softmax(preds, dim=1)
    #
    #     log.info('{} processed -> {} / {}'.format(name, preds.detach().cpu().numpy(), labels))
    #
    #     GT_list.extend(labels.cpu().numpy())
    #     predictions_list = predictions_list + preds.cpu().numpy().tolist()
    #
    #     labels = labels.type(torch.cuda.FloatTensor) if sets.group_classes else labels.type(torch.cuda.LongTensor)
    #     if not sets.patch_evaluation:
    #         batch_val_loss = loss_criterion(preds.cuda(), labels.cuda())
    #         losses_list.append(batch_val_loss.item())
    #
    #     names_list.append(name[0])
    #
    #     if activations is not None:
    #         # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    #         activations_values = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}
    #         # just print out the sizes of the saved activations as a sanity check
    #         for k, v in activations_values.items():
    #             print(k, v.size())
    pred_array = np.stack(predictions_list, axis=0) if sets.group_classes or sets.patch_evaluation else np.argmax(
        np.stack(predictions_list, axis=0), axis=1)

    if sets.group_classes:
        log.info('*** Evaluation *** ')
        gt_array = np.stack(GT_list, axis=0)
        train_auc_score, train_aupr_score = metrics.compute_metrics(gt_array, pred_array)
    else:
        print(GT_list)
        print(pred_array)
        train_accuracy_score = accuracy_score(GT_list, pred_array)
        train_kappa_score = cohen_kappa_score(GT_list, pred_array, weights='quadratic')
        print('Acc:{}'.format(train_accuracy_score))
        print('Kappa:{}'.format(train_kappa_score))

    return predictions_list, [int(v) for v in GT_list], losses_list, num_labels, names_list


# --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True
# --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_512_512_Jul27_23-29-12/epoch_30_batch_9.pth.tar
# --modalities_to_load Fundus --group_classes --patch_evaluation

# --data_root=../GAMMA_data/validation_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=32 --input_struct_H=224 --input_struct_W=512 --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --model_3D=MedNet50
# --model_path=./weights/_classification_OCT_MedNet50_1_224_32_512_Aug04_16-52-23/epoch_361_batch_15.pth.tar --evaluate_file=online_validation_classification.csv --modalities_to_load OCT --patch_evaluation

if __name__ == '__main__':
    sets = settings.parse_opts()
    sets.phase = 'eval'
    set_modalities_to_load(sets)

    print(sets)
    # getting model
    model, parameters = generate_model(sets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint = torch.load(sets.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    log.info("=> loaded checkpoint '{}'".format(sets.model_path))

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))

    # data tensor
    test_nifti_dataset = MultiModelDataset(sets.data_root, sets.evaluate_file, sets.modalities_to_load, 'eval', sets)
    data_loader = DataLoader(test_nifti_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    loss_criterion = torch.nn.BCELoss(
        reduce=None) if sets.group_classes or 'segmentation' in sets.task else torch.nn.CrossEntropyLoss(
        reduce=None)
    loss_criterion = loss_criterion.cuda()

    # testing
    dataframe = pd.read_csv(sets.evaluate_file, header=0)
    predictions, GT_list, losses_list, num_classes, img_names = test(data_loader, model, loss_criterion, sets)

    # writing predictions to csv file
    if sets.group_classes:
        if not sets.patch_evaluation:
            df = pd.DataFrame({"data": img_names, "non": np.stack(predictions, axis=0)[:,0]})
            df.to_csv('{}/{}_classification_results.csv'.format(os.path.dirname(sets.model_path),
                                                          os.path.basename(sets.model_path).replace('.pth.tar', '')), index=False)
    else:
        if not sets.patch_evaluation:
            print(predictions)
            pred_array = np.argmax(np.stack(predictions, axis=0), axis=1)
            print(pred_array)
        else:
            pred_array = np.array(predictions)
            print(pred_array)

        pred_array_values = np.eye(3)[pred_array].astype(int)
        print(pred_array_values)

        df = pd.DataFrame({"data": img_names, "non": pred_array_values[:,0],
                           'early': pred_array_values[:,1], 'mid_advanced': pred_array_values[:,2]})
        df.to_csv('{}/{}_Classification_Results.csv'.format(os.path.dirname(sets.model_path),
                                                            os.path.basename(sets.model_path).replace('.pth.tar', '')),
                  index=False)
        # else:
        #     pred_array = np.stack(predictions_list, axis=0)


    # predictions = np.array(predictions)
    # predictions = predictions.reshape([-1, num_classes])
    # GT_list = np.array(GT_list)
    # GT_list = GT_list.reshape([-1, num_classes])
    # df_pred = pd.DataFrame(data=img_names, columns=['data'])
    # df_pred['Loss_value'] = losses_list
    # for i in range(1, num_classes + 1):
    #     df_pred['Pred_class_{}'.format(i)] = predictions[:, i - 1:i]
    #     df_pred['GT_class_{}'.format(i)] = GT_list[:, i - 1:i]
    # df_pred.to_csv('{}/{}_predictions.csv'.format(os.path.dirname(sets.model_path),
    #                                               os.path.basename(sets.model_path).replace('.pth.tar', '')),
    #                index=False)



