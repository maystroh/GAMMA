#!/usr/bin/python

from __future__ import division
import os
import warnings
from settings import parse_opts, mlflow_save_params
from time import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtools.optim import RangerLars
from torch.nn import functional

from model import generate_model
from modules.utils import tensor_utils
from modules.utils.general import Modalities, get_task_name, set_modalities_to_load
from modules.utils.metrics import compute_metrics
from modules.utils.logger import log
from modules.utils.tensor_utils import img2d
from modules.datasets.patchify import extract_vol_patches, extract_img_patches, patch_score_function
from modules.datasets.multimodal_dataset import MultiModelDataset

import mlflow

import torch.multiprocessing as mp
import torch.distributed as dist

# import timm
# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

# --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256
# --input_struct_H=224 --input_struct_W=256  --task=classification --batch_size=1 --n_classes=3  --input_fundus_W=512
# --input_fundus_H=512 --modalities_to_load OCT --augment_data  --group_classes --patch_evaluation --patch_evaluation_func=max \
# --backprop_after_n_batch=8

# According to these tips : https://www.youtube.com/watch?v=9mS1fIYj1So
torch.backends.cudnn.benchmark = True


# if not sets.cluster_run:
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(gpu, sets):
    rank = sets.nr * sets.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=sets.world_size,
        rank=rank
    )

    # It is always important to set the seed before starting launching tests..
    if sets.manual_seed is not None:
        torch.backends.cudnn.deterministic = True
        tensor_utils.config_torch_np_seed(sets.manual_seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Saving the parameters for mlflow experiments tracking tool
    mlflow.set_experiment('Classification GAMMA random seed')
    mlflow_save_params(sets)
    torch.cuda.set_device(gpu)

    set_modalities_to_load(sets)

    if not sets.dry_run:
        writer = SummaryWriter(comment=get_task_name(sets))
        mlflow.set_tag("mlflow.runName", get_task_name(sets))
        models_dir = os.path.join(sets.save_folder, get_task_name(sets))

    volumes_train_gt = f'./train_{sets.task}.csv'
    volumes_dev_gt = f'./validation_{sets.task}.csv'

    weight_decay = 2e-5
    momentum = 0.9
    sets.pin_memory = True
    num_workers = sets.num_workers

    if sets.group_classes:
        sets.n_classes = 1

    model, parameters = generate_model(sets)
    model.cuda(gpu)
    print(model)
    # Wrapper around our model to handle parallel training
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    training_dataset = MultiModelDataset(sets.data_root, volumes_train_gt, sets.modalities_to_load, 'train', sets)
    # Sampler that takes care of the distribution of the batches such that
    # the data is not repeated in the iteration and sampled accordingly
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        training_dataset,
        num_replicas=sets.world_size,
        rank=rank
    )
    training_dataloader = DataLoader(training_dataset, batch_size=sets.batch_size,
                                     shuffle=(False if sets.overfit_one_batch else True),
                                     num_workers = 0,pin_memory = True, sampler = train_sampler)

    validation_dataset = MultiModelDataset(sets.data_root, volumes_dev_gt, sets.modalities_to_load, 'eval', sets)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_dataset,
        num_replicas=sets.world_size,
        rank=rank
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=sets.batch_size, shuffle=False,
                                       num_workers = 0,pin_memory = True, sampler = val_sampler)

    if sets.mse_loss:
        loss_criterion = torch.nn.SmoothL1Loss()
    else:
        loss_criterion = torch.nn.BCELoss(
            reduce=None) if sets.group_classes or 'segmentation' in sets.task else torch.nn.CrossEntropyLoss(
            reduce=None)

    optimizer = RangerLars(model.parameters())
    # optimizer = torch.optim.Adam(model.parameters(), lr=sets.lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5 ,max_lr=1e-2, cycle_momentum=False,
    #                                         step_size_up=len(training_dataloader),
    #                                         gamma=0.99)

    # Resume training from a previous model
    if sets.resume_train_path and os.path.isfile(sets.resume_train_path):
        print("=> loading checkpoint '{}'".format(sets.resume_train_path))
        checkpoint = torch.load(sets.resume_train_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = sets.lr
        print("=> loaded checkpoint '{}' (epoch {})".format(sets.resume_train_path, checkpoint['epoch']))

    # settings
    batches_per_epoch = len(training_dataloader)
    steps_saving_model = batches_per_epoch if sets.save_intervals == 0 else sets.save_intervals
    log.info('{} epochs in total, {} batches per epoch'.format(sets.n_epochs, batches_per_epoch))

    log.info("Current settings are:")
    log.info(sets)

    loss_criterion = loss_criterion.cuda(gpu)

    model.train()
    train_time_sp = time()
    total_step_train = 0
    best_val_acc = 0
    best_val_auc = 0
    for epoch in range(sets.n_epochs):
        # log.info('**** Start epoch {} with lr = {}'.format(epoch + 1, scheduler.get_last_lr()))
        log.info('**** Start epoch {}'.format(epoch + 1))

        loss_batch = []

        model.train()
        GT_list = []
        predictions_list = []

        loss = 0
        for batch_id, (data, labels, names) in enumerate(training_dataloader):
            # if batch_id > 10:
            #     break
            if sets.overfit_one_batch and batch_id > 0:
                break

            # getting data batch
            total_step_train = epoch * batches_per_epoch + batch_id + 1
            batch_id_sp = epoch * batches_per_epoch
            if sets.modalities_to_load != Modalities.OCT_FUNDUS:
                data = data.cuda(non_blocking=True)
                predictions = model(data)
            else:
                oct_data = data[0].cuda(non_blocking=True)
                fundus_data = data[1].cuda(non_blocking=True)
                predictions = model({'image': fundus_data,
                                     'video': oct_data})

            if predictions.ndim == 1:
                predictions = predictions.expand(1, sets.n_classes)

            if sets.mse_loss:
                predictions[predictions < 0.5] = 0
                predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
                predictions[(predictions >= 1.5)] = 2
                labels_task = functional.one_hot(labels, num_classes=sets.n_classes)
            else:
                predictions = torch.sigmoid(predictions) if sets.group_classes else torch.softmax(predictions, dim=1)
                labels_task = labels

            labels_task = labels_task.type(
                torch.cuda.FloatTensor) if sets.group_classes or sets.mse_loss else labels_task.type(
                torch.cuda.LongTensor)

            GT_list.extend(labels.cpu().numpy())
            predictions_list = predictions_list + predictions.detach().cpu().numpy().tolist()
            loss_value = loss_criterion(predictions.cuda(non_blocking=True), labels_task)
            loss_value = loss_value / sets.backprop_after_n_batch  # Normalize our loss (if averaged)
            loss_value.backward()  # Backward pass
            if (batch_id + 1) % sets.backprop_after_n_batch == 0:
                # log.info('Updating weights..')
                # loss_value = loss_value / sets.backprop_after_n_batch
                optimizer.step()  # Update weights
                # According to these tips : ~/https://www.youtube.com/watch?v=9mS1fIYj1So
                # It is better to use optimizer.zero_grad()
                for param in model.parameters():  # Reset gradients tensors
                    param.grad = None

            avg_batch_time = (time() - train_time_sp) / (1 + batch_id_sp)
            if gpu == 0:
                log.info('Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}' \
                         .format(epoch + 1, batch_id, batch_id_sp, loss_value.item(), avg_batch_time))

            loss_batch.append(loss_value.item())

            if not sets.dry_run and total_step_train != 0 and total_step_train % steps_saving_model == 0:
                model_save_path = '{}/epoch_{}_batch_{}.pth.tar'.format(models_dir, epoch + 1, batch_id)
                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)

                log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch + 1, batch_id))
                torch.save({
                    'epoch': epoch + 1,
                    'batch_id': batch_id,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    model_save_path)

        mean_train_loss = np.mean(loss_batch)
        pred_array = np.stack(predictions_list, axis=0) if sets.group_classes else np.argmax(
            np.stack(predictions_list, axis=0), axis=1)

        if sets.group_classes:
            log.info('*** Train evaluation *** ')
            gt_array = np.stack(GT_list, axis=0)
            train_auc_score, train_aupr_score = compute_metrics(gt_array, pred_array)
        else:
            train_accuracy_score = accuracy_score(GT_list, pred_array)
            train_kappa_score = cohen_kappa_score(GT_list, pred_array, weights='quadratic')

        model.eval()
        val_loss = []
        GT_list = []
        predictions_list = []
        log.info('*** Val evaluation *** ')
        loop = tqdm(validation_dataloader)
        for batch_val_id, (data, labels, names) in enumerate(loop):

            if not sets.no_cuda:
                data = data.cuda(non_blocking=True)

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
                        patches_vol = patches_vol.cuda(non_blocking=True)
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

                            print(predictions)
                            predictions_argmax = torch.argmax(predictions, dim=1)
                            # TODO:  Affecting maximum class occurence as a predicted class for the image
                            # preds.append(torch.tensor(
                            #     [predictions_argmax.unique(return_counts=True, sorted=True)[0].cpu().tolist()[-1]]
                            #     , device='cuda:0'))
                            print(predictions_argmax)
                            preds.append(torch.max(predictions_argmax))
                print(preds)
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
                labels_task = labels_task.type(
                    torch.cuda.FloatTensor) if sets.group_classes or sets.mse_loss else labels_task.type(
                    torch.cuda.LongTensor)
                batch_val_loss = loss_criterion(preds.cuda(non_blocking=True), labels_task.cuda(non_blocking=True))
                val_loss.append(batch_val_loss.item())

        # pred_array = np.stack(predictions_list, axis=0) if sets.group_classes else np.argmax(
        #     np.stack(predictions_list, axis=0), axis=1)

        pred_array = np.stack(predictions_list, axis=0)
        if sets.group_classes:
            gt_array = np.stack(GT_list, axis=0)
            val_auc_score, val_aupr_score = compute_metrics(gt_array, pred_array)
        else:
            val_accuracy_score = accuracy_score(GT_list, pred_array)
            val_kappa_score = cohen_kappa_score(GT_list, pred_array, weights='quadratic')
            log.info("Acc = %.4f, Kappa = %.4f" % (val_accuracy_score, val_kappa_score))
        mean_val_loss = np.mean(val_loss) if not sets.patch_evaluation else 0

        if not sets.dry_run:
            # writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch + 1)
            writer.add_scalar("dev Loss/epoch", mean_val_loss, epoch + 1)
            writer.add_scalar("train Loss/epoch", mean_train_loss, epoch + 1)

            if sets.group_classes:
                writer.add_scalar("dev AUC/epoch", val_auc_score, epoch + 1)
                writer.add_scalar("dev mAP/epoch", val_aupr_score, epoch + 1)
                writer.add_scalar("train AUC/epoch", train_auc_score, epoch + 1)
                writer.add_scalar("train mAP/epoch", train_aupr_score, epoch + 1)
                if val_auc_score > best_val_auc:
                    best_val_auc = val_auc_score
                    mlflow.log_metric("Val_AUC", val_auc_score)
                    mlflow.log_metric("Val_mAP", val_aupr_score)
                    mlflow.log_metric("Best_epoch", epoch)
                log.info(
                    "Epoch [%d] Train: Loss = %.4f, AUC = %.4f, mAP = %.4f / Val: Loss = %.4f, AUC = %.4f, mAP = %.4f" % (
                        epoch + 1, mean_train_loss, train_auc_score, train_aupr_score, mean_val_loss, val_auc_score,
                        val_aupr_score))
            else:
                writer.add_scalar("dev Acc/epoch", val_accuracy_score, epoch + 1)
                writer.add_scalar("dev Kappa/epoch", val_kappa_score, epoch + 1)
                writer.add_scalar("train Acc/epoch", train_accuracy_score, epoch + 1)
                writer.add_scalar("train Kappa/epoch", train_kappa_score, epoch + 1)
                if val_accuracy_score > best_val_acc:
                    best_val_acc = val_accuracy_score
                    mlflow.log_metric("Val_ACC", val_accuracy_score)
                    mlflow.log_metric("Val_Kappa", val_kappa_score)
                    mlflow.log_metric("Best_epoch", epoch)
                log.info(
                    "Epoch [%d] Train: Loss = %.4f, Acc = %.4f, Kappa = %.4f / Val: Loss = %.4f, Acc = %.4f, Kappa = %.4f" % (
                        epoch + 1, mean_train_loss, train_accuracy_score, train_kappa_score, mean_val_loss,
                        val_accuracy_score,
                        val_kappa_score))

            writer.flush()

        # scheduler.step()

    if gpu == 0:
        print('Finished training')

    writer.close()


if __name__ == '__main__':
    sets = parse_opts()

    #########################################################
    sets.world_size = sets.gpus * sets.nodes  #
    os.environ['MASTER_ADDR'] = '135.125.87.167'  #
    os.environ['MASTER_PORT'] = '8888'  #
    mp.spawn(train, nprocs=sets.gpus, args=(sets,))  #
    #########################################################
