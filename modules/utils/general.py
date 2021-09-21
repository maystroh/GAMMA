import torch
import numpy as np
from datetime import datetime
import socket
import torch.backends.cudnn as cudnn
from enum import Enum


class Modalities(Enum):
    OCT = 2
    Fundus = 3
    OCT_FUNDUS = 6


def set_modalities_to_load(sets):
    if 'OCT' in sets.modalities_to_load and 'Fundus' in sets.modalities_to_load:
        modalities = Modalities.OCT_FUNDUS
    else:
        if 'OCT' in sets.modalities_to_load:
            modalities = Modalities.OCT
        if 'Fundus' in sets.modalities_to_load:
            modalities = Modalities.Fundus
    sets.modalities_to_load = modalities

def get_task_name(sets):
    if sets.modalities_to_load == Modalities.OCT_FUNDUS:
        task_model = '_' + sets.task + '_' + sets.model_3D + '_' + str(sets.input_struct_C) + '_' + str(sets.input_struct_H) + \
                     '_' + str(sets.input_struct_D) + '_' + str(sets.input_struct_W) + '_' + sets.model_2D + \
                     '_' + str(sets.input_fundus_H) + '_' + str(sets.input_fundus_W)
    if sets.modalities_to_load == Modalities.OCT:
        task_model = '_' + sets.task + '_OCT_' + sets.model_3D + '_' + str(sets.input_struct_C) + '_' + str(sets.input_struct_H)\
                     + '_' + str(sets.input_struct_D) + '_' + str(sets.input_struct_W)
    if sets.modalities_to_load == Modalities.Fundus:
        task_model = '_' + sets.task + '_Fundus_' + sets.model_2D + '_' + str(sets.input_fundus_H) + '_' + str(sets.input_fundus_W)
    current_task_time = task_model + '_' + datetime.now().strftime('%b%d_%H-%M-%S')
    return current_task_time


def reproducibility(args, seed):
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
