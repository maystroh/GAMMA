
import torch
import imageio
import random as rn
import numpy as np
import os

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    input_tensor = img2d(input_tensor)
    imageio.imwrite(filename, input_tensor)


def config_torch_np_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    rn.seed(seed)  # 2. Set `python` built-in pseudo-random generator at a fixed value
    np.random.seed(seed)  # 3. Set `numpy` pseudo-random generator at a fixed value
    torch.manual_seed(seed)  # 4. Set `tensorflow` pseudo-random generator at a fixed value
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def img2d(input_tensor: torch.Tensor):
    if input_tensor.ndim == 3:
        input_tensor = input_tensor[0, :, :]
    elif input_tensor.ndim == 2:
        input_tensor = input_tensor
    elif input_tensor.ndim == 4:
        input_tensor = input_tensor[0, 0, :, :]
    elif input_tensor.ndim == 5:
        input_tensor = input_tensor[0, 0, 0, :, :]
    return input_tensor