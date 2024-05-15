import math
import torch
import numpy as np
import torch.nn.functional as F


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except:
        pass


def set_device():
    """
    Function for setting the device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
    except:
        device = torch.device("cpu")
    return device


def to_tensor(x) -> torch.tensor:
    if not isinstance(x, torch.tensor):
        x = torch.tensor(x)

    return x
