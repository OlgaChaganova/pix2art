import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import Pix2PixDataset
from utils import config

def get_dataloader(type, batch_size=2):

    DIR = Path(config.PATH_TO_DATA)
    files = np.random.permutation(list(DIR.rglob('*.jpg')))

    if type == 'train':
        files = files[:-9]
    elif type == 'val':
        files = files[-9:-3]
        batch_size = 6
    elif type == 'test':
        files = files[-3:]
        batch_size = 3

    data = Pix2PixDataset(files)
    data_loader = DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)
    return data_loader

