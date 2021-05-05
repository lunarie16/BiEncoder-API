import logging
import os
from typing import Dict, Any, Tuple
from typing import List
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class StringIdTensorDataset(Dataset):
    """
    Holds tensors with a String as identifier, e.g. a CUI.
    """

    def __init__(self, data: List[Tuple[str, torch.Tensor]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_device() -> Tuple[str, int]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = None
    if str(device) == 'cuda':
        torch.cuda.empty_cache()
        n_gpu = torch.cuda.device_count()
        logger.info("Number of GPUs: {}".format(n_gpu))
    return str(device), n_gpu


def get_config_from_env() -> Dict[str, Any]:
    config = {
              'bert_model': os.getenv('BERT_MODEL', 'bert-base-german-dbmdz-uncased'),
              'batch_size': int(os.getenv('BATCH_SIZE', '64')), 'epochs': int(os.getenv('EPOCHS', '100')),
              'input_length': int(os.getenv('INPUT_LENGTH', '50')),
              'device': (get_device())[0], 'num_gpu': (get_device())[1],
              'biencoder_model': os.getenv('BIENCODER_MODEL', 'train_default_nbs8-il50-bs58-lr0.0005361114500366287-wu100-ep125-uncased-7-cuda'),
              'paths': {
                'kb': os.getenv('PATH_KB', '/data/datasets/krohne_products_description_texoo.json'),
                'model': os.getenv('PATH_MODEL', '//data/biencoder/model/')}}

    logger.info('Using {}'.format(config['device']))

    return config


def get_hp_information(string: str, substring: str):
    splitted = string.split('-')
    for s in splitted:
        if substring in s and 'nbs' not in s:
            return s.replace(substring, '')
