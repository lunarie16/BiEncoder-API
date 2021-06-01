from biencodernel.datasets import KBDataset, DataLoader, NELPredictDataset
from biencodernel.ner import NER
from biencodernel.biencoder import BiEncoder
from texoopy import Dataset
import os
import logging
import copy
from biencodernel.utils import get_config_from_env, get_hp_information
import torch

config = get_config_from_env()
logger = logging.getLogger(__name__)

ner = NER(
        device=config['device'],
        model_base_path=config['paths']['model'],
    )


def predict_ner_nel(dataset: dict, input_length: int, bert_model: str, batch_size: int, biencoder_model, kb_dl: DataLoader):
    logger.info(f'start prediction')
    dataset = fromJson2Dataset(dataset)

    logger.info(f'predicting NER')
    ner.predict(dataset)
    logger.error(f'\ndataset after NER \n {dataset.to_json()}\n\n')
    nel_ds = NELPredictDataset(dataset=dataset, max_length=input_length, allowed_ner_sources=['PRED'],
                               bert_model=bert_model)
    nel_dl = DataLoader(dataset=nel_ds, batch_size=batch_size, drop_last=False, pin_memory=True)
    nel = BiEncoder.from_pretrained(
        model_path=os.path.join(config['paths']['model'], biencoder_model),
        tokenizer=nel_ds.tokenizer,
        device=config['device'],
        bert_model=bert_model
    )
    logger.info(f'predicting NEL')
    nel.predict(prediction_dataloader=nel_dl, prediction_dataset=dataset, kb_dataloader=kb_dl)
    logger.info(f'this is the resulting dataset {dataset.to_json()}')
    return dataset.to_json()


def fromJson2Dataset(json_data: dict):
    json_data = copy.deepcopy(json_data)
    dataset = Dataset.from_json(json_data) if json_data is not None else None
    return dataset
