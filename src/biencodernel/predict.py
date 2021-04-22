from biencodernel.datasets import KBDataset, DataLoader, NELPredictDataset
from biencodernel.ner import NER
from biencodernel.biencoder import BiEncoder
from texoopy import Dataset
import os
import logging
import copy
from biencodernel.utils import get_config_from_env, get_hp_information

config = get_config_from_env()
logger = logging.getLogger(__name__)


def predict_ner_nel(dataset: dict, biencoder_model: str):
    batch_size = int(get_hp_information(string=biencoder_model, substring='bs'))
    input_length = int(get_hp_information(string=biencoder_model, substring='il'))
    bert_model = 'bert-base-german-dbmdz-uncased' if 'uncased' in biencoder_model else 'bert-base-german-cased'
    dataset = fromJson2Dataset(dataset)
    kb_dataset = KBDataset(path_to_json=config['paths']['kb'], max_length=input_length,
                           bert_model=bert_model)
    kb_dl = DataLoader(dataset=kb_dataset, batch_size=batch_size, drop_last=False, pin_memory=True)
    ner = NER(
        device=config['device'],
        model_base_path=config['paths']['model'],
    )
    ner.predict(dataset)
    nel_ds = NELPredictDataset(dataset=dataset, max_length=input_length, allowed_ner_sources=['PRED'],
                               bert_model=bert_model)
    nel_dl = DataLoader(dataset=nel_ds, batch_size=batch_size, drop_last=False, pin_memory=True)

    nel = BiEncoder.from_pretrained(
        model_path=os.path.join(config['paths']['model'], biencoder_model),
        tokenizer=nel_ds.tokenizer,
        device=config['device'],
        bert_model=bert_model
    )
    nel.predict(prediction_dataloader=nel_dl, prediction_dataset=dataset, kb_dataloader=kb_dl)
    logger.info(f'this is the resulting dataset {dataset.to_json()}')
    return dataset.to_json()


def fromJson2Dataset(json_data: dict):
    json_data = copy.deepcopy(json_data)
    dataset = Dataset.from_json(json_data) if json_data is not None else None
    return dataset
