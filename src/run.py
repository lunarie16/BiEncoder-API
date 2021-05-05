from flask import Flask, request, abort, Response
import logging
import os
from biencodernel.predict import predict_ner_nel
from biencodernel.utils import get_config_from_env, get_hp_information
from biencodernel.datasets import KBDataset, DataLoader, NELPredictDataset


app = Flask(__name__)

logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

config = get_config_from_env()


@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    if body:
        dataset = body.get('dataset')
        resulting_dataset = predict_ner_nel(dataset, input_length=input_length, batch_size=batch_size,
                                            bert_model=bert_model, kb_dl=kb_dl, biencoder_model=config['biencoder_model'])
        Response(resulting_dataset, 200)
    else:
        abort(400)


if __name__ == '__main__':

    batch_size = int(get_hp_information(string=config['biencoder_model'], substring='bs'))
    input_length = int(get_hp_information(string=config['biencoder_model'], substring='il'))
    bert_model = 'bert-base-german-dbmdz-uncased' if 'uncased' in config[
        'biencoder_model'] else 'bert-base-german-cased'
    kb_dataset = KBDataset(path_to_json=config['paths']['kb'], max_length=input_length,
                           bert_model=bert_model)
    kb_dl = DataLoader(dataset=kb_dataset, batch_size=batch_size, drop_last=False, pin_memory=True)

    app.run(host='0.0.0.0', debug=True)
