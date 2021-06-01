from flask import Flask, request, abort, Response
import logging
import os
from biencodernel.predict import predict_ner_nel
from biencodernel.utils import get_config_from_env, get_hp_information
from biencodernel.datasets import KBDataset, DataLoader, NELPredictDataset
import json

app = Flask(__name__)

logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

config = get_config_from_env()


@app.route('/predict', methods=['POST'])
def predict():
    dataset = request.get_json()
    if dataset:
        resulting_dataset = predict_ner_nel(dataset, input_length=input_length, batch_size=batch_size,
                                            bert_model=bert_model, kb_dl=kb_dl, biencoder_model=config['biencoder_model'])
        response = app.response_class(
            response=resulting_dataset,
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        abort(400)


@app.route('/setmodel/<model_name>', methods=['POST'])
def set_model(model_name: str):
    if model_name:
        config['biencoder_model'] = model_name
        return Response(response=f'set biencoder_model to: {model_name}', status=200)
    else:
        abort(400)


@app.route('/setmodel/default', methods=['POST'])
def set_model_default():
    config['biencoder_model'] = get_config_from_env()['biencoder_model']
    return Response(response=f'set biencoder_model back to default: {config["biencoder_model"]}', status=200)


if __name__ == '__main__':

    batch_size = int(get_hp_information(string=config['biencoder_model'], substring='bs'))
    input_length = int(get_hp_information(string=config['biencoder_model'], substring='il'))
    bert_model = 'bert-base-german-dbmdz-uncased' if 'uncased' in config[
        'biencoder_model'] else 'bert-base-german-cased'
    kb_dataset = KBDataset(path_to_json=config['paths']['kb'], max_length=input_length,
                           bert_model=bert_model)
    kb_dl = DataLoader(dataset=kb_dataset, batch_size=batch_size, drop_last=False, pin_memory=True)

    app.run(host='0.0.0.0', debug=True)
