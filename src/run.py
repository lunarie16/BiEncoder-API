from flask import Flask, request, abort, Response
import logging
import os
from biencodernel.predict import predict_ner_nel

app = Flask(__name__)

logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())


@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    if body:
        dataset = body.get('dataset')
        model_name = body.get('model_name')
        resulting_dataset = predict_ner_nel(dataset, model_name)
        Response(resulting_dataset, 200)
    else:
        abort(400)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
