# BiEncoder-API

Config API in predict_nelner.yaml with values as:
```
- name: FLASK_ENV
      value: local
- name: PORT
      value: "5000"
- name: PATH_MODEL
      value: Path to model
- name: PATH_KB
      value: KB to use
- name: BERT_MODEL
      value: bert model 
- name: INPUT_LENGTH
      value: "50"
- name: BATCH_SIZE
      value: "82"
- name: BIENCODER_MODEL
      value: folder where statedicts are located
```
Input_length and Batch Size will be extracted from folder name as well


## Route to predict: 
_localhost:5000/predict_ and method: POST

creates a Flask Server where to send json body with Dataset in Texoo format to predict.
json-body looks like:
```
"dataset": {
        "name": "dataset_to_pred",
        "language": "de",
        "documents": [
            {
                "begin": 0,
                "length": 36,
                "text": "text text text",
                "id": "dummyid52",
                "title": "title",
                "language": "de",
                "type": "Optibar",
                "annotations": [],
                "class": "Document"
            },
            {
                "begin": 0,
                "length": 94,
                "text": "text text text",
                "id": "dummyid5",
                "title": "Grund der Anforderung",
                "language": "de",
                "type": "title",
                "annotations": [],
                "class": "Document"
            }
        ],
        "queries": []
}
```
