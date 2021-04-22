# BiEncoder-API
creates a Flask Server where to send json body with Dataset in Texoo format and model-descriptor to predict.
json-body looks like:
```{
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
    },
    "model_name": "train_default_eval-ner-nbs8-il50-bs64-lr0.0001-wu100-ep100-uncased-hpo-cuda"
}
```
