import logging
import os
from typing import List, Dict, Tuple, Union
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertModel
from texoopy import Dataset, MentionAnnotation, NamedEntityAnnotation
from biencodernel.knn import FaissExactKNNIndex
from datetime import datetime

logger = logging.getLogger(__name__)


class CrossEntropyLoss:

    def __init__(self, device: str = 'cpu', reduction: str = 'mean'):
        """
        Cross Entropy Loss as mentioned in Humeau et al. 2019 - Poly Encoders
        :param device: cpu | cuda
        """
        logger.info('Initializing CrossEntropyLoss on device {} with reduction {}.'.format(device, reduction))
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss(reduction=reduction).to(self.device)

    def loss(self, mention_vecs: torch.Tensor, concept_vecs: torch.Tensor) -> torch.Tensor:
        assert concept_vecs.size() == mention_vecs.size()
        dot_products = torch.matmul(mention_vecs, concept_vecs.t()).to(self.device)
        y_target = torch.arange(0, concept_vecs.shape[0]).to(self.device)
        return self.loss_func(dot_products, y_target)

    def __call__(self, mention_vecs: torch.Tensor, concept_vecs: torch.Tensor) -> torch.Tensor:
        return self.loss(mention_vecs=mention_vecs, concept_vecs=concept_vecs)


class Encoder(nn.Module):

    def __init__(self, tokenizer: BertTokenizer, freeze_embeddings, bert_model: str = 'bert-base-german-cased'):
        """
        :param tokenizer: The tokenizer that was used to generate the token ids (Necessary for resizing the vocab of the model)
        :param pooling: CLS (CLS token) or AVG
        :param freeze_embeddings: freeze embedding layer as suggested by Hummeau et al. 2019
        """

        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        if freeze_embeddings:
            for param in list(self.bert.embeddings.parameters()):
                param.requires_grad = False
        self.bert.resize_token_embeddings(len(tokenizer))

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden_states, cls_tokens = self.bert(token_ids, return_dict=False)
        return cls_tokens


class BiEncoder:

    def __init__(self, device: str, tokenizer: BertTokenizer,
                 freeze_embeddings: bool = True, bert_model: str = 'bert-base-german-dbmdz-uncased'):
        self.device: str = device
        self.tokenizer: BertTokenizer = tokenizer
        self.bert_model = bert_model
        self.encoder_mention: Encoder = Encoder(tokenizer=self.tokenizer, freeze_embeddings=freeze_embeddings, bert_model=self.bert_model).to(self.device)
        self.encoder_concept: Encoder = Encoder(tokenizer=self.tokenizer, freeze_embeddings=freeze_embeddings, bert_model=self.bert_model).to(self.device)
        self.loss_func = CrossEntropyLoss(self.device)

    @classmethod
    def from_pretrained(cls, model_path: str, tokenizer: BertTokenizer,
                        device: str, bert_model: str) -> 'BiEncoder':
        biencoder = cls(device=device, tokenizer=tokenizer, bert_model=bert_model)
        mention_encoder_path = os.path.join(model_path, 'encoder_mention.statedict')
        concept_encoder_path = os.path.join(model_path, 'encoder_concept.statedict')
        biencoder.encoder_mention.load_state_dict(torch.load(mention_encoder_path, map_location=device))
        biencoder.encoder_concept.load_state_dict(torch.load(concept_encoder_path, map_location=device))
        return biencoder

    def predict(self, prediction_dataloader: DataLoader, prediction_dataset: Dataset, kb_dataloader: DataLoader, omp_num_threads: int = 1, use_exact_knn: bool = True) -> None:
        """
        Applies NEL model onto all MentionAnnotations and transforms them into NamedEntityAnnotations with confidence
        :return:
        """
        with torch.no_grad():
            self.encoder_mention.eval()
            self.encoder_concept.eval()

            kb_embeddings_cache = dict()
            for step_num, batch_data in tqdm(enumerate(kb_dataloader), desc='Generating KB candidate embeddings',
                                             total=len(kb_dataloader)):
                concept_ids, concept_tokens = batch_data
                concept_tokens = concept_tokens.to(self.device)
                concept_embeddings = self.encoder_concept(concept_tokens)
                for kb_id, concept_embedding in zip(concept_ids, concept_embeddings):
                    kb_embeddings_cache[kb_id] = concept_embedding.to('cpu')

            knn_index = FaissExactKNNIndex(kb_embeddings_cache, omp_num_threads=omp_num_threads)
            del kb_embeddings_cache

            predictions: Dict[int, Dict[str, Union[str, float]]] = dict()

            for step_num, batch_data in tqdm(enumerate(prediction_dataloader), desc='NEL prediction',
                                             total=len(prediction_dataloader)):

                mention_tokens, annotation_ids = batch_data
                mention_tokens = mention_tokens.to(self.device)
                mention_embeddings = self.encoder_mention(mention_tokens)

                for mention_embedding, annotation_id in zip(mention_embeddings.to('cpu'), annotation_ids):
                    annotation_id = annotation_id.item()
                    knn_ids, distances = zip(*knn_index.get_knn_ids_for_vector(mention_embedding, k=2))
                    confidence = max(distances[1] - distances[0], 0)
                    predictions[annotation_id] = {'refId': knn_ids[0], 'confidence': confidence}

            for doc in prediction_dataset.documents:
                new_annotations: List[NamedEntityAnnotation] = list()
                old_annotations: List[MentionAnnotation] = list()
                for ann in doc.annotations:
                    if ann.uid in predictions.keys() and type(ann) is MentionAnnotation:
                        new_annotations.append(NamedEntityAnnotation(
                            uid=ann.uid,
                            begin=ann.begin,
                            length=ann.length,
                            text=ann.text,
                            source='PRED',
                            refId=predictions[ann.uid]['refId'],
                            confidence=ann.confidence + predictions[ann.uid]['confidence'],
                        ))
                        old_annotations.append(ann)
                for ann in old_annotations:
                    doc.annotations.remove(ann)
                doc.annotations += new_annotations
