import logging
import math
import os
import codecs
import torch
import ujson as json
from typing import Tuple, Union, Dict, List
from pathlib import Path
from texoopy import Dataset, Document, Annotation, NamedEntityAnnotation, MentionAnnotation
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class TokenTools:

    @staticmethod
    def get_tokenizer(bert_model: str = 'bert-base-german-cased', ms_token: str = '[ms]', me_token: str = '[me]',
                      ent_token: str = '[ent]') -> BertTokenizer:
        # TODO: check if more sexy way than if else
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=('uncased' in bert_model))
        tokenizer.add_special_tokens({'additional_special_tokens': [ent_token, ms_token, me_token]})

        tok = tokenizer.encode('{} {} {} {} {} {}'.format(
            ent_token.upper(),
            ent_token,
            ms_token.upper(),
            ms_token,
            me_token.upper(),
            me_token
        ))
        if 'uncased' in bert_model:
            if len(tok) != 8:
                logger.error(tok)
                logger.error(
                    'An error occurred during the initialization of the tokenizer w.r.t. the special tokens. Length of tokenizer {}'.format(
                        len(tok)))
                # exit(1)
        return tokenizer

    @staticmethod
    def pad(token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) < max_length:
            token_ids = token_ids + [0] * (max_length - len(token_ids))
        return token_ids

    @staticmethod
    def center_mention(tokens: List[str], max_length: int, ms_token: str = '[ms]', me_token: str = '[me]'):
        """
        Centers and crops the mention based on the position of the boundary tokens and adds [CLS] and [SEP]
        :param tokens: tokens (without [CLS] and [SEP])
        :param max_length: maximal length of the final output (no padding)
        :param ms_token: start token of mention
        :param me_token: end token of mention
        :return: List of tokens
        """
        idx_ms, idx_me = tokens.index(ms_token), tokens.index(me_token)
        num_tokens_mention = idx_me - idx_ms + 1
        max_context_length = int(math.ceil((max_length - num_tokens_mention) / 2 - 1))
        tokens_mention = tokens[idx_ms:idx_me + 1]
        tokens_left = tokens[max(0, idx_ms - max_context_length):idx_ms]
        tokens_right = tokens[idx_me + 1:min(len(tokens), idx_me + 1 + max_context_length)]
        tokens_cropped = ['[CLS]'] + tokens_left + tokens_mention + tokens_right
        tokens_with_sep = tokens_cropped[:max_length - 1] + ['[SEP]']
        return tokens_with_sep

    @staticmethod
    def filter_out_special_tokens(text: str):
        """
        Replaces every [ or ] with < or >, this takes care of special tokens that are part of the text.
        :return: text
        """
        return text.replace('[', '<').replace(']', '>')


class KBDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_json: str, max_length: int, ent_token='[ent]',
                 bert_model: str = 'bert-base-german-cased'):
        self.path_to_json: str = path_to_json
        self.max_length: int = max_length
        self.ent_token = ent_token
        self.bert_model = bert_model
        self.tokenizer: BertTokenizer = TokenTools.get_tokenizer(ent_token=self.ent_token, bert_model=self.bert_model)
        self.tokenized_kb_concepts: List[Tuple[str, torch.Tensor]] = []
        self.tokenized_kb_concepts_by_kb_id: Dict[str, torch.Tensor] = dict()
        with codecs.open(self.path_to_json, 'r', 'utf-8-sig') as f:
            dataset: Dataset = Dataset.from_json(json.load(f))
        doc: Document
        for doc in tqdm(dataset.documents, desc='Processing knowledge base {}'.format(dataset.name)):
            kb_id = doc.id
            concept_name = doc.title
            concept_description = doc.text
            token_ids = self.tokenizer.encode(
                '{} {} {}'.format(concept_name, self.ent_token, concept_description),
                max_length=self.max_length, padding='max_length', truncation=True)
            token_ids_tensor = torch.tensor(token_ids)
            self.tokenized_kb_concepts.append((kb_id, token_ids_tensor))
            self.tokenized_kb_concepts_by_kb_id[kb_id] = token_ids_tensor

    def __len__(self):
        return len(self.tokenized_kb_concepts)

    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]:
        """
        :param idx:
        :return: (kb_id, token ids as tensor)
        """
        return self.tokenized_kb_concepts[idx]

    def get_tensor(self, kb_id: str) -> torch.Tensor:
        try:
            return self.tokenized_kb_concepts_by_kb_id[kb_id]
        except KeyError:
            logger.error('Concept {} not found in knowledge base, using dummy encoding!'.format(kb_id))
            # return dummy word piece tokens so the dataloader never returns None
            return torch.tensor(self.tokenizer.encode('{} {} {}'.format(kb_id, self.ent_token, ''),
                                                      max_length=self.max_length,
                                                      padding='max_length',
                                                      truncation=True)
                                )


class NELPredictDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, max_length: int, bert_model: str, ms_token='[ms]', me_token='[me]',
                 allowed_ner_sources: List[str] = ['PRED']):
        """
        Initializes a TeXoo NELPredictDataset that is used to tokenize and to feed MentionAnnotations through the
        Bi-Encoder in order to get predictions and transform the MentionAnnotations to NamedEntityAnnotations.
        In order to be able to maintain the association between MentionAnnotation and prediction, every
        MentionAnnotation is given a unique id that is part of the return of __get_item__().
        """

        self.max_length: int = max_length
        self.ms_token: str = ms_token
        self.me_token: str = me_token
        self.tokenizer: BertTokenizer = TokenTools.get_tokenizer(ms_token=self.ms_token, me_token=self.me_token,
                                                                 bert_model=bert_model)
        self.tokenized_mentions: List[torch.Tensor] = []
        self.annotation_ids: List[int] = []
        self.dataset = dataset
        doc: Document
        ann: MentionAnnotation

        id = 0
        for doc in self.dataset.documents:
            for ann in doc.annotations:
                ann.uid = id
                id += 1

        for doc in tqdm(self.dataset.documents, desc='Processing dataset {}'.format(dataset.name)):
            for ann in doc.annotations:
                if type(ann) is not MentionAnnotation or ann.source not in allowed_ner_sources:
                    continue
                doc_text = TokenTools.filter_out_special_tokens(doc.text)
                mention_with_context = '{}{} {} {}{}'.format(
                    doc_text[0:ann.begin],
                    self.ms_token,
                    doc_text[ann.begin:ann.begin + ann.length],
                    self.me_token,
                    doc_text[ann.begin + ann.length:]
                )
                tokens = self.tokenizer.tokenize(mention_with_context)
                tokens = TokenTools.center_mention(tokens=tokens, max_length=self.max_length, ms_token=self.ms_token,
                                                   me_token=self.me_token)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                token_ids = TokenTools.pad(token_ids, max_length=self.max_length)
                self.tokenized_mentions.append(torch.tensor(token_ids))
                self.annotation_ids.append(ann.uid)

    def get_dataset_with_annotation_ids(self):
        return self.dataset

    def __len__(self):
        return len(self.tokenized_mentions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        :param idx:
        :return: mention tokens, annotation id
        """
        return self.tokenized_mentions[idx], self.annotation_ids[idx]
