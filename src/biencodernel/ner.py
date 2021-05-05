import logging
import os
import flair
import torch
from flair.data import Corpus, Sentence
from flair.models import SequenceTagger
from texoopy import Dataset, Document, Annotation, MentionAnnotation
from tqdm import tqdm

logger = logging.getLogger('__name__')


def _teXooDocument2FlairBIOESSentence(document: Document, apply_sentence_split: bool = False) -> Sentence:
    if apply_sentence_split:
        raise NotImplemented()  # TODO implement me

    flair_sentence: Sentence = Sentence(document.text, use_tokenizer=True)
    for token in flair_sentence.tokens:
        token.add_tag('BIOES', 'O-ENT')
    annotation: Annotation
    for ann in document.annotations:
        begin = ann.begin
        end = ann.begin + ann.length
        tokens = list()
        for token in flair_sentence.tokens:
            if token.start_pos >= begin and token.end_pos <= end:
                tokens.append(token)
        if len(tokens) == 1:
            if tokens[0].get_tag('BIOES').value != 'O-ENT':
                continue
            tokens[0].add_tag('BIOES', 'S-ENT')
        elif len(tokens) > 1:
            # just make sure that tokens are sorted properly
            tokens = sorted(tokens, key=lambda tok: tok.start_pos)
            existing_tags = set([tok.get_tag('BIOES').value for tok in tokens])
            if existing_tags != {'O-ENT'}:
                # Some tokens are already tagged, skip annotation
                continue
            for tok in tokens:
                tok.add_tag('BIOES', 'I-ENT')
            tokens[0].add_tag('BIOES', 'B-ENT')
            tokens[-1].add_tag('BIOES', 'E-ENT')
    return flair_sentence


class NER:
    def __init__(self, model_base_path: str,  device: str = 'cpu'):
        """
        Initialises an abstraction layer over the Flair SequenceTagger
        :param device:
        :param model_base_path: Path to model base folder
        """
        self.device = device
        self.model_path = os.path.join(model_base_path, 'ner')
        flair.device = torch.device(device)
        try:
            self.ner: SequenceTagger = SequenceTagger.load(os.path.join(self.model_path, 'best-model.pt'))
        except FileNotFoundError:
            logger.info('No NER model found, needs to be trained.')

    def predict(self, dataset: Dataset) -> None:
        """
        Applies NER model onto the given dataset and creates MentionAnnotations of type PREDICT with confidence score.
        :param dataset:
        :return:
        """
        for doc in tqdm(dataset.documents, desc='Applying pre-trained NER model'):
            flair_doc = _teXooDocument2FlairBIOESSentence(doc)
            logger.error(f'this is a flair BIOES senteces {flair_doc}')
            self.ner.predict(flair_doc)
            for entity in flair_doc.get_spans('BIOES'):
                doc.annotations.append(MentionAnnotation(
                    begin=entity.start_pos,
                    length=entity.end_pos - entity.start_pos,
                    text=entity.text,
                    source='PRED',
                    confidence=entity.score
                ))
                logger.info(f'fount this entity {entity}: {entity.text}')