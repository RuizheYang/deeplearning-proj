"""
Model Description

@author: Xiayu Li
@contact: xiayu_li@shannonai.com
@version: 0.1
@license: Apache Licence
@file: identity_predictor.py
@time: 2019/11/13 7:54 PM
"""

import json
from typing import List

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from extractor.common.base_predictor import BasePredictor, NERAnswer, EntityIdentityPredictor


class PersonIdentityPredcitor(EntityIdentityPredictor):
    def __init__(self, model_path, cuda_device: int=0, batch_size=32):
        super().__init__(model_path, cuda_device, batch_size)
        override_json = {
            "dataset_reader.token_indexers.bert.pretrained_model":
                '/mnt/data/path/to/policy_brain/ner_service_data/chinese_bert',
            "model.text_field_embedder.token_embedders.bert.pretrained_model":
                '/mnt/data/path/to/policy_brain/ner_service_data/chinese_bert'
        }
        self.model = load_archive(self.model_path, self.cuda_device, overrides=json.dumps(override_json))
        # self.model.config.params['dataset_reader']['token_indexers']['bert'][
        #     'pretrained_model'] = '/mnt/data/path/to/policy_brain/ner_service_data/chinese_bert'
        # self.model.config.params['model']['text_field_embedder']['token_embedders']['bert'][
        #     'pretrained_model'] = '/mnt/data/path/to/policy_brain/ner_service_data/chinese_bert'
        self.predictor = Predictor.from_archive(self.model, "shannon_bert_multi_seq_clf")

