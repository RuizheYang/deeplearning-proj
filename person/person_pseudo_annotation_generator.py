"""
Model Description

@author: Xiayu Li
@contact: xiayu_li@shannonai.com
@version: 0.1
@license: Apache Licence
@file: person_pseudo_annotation_generator.py
@time: 2019/11/15 11:05 AM
"""

import os
import json

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from extractor.common.pseudo_annotation_generator import BasePseudoAnnotationGenerator
from extractor.person.person_processor import PersonProcessor
from extractor.article_type.article_type_proccessor import ArticleTypeV1Processor


class PersonPseudoAnnotationGenerator(BasePseudoAnnotationGenerator):
    def __init__(self, config_path: str, cuda_device=0, batch_size=64):
        super().__init__(config_path)
        self.peron_processor = PersonProcessor(config_path, cuda_device=cuda_device, batch_size=batch_size)
        self.article_type_processor = None

    def generate_from_dir(self, txt_dir: str):
        for txt_filename in os.listdir(txt_dir):
            if not txt_filename.endswith(".txt"):
                continue
            with open(txt_filename) as f_txt:
                content = f_txt.read()
                article_type = self.article_type_processor.predict
                processor_output = self.peron_processor.predict_title(content, article_type=self.article_type_processor.predict(content))
