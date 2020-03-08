"""
Model Description

@author: Xiayu Li
@contact: xiayu_li@shannonai.com
@version: 0.1
@license: Apache Licence
@file: ner_predictor.py
@time: 2019/11/13 7:53 PM
"""

import os
import json

from shannon_ner.shannon_ner import ShannonNER
from allennlp.predictors import Predictor

from extractor.common.base_predictor import BasePredictor, NERAnswer, NERPredictor


class PersonNERPredictor(NERPredictor):
    def __init__(self, model_path, bert_path, cuda_device=0, batch_size=64):
        super().__init__(model_path, cuda_device=cuda_device, batch_size=batch_size, bert_path=bert_path)

    def post_process(self, results, content):
        final_answers_list = []
        mark_bool = 0
        if "《" in content and "》" in content:
            mark_bool = 1
            mark_s = content.index("《")
            mark_e = content.index("》")

        quote_bool = 0
        if "“" in content and "”" in content:
            quote_bool = 1
            quote_s = content.index("“")
            quote_e = content.index("”")

        jilin_bool = 0
        if "吉林省" in content:
            jinlin_bool = 1

        for result in results:
            # 1.
            if result.end - result.begin == 1:
                continue
                # 2.
            if mark_bool == 1:
                if result.begin >= mark_s and result.end <= mark_e:
                    continue
                    # 3.
            if quote_bool == 1:
                if result.begin >= quote_s and result.end <= quote_e:
                    continue

                    # 4.
            if jilin_bool == 1:
                if result.term == "吉林":
                    continue

                    # 5.
            if " " in result.term:
                split_s = result.term.index(" ")
                final_answers_list.append(NERAnswer(answer=result.term[:split_s],
                                                 span_start=result.begin,
                                                 span_end=result.begin + split_s,
                                                 tag=result.tag))
                final_answers_list.append(NERAnswer(answer=result.term[split_s + 1:],
                                                 span_start=result.begin + split_s + 1,
                                                 span_end=result.end,
                                                 tag=result.tag))

            if "中央纪委国家监委" == result.term:
                final_answers_list.append(NERAnswer(answer="中央纪委",
                                                 span_start=result.begin,
                                                 span_end=result.begin + 4,
                                                 tag=result.tag))
                final_answers_list.append(NERAnswer(answer="国家监委",
                                                 span_start=result.begin + 4,
                                                 span_end=result.end,
                                                 tag=result.tag))

            if "中共中央办公厅国务院办公厅" == result.term:
                final_answers_list.append(NERAnswer(answer="中共中央办公厅",
                                                 span_start=result.begin,
                                                 span_end=result.begin + 7,
                                                 tag=result.tag))
                final_answers_list.append(NERAnswer(answer="国务院办公厅",
                                                 span_start=result.begin + 7,
                                                 span_end=result.end,
                                                 tag=result.tag))

            if "中办国办" == result.term:
                final_answers_list.append(NERAnswer(answer="中办",
                                                 span_start=result.begin,
                                                 span_end=result.begin + 2,
                                                 tag=result.tag))
                final_answers_list.append(NERAnswer(answer="国办",
                                                 span_start=result.begin + 2,
                                                 span_end=result.end,
                                                 tag=result.tag))

            if "公司" in result.term or "学校" in result.term or "大学" in result.term or "医院" in result.term:
                continue

                ######################
            answer = NERAnswer(answer=result.term,
                            span_start=result.begin,
                            span_end=result.end,
                            tag=result.tag)

            final_answers_list.append(answer)

        return final_answers_list


