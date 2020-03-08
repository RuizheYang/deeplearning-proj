# -*- coding: UTF-8 -*-
"""
@author: liangjunjun
@contact: liangjunjun@pku.edu.cn
@version: v0.0.1
@file: person_collector.py
@time: 2019/12/9 19:55
@desc: 
"""

import os
from collections import namedtuple, defaultdict
import json
from typing import Dict, List

from extractor.common.base_collector import BaseCollector
from extractor.common import CONFIG_PATH
from extractor.common.collector_input_output_record import PersonOutputRecord, PersonInputRecord
from extractor.utils.namedtuple_to_dict import namedtuple_to_dict

Person = namedtuple('Person', ['ID', 'NAME', 'AUTHORITY', 'REGION', 'ORDER'])


class PersonCollector(BaseCollector):
    def __init__(self, config_path: str = CONFIG_PATH):
        super().__init__(config_path)
        self.person_dict = self.parse_collector_material()
        # 定义一些重要人物，做一些特殊规则
        self.primary_person = ['习近平', '李克强', '栗战书', '汪洋', '王沪宁', '赵乐际', '韩正']

    def parse_collector_material(self) -> Dict[str, Person]:
        """
        解析读到的人物候选表
        :return:
        """
        person_dict = {}
        with open(os.path.join(self.data_path, 'final_person.jsonl')) as fp_in:
            for per in fp_in.readlines():
                per = json.loads(per)
                person_dict[per['NAME']] = Person(**per)
        return person_dict

    def result_collect(self, records: List[PersonInputRecord]) -> List[Dict]:
        """
        collector类调用接口，输入是经过processor抽取得到的新闻结果，输出是经过一些规则过滤映射后得到的结果
        :param records:
        :return:
        """
        person_list = []
        found_person = []
        primary_person_count_dict = defaultdict(int)
        article_type = ''
        for record in records:
            article_type = record.ARTICLE_TYPE
            name = record.PER_NAME.replace('・', '·')
            attribute = record.PER_ATTRIBUTE
            hint_sent = record.HINT_SENT
            title = record.TITLE
            if self.hard_rule_filter(record):
                continue
            # 判断重要人物,只要不同时满足(attribute不等于无且句子来自于title),则需要比较无和作者对象的数量
            if name in self.primary_person and not (hint_sent == title and attribute != '无'):
                if attribute == '无':
                    primary_person_count_dict[name] -= 1
                else:
                    primary_person_count_dict[name] += 1
                continue
            if not name or name not in self.person_dict or attribute == '无':
                continue
            if name in found_person:
                continue
            person_list.append(namedtuple_to_dict(
                PersonOutputRecord(name=name,
                                   region=self.person_dict[name].REGION,
                                   level=self.person_dict[name].AUTHORITY,
                                   order=self.person_dict[name].ORDER,
                                   attribute=attribute,
                                   hint_sent=hint_sent,
                                   article_type=article_type,
                                   source_field=record.SOURCE_FIELD,
                                   confidence=record.CONFIDENCE)))
            found_person.append(name)
        for primary_person, count in primary_person_count_dict.items():
            if count > 0 and primary_person not in found_person:
                person_list.append(namedtuple_to_dict(
                    PersonOutputRecord(name=primary_person, region=self.person_dict[primary_person].REGION,
                                       level=self.person_dict[primary_person].AUTHORITY,
                                       order=self.person_dict[primary_person].ORDER,
                                       attribute='',
                                       hint_sent='',
                                       article_type=article_type,
                                       source_field='',
                                       confidence=1.0)))
                found_person.append(primary_person)
        return person_list

    def hard_rule_filter(self, record: PersonInputRecord):
        """
        根据badcase定义的规则，如果需要增加一些规则，就在该函数里面添加相应的正则或者相应的判断逻辑
        :param record:
        :return:
        """
        name = record.PER_NAME.replace('・', '·')
        attribute = record.PER_ATTRIBUTE
        hint_sent = record.HINT_SENT
        title = record.TITLE
        article_type = record.ARTICLE_TYPE
        # 过滤掉学习贯彻，学习传达，学习重要人物
        if name in self.primary_person and attribute != '无':
            for study in ['学习贯彻', '学习传达', '学习']:
                if study + name in hint_sent:
                    return True
        # 过滤掉习近平主席特使***/习近平特使
        if name == '习近平' and ('习近平主席特使' in hint_sent or '习近平特使' in hint_sent):
            return True
        # 过滤吉林省
        if name == "吉林" and ("吉林省" in hint_sent or "吉林药监局" in hint_sent):
            return True
        # 如果标题中含有大使
        if name == '习近平' and (
                '大使' in title or '使馆' in title or '领事' in title) and article_type != '人事任免' and '习近平' not in title:
            return True
        # 如果type是官方解读并且人名是习近平或者李克强
        if (name == '习近平' or name == '李克强') and article_type == '官方解读':
            return True
        return False


if __name__ == '__main__':
    collector = PersonCollector(CONFIG_PATH)
    per_input = PersonInputRecord(PER_NAME='习近平', PER_ATTRIBUTE='作者_对象',
                                  HINT_SENT='月19日上午,市委办公室系统召开“两学一做”专题学习会议,及时传达学习贯彻市委中心组“两学一做”专题学习会议精神。',
                                  TITLE='月19日上午,市委办公室系统召开“两学一做”专题学习会议,及时传达学习贯彻市委中心组“两学一做”专题学习会议精神。',
                                  ARTICLE_TYPE='人事任免',SOURCE_FIELD='title',CONFIDENCE=0.96)
    collect_per = collector.result_collect([per_input])
    print(collect_per)
