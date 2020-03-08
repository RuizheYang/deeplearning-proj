"""
Model Description

@author: Xiayu Li
@contact: xiayu_li@shannonai.com
@version: 0.1
@license: Apache Licence
@file: person_processor.py
@time: 2019/11/14 3:08 PM
"""

import json
import os
from typing import List, Tuple, Dict

from allennlp.predictors.predictor import Predictor
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel

from extractor.common.base_processor import BaseProcessor, EntityProcessor
from extractor.common.collector_input_output_record import PersonInputRecord
from extractor.person.ner_predictor import PersonNERPredictor
from extractor.person.identity_predictor import PersonIdentityPredcitor
from extractor.person.person_collector import PersonCollector
from extractor.utils.text_process import find_sentence
from extractor.common import CONFIG_PATH


class PersonProcessor(EntityProcessor):
    def __init__(self, config_path, cuda_device=0, batch_size=64):
        super().__init__(config_path, cuda_device=cuda_device, batch_size=batch_size)
        self.ner_engine = PersonNERPredictor(model_path=getattr(self.config.model_path, "person_ner"),
                                             cuda_device=cuda_device,
                                             batch_size=batch_size,
                                             bert_path=self.config.chinese_bert_path)
        PretrainedBertModel._cache.clear()
        self.identify_engine = PersonIdentityPredcitor(model_path=getattr(self.config.model_path, "person_classifier"),
                                                       cuda_device=cuda_device,
                                                       batch_size=batch_size)
        self.collector = PersonCollector(config_path)

    def process_batch_title_content(self, title_content_aritcle_type: List[Tuple[str, str]], meta_data: List[str]=None):
        merged_results = []
        for i, title_content_single in enumerate(title_content_aritcle_type):
            title, content = title_content_single
            article_type = meta_data[i]
            title_result, content_result = self.process_title_content(title, content, article_type)
            title_content_results = []
            for result in title_result:
                title_content_results.append(PersonInputRecord(PER_NAME=result['name'],
                                                                     HINT_SENT=result['hint_sent'],
                                                                     PER_ATTRIBUTE=result['attribute'],
                                                                     SOURCE_FIELD=result['source_field'],
                                                                     ARTICLE_TYPE=result['article_type'],
                                                                     TITLE=title,
                                                                     CONFIDENCE=result['confidence']))
            for result in content_result:
                title_content_results.append(PersonInputRecord(PER_NAME=result['name'],
                                                                     HINT_SENT=result['hint_sent'],
                                                                     PER_ATTRIBUTE=result['attribute'],
                                                                     SOURCE_FIELD=result['source_field'],
                                                                     ARTICLE_TYPE=result['article_type'],
                                                                     TITLE=title,
                                                                     CONFIDENCE=result['confidence']))
            title_content_results = self.collector.result_collect(title_content_results)
            merged_results.append(title_content_results)
        return merged_results

if __name__ == "__main__":
    person_processor = PersonProcessor(CONFIG_PATH, cuda_device=1, batch_size=64)
    ans = person_processor.process_content(content='习近平 在 参加 党 的 十九大 贵州省 代表团 讨论 时 强调   万众一心 开拓进取 把 新时代 中国 特色 社会主义 推向 前进                                               | 习近平在参加党的十九大贵州省代表团讨论时强调万众一心开拓进取把新时代中国特色社会主义推向前进10月19日,习近平同志参加党的十九大贵州省代表团讨论。新华社记者李涛摄习近平同志19日上午在参加党的十九大贵州省代表团讨论时强调,党的十九大报告进一步指明了党和国家事业的前进方向,是我们党团结带领全国各族人民在新时代坚持和发展中国特色社会主义的政治宣言和行动纲领。要深刻学习领会中国特色社会主义进入新时代的新论断,深刻学习领会我国社会主要矛盾发生变化的新特点,深刻学习领会分两步走全面建设社会主义现代化国家的新目标,深刻学习领会党的建设的新要求,激励全党全国各族人民万众一心,开拓进取,把新时代中国特色社会主义推向前进。贵州省代表团讨论气氛热烈。孙志刚、谌贻琴、余留芬、潘克刚、周建琨、钟晶、杨波、张蜀新、黄俊琼等9位代表分别结合实际,对报告发表了意见,畅谈了认识体会。大家认为,党的十九大报告是一个实事求是、与时俱进,凝心聚力、催人奋进的报告,是一个动员和激励全党为决胜全面建成小康社会,夺取新时代中国特色社会主义伟大胜利,实现中华民族伟大复兴的中国梦不懈奋斗的报告,一致表示拥护这个报告。习近平边听边记,同代表们深入讨论。六盘水市盘州市淤泥乡岩博村党委书记余留芬发言时说,广大农民对党的十九大报告提出土地承包到期后再延长30年的政策十分满意,习近平听了十分高兴,说这是要给广大农民吃个“定心丸”。遵义市播州区枫香镇花茂村党总支书记潘克刚讲到乡村农家乐旅游成为乡亲致富新路,习近平说既要鼓励发展乡村农家乐,也要对乡村旅游作分析和预测,提前制定措施,确保乡村旅游可持续发展。毕节市委书记周建琨讲到把支部建在生产小组上、发展脱贫攻坚讲习所,习近平强调,新时代的农民讲习所是一个创新,党的根基在基层,一定要抓好基层党建,在农村始终坚持党的领导。黔西南州贞丰县龙场镇龙河村卫生室医生钟晶讲到农村医疗保障问题,习近平详细询问现在农民一年交多少医疗保险费、贫困乡村老百姓生产生活条件有没有改善。贵州六盘水市钟山区大湾镇海嘎村党支部第一书记杨波谈了自己连续8年坚持当驻村第一书记、带领乡亲脱贫致富的体会,习近平表示,对在脱贫攻坚一线的基层干部要关心爱护,各方面素质好、条件具备的要提拔使用,同时要鼓励年轻干部到脱贫攻坚一线去历练。习近平还对黔东南州镇远县江古镇中心小学教师黄俊琼说,老少边穷地区的教育培训工作要加大力度,让更多乡村和基层教师受到专业培训。在认真听取代表发言后,习近平表示,很高兴作为贵州省代表团的代表参加讨论。习近平向在座各位代表和贵州全省各族干部群众致以诚挚的问候。习近平指出,5年来,贵州认真贯彻落实党中央决策部署,各方面工作不断有新进展。综合实力显著提升,脱贫攻坚成效显著,生态环境持续改善,改革开放取得重大进展,人民群众获得感不断增强,政治生态持续向好。贵州取得的成绩,是党的十八大以来党和国家事业大踏步前进的一个缩影。这从一个角度说明了党的十八大以来党中央确定的大政方针和工作部署是完全正确的。习近平希望贵州的同志全面贯彻落实党的十九大精神,大力培育和弘扬团结奋进、拼搏创新、苦干实干、后发赶超的精神,守好发展和生态两条底线,创新发展思路,发挥后发优势,决战脱贫攻坚,决胜同步小康,续写新时代贵州发展新篇章,开创百姓富、生态美的多彩贵州新未来。习近平指出,中国特色社会主义进入了新时代,这是我国发展新的历史方位。作出这个重大政治判断,是一项关系全局的战略考量,我们必须按照新时代的要求,完善发展战略和各项政策,推进和落实各项工作。我国社会主要矛盾的变化是关系全局的历史性变化,对党和国家工作提出了许多新要求,我们要深入贯彻新发展理念,着力解决好发展不平衡不充分问题,更好满足人民多方面日益增长的需要,更好推动人的全面发展、全体人民共同富裕。我们要紧密结合党的十九大对我国未来发展作出的战略安排,推进党和国家各项工作,特别是要保持各项战略、工作、政策、措施的连续性和前瞻性,一步接一步,连续不断朝着我们确定的目标前进。习近平强调,办好中国的事情,关键在党。全面从严治党不仅是党长期执政的根本要求,也是实现中华民族伟大复兴的根本保证。我们党要团结带领人民进行伟大斗争、推进伟大事业、实现伟大梦想,必须毫不动摇把党建设得更加坚强有力。全面从严治党永远在路上。在全面从严治党这个问题上,我们不能有差不多了,该松口气、歇歇脚的想法,不能有打好一仗就一劳永逸的想法,不能有初见成效就见好就收的想法。必须持之以恒、善作善成,把管党治党的螺丝拧得更紧,把全面从严治党的思路举措搞得更加科学、更加严密、更加有效,推动全面从严治党向纵深发展。各级党组织和全体党员、各级领导干部必须坚决维护党中央权威,坚决服从党中央集中统一领导,把“四个意识”落实在岗位上、落实在行动上,不折不扣执行党中央决策部署,始终在思想上政治上行动上同党中央保持高度一致。习近平指出,大会之后,要认真组织好党的十九大精神宣传教育工作和学习培训工作,注重宣传各地区各部门学习贯彻的具体举措和实际行动,注重反映基层干部群众学习贯彻的典型事迹和良好风貌。要充分利用各种宣传形式和手段,采取人民群众喜闻乐见的形式,推动党的十九大精神进企业、进农村、进机关、进校园、进社区、进军营,让干部鼓足干劲。要组织好集中宣讲活动,把党的十九大精神讲清楚、讲明白,让老百姓听得懂、能领会、可落实。',
                                                article_type='新闻报道_讲话稿')

    print(ans)
