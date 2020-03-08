# 政策大脑人物抽取

## 训练数据
```
test库中的person_annotation表
```

## 模型路径
```
ner模型: /data/nfsdata/nlp/extraction_projects/policy_brain/all_models/policy_brain/policy_brain_ner/new_org/
分类模型: /data/nfsdata/nlp/extraction_projects/policy_brain/all_models/policy_brain/extract_entity/org_classify_model
```

## 模型相关文件
```
extractor/library/dataset_readers/entity_bert_reader.py
extractor/library/models/entity_bert_classifier.py
extractor/library/predictor/bert_clf_predictor.py
```


## predictor
先使用ner抽取得到的文中的人物实体，然后根据实体和体裁预测它是否属于作者对象

## collector
根据不同的人物，以及在文章中出现的位置，和每个位置的身份来做综合判断该人物是否是作者对象