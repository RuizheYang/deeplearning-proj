# -*- coding: utf-8 -*-
import logging
from typing import Dict

# from allennlp.modules.token_embedders.bert_token_embedder import
import numpy
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from extractor.library.modules.focal_loss_ner import FocalLoss

@Model.register("label_emb_ner_bert_tagger")
class LabelEmbNerModel(Model):
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 type_field_embedder: TextFieldEmbedder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        # self.hidden2tag = torch.nn.Linear(in_features=self.text_field_embedder.get_output_dim(),
        #                                   out_features=vocab.get_vocab_size('labels'))
        self.type_field_embedder = type_field_embedder
        self.hidden2medium = torch.nn.Linear(in_features=self.text_field_embedder.get_output_dim()+20, out_features=int((self.text_field_embedder.get_output_dim()+20)/2))
        self.dropout = torch.nn.Dropout(0.2)
        self.medium2tag = torch.nn.Linear(in_features=int((self.text_field_embedder.get_output_dim()+20)/2), out_features=vocab.get_vocab_size("labels"))
        self.loss = FocalLoss()
        self.metrics = {
            # "accuracy": CategoricalAccuracy(),
            "f1_measure": SpanBasedF1Measure(vocabulary=vocab, tag_namespace="labels", ignore_classes=[""])
        }

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                types: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.text_field_embedder(sentence)
        embeddings = self.dropout(embeddings)
        type_embedding = self.type_field_embedder(types)
        type_embedding = type_embedding.repeat(1, embeddings.shape[1], 1)
        final_embedding = torch.cat([embeddings, type_embedding], dim=-1)
        medium_logits = self.hidden2medium(final_embedding)
        medium_logits = torch.nn.ReLU()(medium_logits)
        medium_logits = self.dropout(medium_logits)
        tag_logits = self.medium2tag(medium_logits)
        output = {"tag_logits": tag_logits, "mask": mask}
        if labels is not None:
            for metric in self.metrics:
                self.metrics[metric](tag_logits, labels, mask)
            # 这个使用的是focalloss，可以测试不实用focalloss的原始版本
            # output["loss"] = self.loss(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.metrics["f1_measure"].get_metric(reset)
        # metrics.pop("precision-overall")
        # metrics.pop("recall-overall")
        # metrics.pop("f1-measure-overall")
        # metrics["accuracy"] = self.metrics["accuracy"].get_metric(reset)
        return metrics

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['tag_logits'].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [[self.vocab.get_token_from_index(x, namespace="labels")
                   for x in indices] for indices in argmax_indices]
        output_dict['labels'] = labels
        return output_dict