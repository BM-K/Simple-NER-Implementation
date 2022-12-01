import torch

from torch import nn

from transformers import (
        AutoConfig,
        BertForTokenClassification,
        RobertaForTokenClassification
)

class ClassificationModel(nn.Module):
    def __init__(self, args, label_dict):
        super(ClassificationModel, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(self.args.model,
                                                 num_labels=len(label_dict),
                                                 )

        if 'roberta' in self.args.model:
            self.model = RobertaForTokenClassification.from_pretrained(self.args.model,
                                                                       config=self.config)
        elif 'bert' in self.args.model:
            self.model = BertForTokenClassification.from_pretrained(self.args.model,
                                                                    config=self.config)
        else:
            self.model = BertForTokenClassification.from_pretrained(self.args.model,
                                                                    config=self.config)
            #raise NotImplementedError

        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, config, inputs, mode):
        outputs = self.model(input_ids=inputs['source'],
                             attention_mask=inputs['attention_mask'],
                             labels=inputs['labels'])
        return outputs

    def inference(self, inputs):
        outputs = self.model(input_ids=inputs['input_ids'].to(self.args.device),
                             attention_mask=inputs['attention_mask'].to(self.args.device))
        return outputs
