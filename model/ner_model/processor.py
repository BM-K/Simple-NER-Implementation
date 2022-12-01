import logging
from apex import amp
import torch.nn as nn
from tqdm import tqdm
import torch.quantization
import torch.optim as optim

from model.utils import Metric
from accelerate import Accelerator

from model.ner_model.models import ClassificationModel
from data.dataloader import get_loader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.metric = Metric(args)

        self.total_steps = 0
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss': float('inf')}

        self.dev_progress = {'loss': 0, 'iter': 0, 'acc': 0}
        self.model_progress = {'loss': 0, 'iter': 0, 'acc': 0}

    def run(self, inputs, type=None):
        outputs = self.config['model'](self.config, inputs, type)
        acc = self.metric.cal_acc(outputs.logits, inputs['labels'], inputs['attention_mask'],
                                  len(self.config['label_dict']) - 1)

        return outputs.logits, outputs.loss, acc

    def progress(self, loss, acc):
        self.model_progress['loss'] += loss
        self.model_progress['acc'] += acc
        self.model_progress['iter'] += 1

    def progress_validation(self, loss, acc):
        self.dev_progress['loss'] += loss
        self.dev_progress['acc'] += acc
        self.dev_progress['iter'] += 1

    def return_value(self):
        loss = self.model_progress['loss'].data.cpu().numpy() / self.model_progress['iter']
        acc = self.model_progress['acc'].data.cpu().numpy() / self.model_progress['iter']
        return loss, acc

    def get_object(self, model):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(optim,
                                                    num_warmup_steps=self.args.warmup_ratio * train_total,
                                                    num_training_steps=train_total)

        return scheduler, train_total

    def model_setting(self):
        accelerator = Accelerator(fp16=True)

        loader, tokenizer, label_dict = get_loader(self.args, self.metric)
        model = ClassificationModel(self.args, label_dict)

        if self.args.multi_gpu == 'True':
            model = nn.DataParallel(model, output_device=0)
        model.to(self.args.device)

        criterion, optimizer = self.get_object(model)

        if self.args.train == 'True':
            scheduler, total_steps = self.get_scheduler(optimizer, loader['train'])
            self.total_steps = total_steps
        else:
            scheduler = None

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'accelerator': accelerator,
                  'args': self.args,
                  'label_dict': label_dict,
                  'model': model}
        config['model'], config['optimizer'] = accelerator.prepare(model, optimizer)
        self.config = config

        return self.config

    def train(self, epoch):
        self.config['model'].train()

        train_loader = self.config['accelerator'].prepare(self.config['loader']['train'])
        for step, inputs in enumerate(tqdm(train_loader)):
            self.config['optimizer'].zero_grad()

            logits, loss, acc = self.run(inputs, type='train')

            loss = torch.mean(loss)
            acc = torch.mean(acc)

            self.config['accelerator'].backward(loss)

            self.config['optimizer'].step()
            self.config['scheduler'].step()

            self.progress(loss.data, acc)

            if self.model_progress['iter'] % self.args.eval_steps == 0 or self.model_progress[
                'iter'] == self.total_steps:
                self.valid()

                performance = {'tl': self.model_progress['loss'] / self.model_progress['iter'],
                               'vl': self.dev_progress['loss'] / self.dev_progress['iter'],
                               'ea': self.model_progress['acc'] / self.model_progress['iter'],
                               'va': self.dev_progress['acc'] / self.dev_progress['iter'],
                               'ep': epoch,
                               'step': self.model_progress['iter']}

                self.metric.save_model(self.config, performance, self.model_checker)

    def valid(self):
        self.config['model'].eval()
        self.dev_progress = self.dev_progress.fromkeys(self.dev_progress, 0)

        valid_loader = self.config['accelerator'].prepare(self.config['loader']['valid'])
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                inputs = batch
                logits, loss, acc = self.run(inputs, type='valid')
                loss = torch.mean(loss)
                acc = torch.mean(acc)
                self.progress_validation(loss.data, acc)

    def test(self):
        sorted_path = self.config['args'].path_to_save + self.config['args'].ckpt
        self.config['model'].load_state_dict(torch.load(sorted_path))
        self.config['model'].eval()

        self.dev_progress = self.dev_progress.fromkeys(self.dev_progress, 0)
        with torch.no_grad():
            for step, inputs in enumerate(self.config['loader']['test']):
                logits, loss, acc = self.run(inputs, type='test')
                self.progress_validation(loss.data, acc)

        logger.info('### TEST SCORE ###')
        self.metric.print_test_score(logits,
                                     self.config['label_dict'],
                                     self.dev_progress,
                                     self.config['tokenizer'].pad_token,
                                     inputs['labels'], )
    """
    def post_training(self):
        self.config['model'].load_state_dict(torch.load(self.args.path_to_saved_model))
        self.config['model'].eval()
        from tqdm import tqdm

        total_token = []
        total_label = []
        with open(self.args.post_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = lines[:1000000]
            for _, line in enumerate(tqdm(lines)):
                tokenized_line = self.config['tokenizer'](line,
                                                          truncation=True,
                                                          return_tensors="pt",
                                                          max_length=self.args.max_len,
                                                          padding='max_length')

                outputs = self.config['model'].inference(tokenized_line)
                y_predicted = outputs.logits.max(dim=-1)[1]
                f_label = [i for i, _ in self.config['label_dict'].items()]

                y_predicted = y_predicted.view(-1).cpu().numpy()
                y_predicted = [f_label[x] for x in y_predicted]

                y = self.config['tokenizer'].convert_ids_to_tokens(tokenized_line['input_ids'][0])[1:]
                where_sep = y.index('[SEP]')
                y = y[:where_sep]
                y_predicted = y_predicted[1:len(y)]

                new_tokens, new_labels = [], []
                for token, label in zip(y, y_predicted):
                    if (token.startswith('##')):
                        nt = new_tokens[-1]+token[2:]
                        new_tokens.pop()
                        new_tokens.append(nt)
                    else:
                        new_tokens.append(token)
                        new_labels.append(label)

                assert len(new_tokens) == len(new_labels)

                total_token.append(new_tokens)
                total_label.append(new_labels)

        with open('data/post_training_data.tsv', 'w', encoding='utf-8') as w:
            for text, label in zip(tqdm(total_token), total_label):
                index = 1
                for t, l in zip(text, label):
                    tex = f"{index}\t{t}\t{l}\n"
                    w.write(tex)
                    index+=1
                w.write('\n')
    """
