import os
import torch
import logging
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

logger = logging.getLogger(__name__)
writer = SummaryWriter()


class Metric():

    def __init__(self, args):
        self.args = args

    def get_lr(self, optimizer):
        return optimizer.state_dict()['param_groups'][0]['lr']

    def count_parameters(self, model):
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def cal_acc(self, yhat, y, mask, num_labels):
        with torch.no_grad():
            yhat = yhat.max(dim=-1)[1]  # [0]: max value, [1]: index of max value

        return ((yhat == y).float() * mask).sum() / (y != num_labels).sum()

    def cal_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def draw_graph(self, cp):
        writer.add_scalars('loss_graph', {'train': cp['tl'], 'valid': cp['vl']}, cp['ep'])
        writer.add_scalars('acc_graph', {'train': cp['tma'], 'valid': cp['vma']}, cp['ep'])

    def print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def move2device(self, sample, device):
        if len(sample) == 0:
            return {}

        def _move_to_device(maybe_tensor, device):
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor.to(device)
            elif isinstance(maybe_tensor, dict):
                return {
                    key: _move_to_device(value, device)
                    for key, value in maybe_tensor.items()
                }
            elif isinstance(maybe_tensor, list):
                return [_move_to_device(x, device) for x in maybe_tensor]
            elif isinstance(maybe_tensor, tuple):
                return [_move_to_device(x, device) for x in maybe_tensor]
            else:
                return maybe_tensor

        return _move_to_device(sample, device)

    def print_test_score(self, logits, label_dict, score, pad_token, labels):
        y_predicted = logits.max(dim=-1)[1]
        f_label = [i for i, _ in label_dict.items()]

        y_predicted = y_predicted.view(-1).cpu().numpy()
        y = labels.view(-1).cpu().numpy()

        y_predicted = [f_label[x] for x in y_predicted]
        y = [f_label[x] for x in y]

        f_label.remove(pad_token)
        print(classification_report(y, y_predicted, labels=f_label))
        print(f'\t## TEST SCORE | LOSS: {score["loss"]:.4f} | ACC: {score["acc"]:.4f} ##\n')

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        pco['early_stop_patient'] += 1
        sorted_path = config['args'].path_to_save + config['args'].ckpt
        if cp['vl'] < pco['best_valid_loss']:
            pco['early_stop_patient'] = 0
            pco['best_valid_loss'] = cp['vl']

            unwrapped_model = config['accelerator'].unwrap_model(config['model'])
            config['accelerator'].save(unwrapped_model.state_dict(), sorted_path)

            print(f'\n\t## SAVE {sorted_path} |'
                  f' valid loss: {cp["vl"]:.4f} |'
                  f' valid acc {cp["va"]:.4f} | '
                  f' epochs: {cp["ep"]} |'
                  f' steps: {cp["step"]} ##\n')

        if pco['early_stop_patient'] == self.args.patient:
            raise Exception('\n=====Early Stop=====\n')