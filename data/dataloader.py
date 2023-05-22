import h5py
import torch
import logging
import numpy as np

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer, type_):
        self.type = type_
        self.args = args
        self.metric = metric
        self.tokenizer = tokenizer
        self.file_path = file_path

        self.label = []
        self.input_ids = []
        self.attention_mask = []

        self.init_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.mask_token = self.tokenizer.sep_token

        self.init_token_idx = self.tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        self.mask_token_idx = self.tokenizer.convert_tokens_to_ids(self.mask_token)

        print(f"cls: {self.init_token_idx}")
        print(f"pad: {self.pad_token_idx}")
        print(f"mask: {self.mask_token_idx}")
        
    def load_data(self, type):
        self.get_data_from_hdf5_file()

        assert len(self.input_ids) ==\
               len(self.attention_mask) ==\
               len(self.label)

    def get_data_from_hdf5_file(self,):
        hdf5_file = h5py.File(self.file_path, 'r')
        self.input_ids = np.array(hdf5_file['preprocessed_data_bin_file'].get('input_ids'))
        self.attention_mask = np.array(hdf5_file['preprocessed_data_bin_file'].get('attention_mask'))
        self.label = np.array(hdf5_file['preprocessed_data_bin_file'].get('labels'))
        hdf5_file.close()

    def __getitem__(self, index):
        inputs = {'source': torch.LongTensor(self.input_ids[index]).to(self.args.device),
                  'attention_mask': torch.LongTensor(self.attention_mask[index]).to(self.args.device),
                  'labels': torch.LongTensor(self.label[index]).to(self.args.device)}

        return inputs

    def __len__(self):
        return len(self.input_ids)

def get_label_dict(path):
    import json
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data

# Get train, valid, test data loader and BERT tokenizer
def get_loader(args, metric):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data
    path_to_test_data = args.path_to_data + '/' + args.test_data
    path_to_label_dict = args.path_to_data + '/' + args.label_dict

    label_dict = get_label_dict(path_to_label_dict)

    if args.train == 'True' and args.test == 'False':

        train_iter = ModelDataLoader(path_to_train_data,
                                     args,
                                     metric,
                                     tokenizer,
                                     type_='train')

        valid_iter = ModelDataLoader(path_to_valid_data,
                                     args,
                                     metric,
                                     tokenizer,
                                     type_='valid')

        train_iter.load_data('train')
        valid_iter.load_data('valid')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True),
                  'valid': DataLoader(dataset=valid_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True)}

    elif args.train == 'False' and args.test == 'True':
        test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer, type_='test')
        test_iter.load_data('test')

        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=8192,
                                     shuffle=True)}

    else:
        loader = None

    return loader, tokenizer, label_dict

if __name__ == '__main__':
    get_loader('test')
