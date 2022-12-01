import h5py
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer
from keras_preprocessing.sequence import pad_sequences

def create_label(pad_token, label):
    label_dict = {word: i for i, word in enumerate(label)}
    label_dict.update({pad_token: len(label_dict)})

    index_to_ner = {i: j for j, i in label_dict.items()}

    print(f"\nLabel Dictionary\n{label_dict}")
    return label_dict, index_to_ner

def mapping_token_to_label(data, label_dict):
    mapping_t2l = []
    temp_mapping_t2l = [data[0][1:]]

    for idx, token, label in data:
        if int(idx) != 1:
            temp_mapping_t2l.append([token, label_dict[label]])
        if int(idx) == 1:
            if len(temp_mapping_t2l) != 0:
                mapping_t2l.append(temp_mapping_t2l)
                temp_mapping_t2l = [[token, label_dict[label]]]
    mapping_t2l.pop(0)

    return mapping_t2l

def mapping_sentence_to_target(mapping_t2l, label_dict):
    sentences = []
    targets = []

    for tup in mapping_t2l:
        sentence = []
        target = []

        sentence.append("[CLS]")
        target.append(label_dict['-'])

        for token, label_idx in tup:
            sentence.append(token)
            target.append(label_idx)

        sentence.append("[SEP]")
        target.append(label_dict['-'])

        sentences.append(sentence)
        targets.append(target)

    return sentences, targets


def create_dataset(file_name):
    dataset = []
    label = set()

    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            if len(line) == 1 and line[0] == '\n': continue

            src = line[1].replace(r'[^ㄱ-ㅣ가-힣0-9a-zA-Z.]+', "")
            tgt = line[-1].strip()

            label.add(tgt)
            dataset.append([line[0], src, tgt])

    return dataset, sorted(list(label))

def tokenize_and_preserve_labels(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def sort_dataset(args, source, tags, mask, label_dict):
    file_name = f"{args.data_path}/{args.sorted_bin_file_name}"
    train_slice = int(len(source) * args.train_ratio)
    valid_slice = int(len(source) * args.valid_ratio)
    test_slice = int(len(source)-train_slice-valid_slice)

    print(f"> Training: {train_slice}, "
          f"Validation: {valid_slice}, "
          f"Testing: {test_slice}")

    train_component = [source[:train_slice],
                       mask[:train_slice],
                       tags[:train_slice]]

    valid_component = [source[train_slice:train_slice+valid_slice],
                       mask[train_slice:train_slice+valid_slice],
                       tags[train_slice:train_slice+valid_slice]]

    test_component = [source[train_slice + valid_slice:],
                      mask[train_slice + valid_slice:],
                      tags[train_slice + valid_slice:]]

    components = [train_component, valid_component, test_component]
    assert len(source[:train_slice]) + \
           len(source[train_slice:train_slice+valid_slice]) + \
           len(source[train_slice + valid_slice:]) == \
           len(source)

    file_name_list = [f"{file_name}_train.hdf5", f"{file_name}_valid.hdf5", f"{file_name}_test.hdf5"]
    for name, component in zip(file_name_list, components):
        h5file = h5py.File(name, 'w')
        group = h5file.create_group('preprocessed_data_bin_file')

        group.create_dataset('input_ids', data=component[0])
        group.create_dataset('attention_mask', data=component[1])
        group.create_dataset('labels', data=component[2])

        h5file.close()
    print(f"> Complete to make h5py file")

    # Storing 'Label Dictionary'
    file_name = f"{args.data_path}/label_dict.json"
    with open(file_name, 'w') as f:
        json.dump(label_dict, f, indent=4)

    print(f"> Complete to make label dictionary json")

def main(args):
    file_name = f"{args.data_path}/{args.dataset_name}"
    print(f"\n=====Current Dataset Path=====\n======{file_name}======")

    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)
    pad_token = tokenizer.pad_token

    dataset, label = create_dataset(file_name)
    label_dict, index_to_ner = create_label(pad_token, label)
    mapping_t2l = mapping_token_to_label(dataset, label_dict)

    sentences, targets = mapping_sentence_to_target(mapping_t2l, label_dict)

    print(f"\n> Tokenizing sentence and labels...")
    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(tokenizer, sent, labs)
        for sent, labs in zip(tqdm(sentences), targets)]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    assert len(tokenized_texts) == len(labels)
    print(f"> Finish! Total number of dataset: {len(tokenized_texts)}")

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=args.token_max_length, dtype="int", value=tokenizer.convert_tokens_to_ids(pad_token),
                              truncating="post", padding="post")
    # [-100] is ignored when calculate pytorch cross entropy loss
    tags = pad_sequences([lab for lab in labels], maxlen=args.token_max_length, value=label_dict[pad_token], padding='post', \
                         dtype='int', truncating='post')
    attention_masks = np.array([[int(i != tokenizer.convert_tokens_to_ids(pad_token)) for i in ii] for ii in input_ids])
    assert len(input_ids) == len(tags) == len(attention_masks)

    sort_dataset(args, input_ids, tags, attention_masks, label_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="ner_dataset.tsv")
    parser.add_argument("--sorted_bin_file_name", type=str, default="ner_bin")
    parser.add_argument("--model_tokenizer", type=str, default="klue/bert-base")
    parser.add_argument("--token_max_length", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--valid_ratio", type=float, default=0.05)

    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main(args)
