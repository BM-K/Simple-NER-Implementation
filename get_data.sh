wget https://github.com/naver/nlp-challenge/raw/master/missions/ner/data/train/train_data

mv train_data data/ner_dataset.tsv

python data/preprocessing.py --data_path ./data --dataset_name ner_dataset.tsv --token_max_length 128 --train_ratio 0.9 --valid_ratio 0.05
