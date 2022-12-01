CUDA_VISIBEL_DEVICES=0 python main.py --train True --test False --model klue/bert-base --batch_size 32 --lr 0.00005 --path_to_save ./output/ --ckpt bert_32_5e05.pt
CUDA_VISIBEL_DEVICES=0 python main.py --train False --test True  --model klue/bert-base --path_to_save ./output/ --ckpt bert_32_5e05.pt

CUDA_VISIBEL_DEVICES=0 python main.py --train True --test False --model klue/roberta-base --batch_size 32 --lr 0.00005 --path_to_save ./output/ --ckpt roberta_32_5e05.pt
CUDA_VISIBEL_DEVICES=0 python main.py --train False --test True  --model klue/roberta-base --path_to_save ./output/ --ckpt roberta_32_5e05.pt
