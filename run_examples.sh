#CUDA_VISIBEL_DEVICES=0 python main.py --train True --test False --batch_size 32 --lr 0.00005 --path_to_save ./output/ --ckpt bert_32_5e05.pt
CUDA_VISIBEL_DEVICES=0 python main.py --train False --test True  --path_to_save ./output/ --ckpt bert_32_5e05.pt

CUDA_VISIBEL_DEVICES=0 python main.py --train True --test False --model klue/roberta-base --batch_size 32 --lr 0.00005 --path_to_save ./output/ --ckpt roberta_32_5e05.pt

CUDA_VISIBEL_DEVICES=0 python main.py --train False --test True  --path_to_save ./output/ --ckpt roberta_32_5e05.pt

