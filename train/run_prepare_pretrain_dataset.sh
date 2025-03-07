#!/bin/bash

python prepare_pretrain_dataset.py \
    --data_input_dirs "/home/jiang/dataset/LIRAD_input/" \
    --tokenizer_dir "/home/jiang/model/deepseek-coder-1.3b-base/" \
    --data_output_dirs "/home/jiang/dataset/LIRAD/" \
    --max_length 4096 \
    --num_spliced_dataset_bins 1