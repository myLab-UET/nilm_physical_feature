#!/bin/bash
conda init
conda activate mylab-nilm-env

python train_select_comb_ann.py --data vndale1 --window_size 1800 --train_size 0.7 --is_norm True