#!/bin/bash
# python train_k_fold_model.py --data "vndale1" --is_norm True --n_folds 4 
# python train_k_fold_model.py --data "iawe" --is_norm True --n_folds 4 --numepochs 30
python train_k_fold_model.py --data "rae" --is_norm True --n_folds 4 --numepochs 60
# python train_k_fold_model.py --data "vndale1" --is_norm True --n_folds 4 --numepochs 60
