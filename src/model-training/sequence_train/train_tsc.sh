#!/bin/sh

DATASET=iAWE
data_ratio=1.0
CLF_NAME=resnet

python train_tsc.py --dataset $DATASET \
                    --clf_name "$CLF_NAME" \
                    --data_ratio $data_ratio \
                    --num_workers 4