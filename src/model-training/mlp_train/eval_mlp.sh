#!/bin/bash
conda init
conda activate mylab-nilm-env

# python eval_single_ann_model.py --dataset rae --is_norm True --data_type test --model_name "mlp_['Irms', 'P', 'MeanPF', 'S', 'Q'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type test --model_name "mlp_['Irms', 'P', 'MeanPF', 'S'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type test --model_name "mlp_['Irms', 'P', 'MeanPF'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type test --model_name "mlp_['Irms', 'P'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type test --model_name "mlp_['Irms'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type test --model_name "mlp_['P'].pt"

# Training evaluation for the RAE dataset
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type train --model_name "mlp_['Irms', 'P', 'MeanPF', 'S', 'Q'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type train --model_name "mlp_['Irms', 'P', 'MeanPF', 'S'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type train --model_name "mlp_['Irms', 'P', 'MeanPF'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type train --model_name "mlp_['Irms', 'P'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type train --model_name "mlp_['Irms'].pt"
# python eval_single_ann_model.py --dataset rae --is_norm True --data_type train --model_name "mlp_['P'].pt"

