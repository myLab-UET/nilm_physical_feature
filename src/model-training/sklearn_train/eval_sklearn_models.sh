#!/bin/bash
conda init
conda activate mylab-nilm-env

# Test evaluation for the RAE dataset
# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "rf_['Irms', 'P', 'MeanPF', 'S', 'Q'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "rf_['Irms', 'P', 'MeanPF', 'S'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "rf_['Irms', 'P', 'MeanPF'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "rf_['Irms', 'P'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "rf_['Irms'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "rf_['P'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "xgb_['Irms', 'P', 'MeanPF', 'S', 'Q'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "xgb_['Irms', 'P', 'MeanPF', 'S'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "xgb_['Irms', 'P', 'MeanPF'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "xgb_['Irms', 'P'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "xgb_['P'].joblib"

# python eval_sklearn_model.py --dataset rae \
#                             --data_type test \
#                             --model_file "xgb_['Irms'].joblib"

# Training evaluation for the RAE dataset
python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "rf_['Irms', 'P', 'MeanPF', 'S', 'Q'].joblib"
                            
python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "rf_['Irms', 'P', 'MeanPF', 'S'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "rf_['Irms', 'P', 'MeanPF'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "rf_['Irms', 'P'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "rf_['Irms'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "rf_['P'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "xgb_['Irms', 'P', 'MeanPF', 'S', 'Q'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "xgb_['Irms', 'P', 'MeanPF', 'S'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "xgb_['Irms', 'P', 'MeanPF'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "xgb_['Irms', 'P'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "xgb_['P'].joblib"

python eval_sklearn_model.py --dataset rae \
                            --data_type train \
                            --model_file "xgb_['Irms'].joblib"

# Print a message indicating that the evaluation is complete
echo "Model evaluation complete."