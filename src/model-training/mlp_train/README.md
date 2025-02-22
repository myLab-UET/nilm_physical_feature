# 
python train_select_comb_ann.py --numepochs 80 --scheduler_end_factor 0.05 --lr_sloth_factor 0.5 --learning_rate 1.1e-3 --weight_decay 1e-5 --dropout_rate 0.2 --window_size 1800 --train_size 0.6 --is_bn false --is_norm true

python eval_single_ann_model.py