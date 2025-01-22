run:
python torch_train_sequence_model.py --learning_rate 1e-3 --model LSTM --gpu 0 --workers 12 --window_size 45 --hidden_size 90
python torch_eval_sequence_model.py --file_name "LSTM_['Irms', 'MeanPF', 'P', 'Q', 'S']_best.pt" --window_size 20 --hidden_size 40 --data_type val

python torch_train_sequence_model.py --learning_rate 1e-3 --model GRU --gpu 0 --workers 12 --window_size 15 --hidden_size 30
python torch_eval_sequence_model.py  --file_name "GRU_['Irms', 'MeanPF', 'P', 'Q', 'S']_best.pt" --window_size 20 --hidden_size 40 --data_type val

python torch_train_sequence_model.py --learning_rate 1.2e-3 --model RNN --gpu 0  --workers 12 --window_size 15 --hidden_size 30
python torch_eval_sequence_model.py  --file_name "RNN_['Irms', 'MeanPF', 'P', 'Q', 'S']_best.pt"
