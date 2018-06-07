python3 RNN_model.py RNN_LSTM train --save_dir . --train_path $1 --semi_path $2
python3 RNN_model.py RNN_LSTM semi --save_dir . --load_model RNN_LSTM --train_path $1 --semi_path $2
