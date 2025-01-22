import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt

class TimeSeriesDataset:
    def __init__(self, series, window_size, label):
        self.series = series
        self.window_size = window_size
        self.label = label
        self.data = tf.data.Dataset.from_tensor_slices((series, [label] * (len(series) - window_size)))

    def window_data(self):
        return self.data.window(self.window_size, shift=1, drop_remainder=True)

def get_RMS_data(data_info_df, select_row, data_type):
    base_path = "/opt/nilm-shared-data/nilm_device_detection/ICTA2024_dataset/time_series_data/window_1800"
    data_paths = [data_type + "_" + data_info_df.iloc[select_row][f"data{i}"].replace(".xlsx", ".csv") for i in range(1, 5)]

    dfs = []
    for path in data_paths:
        file_path = f"{base_path}/{path}"
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
        else:
            df = None
        dfs.append(df)
    return dfs

def creating_time_series_dataset(data_info_df, window_size, batch_size, features, data_type):
    combined_dataset = None
    for select_row in tqdm(range(len(data_info_df)), desc=f"Loading {data_type} data"):
        dfs = get_RMS_data(data_info_df, select_row, data_type)
        for df in dfs:
            if df is None:
                continue
            series = df[features].to_numpy()
            dataset = TimeSeriesDataset(series, window_size, 1)
            if combined_dataset is None:
                combined_dataset = dataset.window_data()
            else:
                combined_dataset = combined_dataset.concatenate(dataset.window_data())
    
    return combined_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load data information
data_info = "/opt/nilm-shared-data/nilm_device_detection/ICTA2024_dataset/data_information/data_information.xlsx"
data_info_df = pd.read_excel(data_info)

# Parameters
window_size = 20
batch_size = 512
features = ["Irms", "MeanPF", "P", "Q", "S"]
print(f"Features: {features}")

print("Loading data...")
trainDataset = creating_time_series_dataset(data_info_df, window_size, batch_size, features, "train")
valDataset = creating_time_series_dataset(data_info_df, window_size, batch_size, features, "val")
print("Data loaded successfully!")

# Create the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(40), input_shape=(window_size, len(features))),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Learning rate scheduler
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20)
)

# Train the model
num_epochs = 10
history = model.fit(
    trainDataset,
    validation_data=valDataset,
    epochs=num_epochs,
    callbacks=[lr_scheduler],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig("RNN_tf_model_training_history.png")