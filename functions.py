import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from data_prep import *
from pathlib import Path
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.models import load_model
from keras.layers import LSTM, Dense, Input, SimpleRNN, GRU
from keras.models import Sequential
from keras.losses import MeanSquaredError
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from datetime import date
from datetime import datetime
from data_prep import interpolate_data, row_to_factors
import pandas as pd
import numpy as np
import parameters as params
import calendar
import matplotlib.pyplot as plt

STEPS = params.STEPS
N_FEATURES = params.N_FEATURES

class My_Callback(Callback):
    def __init__(self, model_name, folder_path='checkpoints'):
        super(My_Callback, self).__init__()
        self.folder_path = folder_path
        self.model_name = model_name
        self.best_val_loss = float('inf')
        self.best_model_path = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        val_mape = logs.get('val_mean_absolute_percentage_error')
        print(f'Epoch {epoch + 1} - Validation Loss: {val_loss:.6f} - Validation MAPE: {val_mape:.3f}')

        # Save model if validation loss has improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self.best_model_path:
                self.best_model_path.unlink(missing_ok=True)  # Delete previous best model
            folder = Path(self.folder_path)
            count = sum(1 for x in folder.iterdir())
            self.best_model_path = folder / f'{self.model_name}_{count}_{STEPS} steps.keras'
            self.model.save(self.best_model_path)

def forecast_day(model_name, X, horizon = 1):
    model = load_model(f'checkpoints\\{model_name}')
    X = X.reshape(1, STEPS, N_FEATURES)
    forecast_predictions = []

    for _ in range(int(horizon)):
        prediction = model.predict(X).item()
        new_row = np.array([prediction, X[0][-1][1] - 1] + [X[0][-1][i] for i in range(2, 8)])
        forecast_predictions.append(prediction)
        X = np.append(X[0][1:], new_row).reshape(1, STEPS, N_FEATURES)

    return forecast_predictions

def create_model(neuron_type = "LSTM", n_cells = 3, learn_rate = 0.001):
    initial = RandomNormal(mean = 0.0, stddev = 1)
    model1 = Sequential(name = neuron_type)
    model1.add(Input(shape = (STEPS, N_FEATURES)))
    match neuron_type:
        case "LSTM":
            model1.add(LSTM(n_cells, activation = "relu", kernel_initializer = initial))
        case "RNN":
            model1.add(SimpleRNN(n_cells, activation = "relu", kernel_initializer = initial))
        case "GRU":
            model1.add(GRU(n_cells, activation = "relu", kernel_initializer = initial))
        case _:
            print("Не найден тип")
            return
    #model1.add(Dense(2, activation = "relu", kernel_initializer = initial))
    model1.add(Dense(1, activation = "hard_sigmoid", kernel_initializer = initial))
    model1.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = learn_rate), metrics= ["mean_absolute_percentage_error"])
    model1.summary()
    return model1

def forecast_month(month_to_forecast, model_name):
    predictions = []
    data = interpolate_data(pd.read_excel("Данные по загрузке.xlsx", index_col=0))
    _, number_of_days = calendar.monthrange(2024, month_to_forecast)
    dates = [datetime.combine(date(2024, month_to_forecast, i), datetime.min.time()) for i in range(1, number_of_days + 1)]
    rows = [data.loc[i] for i in dates]
    rows_df = [row_to_factors(rows[i]) for i in range(len(rows))]
    rows_X = [np.array([rows_df[i].iloc[-STEPS + r] for r in range(STEPS)]) for i in range(len(rows_df))]
    for i in range(len(rows_X)):
        prediction_forecast = forecast_day(model_name, rows_X[i], rows_X[i][-1][1])
        predictions.append(prediction_forecast)
    os.makedirs("forecasts", exist_ok=True) # Create "forecasts" directory if it doesn't exist
    file_path = os.path.join("forecasts", f"predictions_{month_to_forecast}_{model_name}.xlsx")
    df = pd.DataFrame(predictions)
    df.interpolate(method = 'pad', axis = 1, inplace=True)
    averages = df.mean().to_frame().T
    averages.index = ['Average']
    df = pd.concat([df, averages])
    df.to_excel(file_path)
    print(f"Predictions saved to: {file_path}")

def visualize_training_history(history):
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot metrics
    if 'mean_absolute_percentage_error' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mean_absolute_percentage_error'], label='Training MAPE')
        plt.plot(history.history['val_mean_absolute_percentage_error'], label='Validation MAPE')
        plt.title('Training and Validation MAPE')
        plt.xlabel('Epoch')
        plt.ylabel('MAPE (%)')
        plt.legend()

    plt.tight_layout()
    plt.show()