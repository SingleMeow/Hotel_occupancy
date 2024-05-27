import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from meteostat import Point, Daily
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from keras.layers import LSTM, Dense, Input, SimpleRNN, GRU
from keras.models import Sequential
from keras.losses import MeanSquaredError
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from other_functions import val_loss_callback
from sklearn.preprocessing import StandardScaler


STEPS = 30
N_FEATURES = 2

# Set time period
start = datetime(2018, 9, 1)
end = datetime(2024, 4, 25)

# Create Point for Hvoya
location = Point(54.793045, 83.110065)

# Get daily temperature data for 2023
data = Daily(location, start, end)
data = pd.DataFrame(data.fetch())
data.drop(columns=['tmin', 'tavg', 'snow', 'wdir', 'wpgt', 'pres', 'tsun', 'prcp'], inplace = True)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled)

X = np.array([])
Y = np.array([])

for i in range(len(data_scaled) - STEPS):
    X_row = [data_scaled.iloc[i + r] for r in range(STEPS)]
    X = np.append(X, X_row)
    Y = np.append(Y, [data_scaled.iloc[i + STEPS, 0]])
X = X.reshape(int(len(X)/STEPS/N_FEATURES), STEPS, N_FEATURES)

X_train, y_train = X[0:-365], Y[0:-365]
X_test, y_test = X[-365:], Y[-365:]

n_cells = 3
neuron_type = 'LSTM'
model1 = Sequential(name = 'meteo')
initial = RandomNormal(mean = 0.0, stddev = 0.5)
model1.add(Input(shape = (STEPS, N_FEATURES)))
match neuron_type:
    case "LSTM":
        model1.add(LSTM(n_cells, activation = "sigmoid", kernel_initializer = initial))
    case "RNN":
        model1.add(SimpleRNN(n_cells, activation = "relu", kernel_initializer = initial))
    case "GRU":
        model1.add(GRU(n_cells, activation = "relu", kernel_initializer = initial))
    case _:
        print("Не найден тип")
# model1.add(Dense(6, activation = "relu", kernel_initializer = initial))
model1.add(Dense(1, activation = "linear", kernel_initializer = initial))
model1.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = 0.01))
model1.summary()
model1.fit(
    x = X_train,
    y = y_train, 
    verbose = 1,
    validation_freq = 3,
    validation_split = 0.1,
    shuffle = True, 
    epochs = 340, 
    batch_size = 300,
    callbacks = [val_loss_callback(model1.name)])

model_meteo = load_model(f'checkpoints\\{model1.name}')

forecasts = model_meteo.predict(X_test)
forecasts = np.array([[forecasts[i].item()] + [0] for i in range(len(forecasts))])
forecasts_scaled = pd.DataFrame(scaler.inverse_transform(forecasts), columns= ['temp', 'wind'])

plot = go.Figure()
plot.add_trace(go.Scatter(y = data['tmax'], x = data.index, mode = 'lines', name = "Real data"))
plot.add_trace(go.Scatter(y = forecasts_scaled['temp'], x = data.index[-365:], mode = 'lines', name = "Lstm"))
plot.show()