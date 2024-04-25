import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from data_prep import manage_row_in_X, row_to_factors
from keras.models import load_model
from functions import forecast_day
from datetime import date
from datetime import datetime
import parameters as params
import pandas as pd
import numpy as np
import plotly.graph_objects as go

STEPS = params.STEPS
N_FEATURES = params.N_FEATURES

model_name = 'GRU_1_15 steps.keras'
data = pd.read_excel('Интерполированные данные.xlsx', index_col = 0)
target_date = datetime.combine(date(2024, 4, 23), datetime.min.time())
row = data.loc[target_date]
row.dropna(inplace = True)
X, _ = manage_row_in_X(row_to_factors(row))

start_forecast_from = len(X)
model_best = load_model(f'checkpoints\\{model_name}')
predictions_step = [model_best.predict(X[i].reshape(1, STEPS, N_FEATURES)).item() for i in range(start_forecast_from)]

real_x = row
y_coords = tuple(row.index)

A = []
for i in range(start_forecast_from):
    predictions_forecast = forecast_day(model_name, X[i], X[i][-1][1])
    A.append(predictions_forecast)

fig = go.Figure()
fig.update_layout(
    title = str(target_date) + f' {model_name}',
    xaxis_title="Observation date",
    yaxis_title="Occupancy")
fig.add_trace(go.Scatter(y = real_x, x = y_coords, mode = 'lines + markers', name = "Real data"))
fig.add_trace(go.Scatter(y = predictions_step, x = y_coords[STEPS:], mode = 'lines + markers', name = "1 step prediction"))
for forecast_start in range(len(A)):
    fig.add_trace(
        go.Scatter(
            visible=False,
            mode = "lines+markers",
            name="forecast starting from = " + str(forecast_start),
            x=y_coords[STEPS+forecast_start:],
            y=A[forecast_start]))
fig.data[2].visible = True
slider_steps = []
for i in range(len(A)):
    step = dict(
        method="update",
        args=[{"visible": [True, True] + [False] * len(A)}]
    )
    step["args"][0]["visible"][i+2] = True  # Toggle i'th trace to "visible"
    slider_steps.append(step)

sliders = [dict(
    active = 0,
    steps = slider_steps
)]

fig.update_layout(
    sliders = sliders
)
fig.show()