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

data = pd.read_excel('Интерполированные данные.xlsx', index_col = 0)
data = data.iloc[45:, :]
data.dropna(axis=1,how= 'all', inplace=True)
tracing = []
for i in range(len(data)):
    date = data.iloc[i]
    trace = date[i:i + 46]
    tracing.append(trace)
tracing = tracing[:-105]
fig = go.Figure()
for t in range(len(tracing)):
    fig.add_trace(go.Scatter(y = tracing[t], x = [x for x in range(1,46)], mode = 'lines', name= str(tracing[t].name), line=dict(color="blue", width = 1)))
fig.show()