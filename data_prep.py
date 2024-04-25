import numpy as np
import pandas as pd
import parameters as params

STEPS = params.STEPS
N_FEATURES = params.N_FEATURES

def interpolate_data(excel_file, rounding = 5):
    raw_data = excel_file
    raw_data = raw_data.replace(0, np.nan)
    raw_data = raw_data.interpolate("linear", limit = 5, axis = 1)
    raw_data[:] = np.tril(raw_data.values)
    raw_data = raw_data.replace(0, np.nan)
    raw_data = raw_data.round(rounding)
    raw_data = raw_data.dropna(axis = 0, how = 'all')
    raw_data = raw_data.dropna(axis = 1, how = 'all')
    return raw_data

def row_to_factors(row):
    target_date = row.name
    row_df = pd.DataFrame(row)
    row_df["N_days_left"] = (pd.to_datetime(target_date) - row_df.index).days
    for weekday in range(6):
        row_df[f"weekday = {weekday}"] = (target_date).date().weekday() == weekday
        row_df[f"weekday = {weekday}"] = row_df[f"weekday = {weekday}"].astype(int)
    row_df.dropna(inplace = True)
    return row_df

def manage_row_in_X(row):
    X = []
    Y = []
    for i in range(len(row) - STEPS):
        X_row = [row.iloc[i + r] for r in range(STEPS)]
        X.append(X_row)
        Y.append([row.iloc[i + STEPS, 0]])
    
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def data_to_inputs(data, train_ratio = 1):
    Big_X = np.array([])
    Big_Y = np.array([])

    for i in data.index:
        row = data.loc[i]
        if len(row) <= STEPS:
            continue
        row_df = row_to_factors(row)
        X, Y = manage_row_in_X(row_df)
        Big_X = np.append(Big_X, X)
        Big_Y = np.append(Big_Y, Y)
    
    Big_X = Big_X.reshape((int(len(Big_X)/STEPS/N_FEATURES), STEPS, N_FEATURES))
    p = np.random.permutation(len(Big_X))
    Big_X, Big_Y = Big_X[p], Big_Y[p]

    train_ratio = round(train_ratio * len(Big_X))

    X_train, y_train = Big_X[ : train_ratio], Big_Y[ : train_ratio]
    X_test, y_test = Big_X[train_ratio : ], Big_Y[train_ratio : ]

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    print(f"Подготавливаю данные,окно {STEPS} наблюдений")
    data = interpolate_data(pd.read_excel("Данные по загрузке.xlsx", index_col = 0))
    data.to_excel("Интерполированные данные.xlsx")
    X_train, y_train, X_test, y_test = data_to_inputs(data)
    np.savez("data_numpy.npz", X_train, y_train, X_test, y_test)
    print("Обработка закончена")
