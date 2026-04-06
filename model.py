import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


def create_dataset(dataset, look_back=1):
    X, Y = [], []

    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])

    return np.array(X), np.array(Y)


def build_model(look_back):
    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model