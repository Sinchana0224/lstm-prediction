import numpy as np


def predict_future(model, data_scaled, scaler, days=30, look_back=5):
    future_preds = []

    current_input = data_scaled[-look_back:].reshape(1, 1, look_back)

    for _ in range(days):
        pred = model.predict(current_input, verbose=0)
        future_preds.append(pred[0][0])

        new_input = np.append(current_input[0][0][1:], pred[0][0])
        current_input = new_input.reshape(1, 1, look_back)

    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    )

    return future_preds