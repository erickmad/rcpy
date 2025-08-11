import numpy as np

def forecast_rcpy(warmup_data, model, forecast_length):

    dim = 1
    # Warm up the model
    warmup_y = model.run(warmup_data, reset=True)

    Y_pred = np.empty((forecast_length, dim))
    x = warmup_y[-1].reshape(1, -1)

    for i in range(forecast_length):
        x = model(x)
        Y_pred[i] = x

    return Y_pred

