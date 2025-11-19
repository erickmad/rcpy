

def train_model(model, data, forecasting_step=1, washout_training=100):
    model.fit(data["train_data"][:-forecasting_step], data["train_data"][forecasting_step:], warmup=washout_training)
    return model