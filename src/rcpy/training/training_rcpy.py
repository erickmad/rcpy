

def train_model(model, train_data, forecasting_step=1, washout_training=100):
    model.fit(train_data[:-forecasting_step], train_data[forecasting_step:], warmup=washout_training)
    return model