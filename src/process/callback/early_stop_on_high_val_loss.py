import keras

class EarlyStopOnHighValLoss(keras.callbacks.Callback):
    def __init__(self, threshold, patience=3):
        super(EarlyStopOnHighValLoss, self).__init__()
        self.threshold = threshold
        self.patience = patience
        self.wait = 0
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss > self.threshold:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
        else:
            self.wait = 0

    def set_model(self, model):
        return super().set_model(model)