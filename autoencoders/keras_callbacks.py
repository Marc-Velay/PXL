import keras
from sklearn.metrics import roc_auc_score, confusion_matrix

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        #y_pred = self.model.predict(self.validation_data[0])
        #self.aucs.append(roc_auc_score(self.validation_data[1].argmax(axis=1), y_pred.argmax(axis=1)))

        #matrix = confusion_matrix(self.validation_data[1].argmax(axis=1), y_pred.argmax(axis=1))
        #print(matrix)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
