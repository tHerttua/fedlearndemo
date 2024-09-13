import numpy as np
import tensorflow as tf
#import tensorflow_federated as tff


class Client:
    def __init__(self, data, model, verbose=0):
        self.data, self.labels = data
        self.model = model
        self.verbose = verbose

    def train(self, epochs=1):
        self.model.fit(self.data, self.labels, epochs=epochs, verbose=self.verbose)
        return self.model.get_weights()

class Server:
    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_weights):
        new_weights = []
        for weights_list in zip(*client_weights):
            new_weights.append(np.mean(weights_list, axis=0))
        self.global_model.set_weights(new_weights)

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.global_model.evaluate(x_test, y_test, verbose=1)
        return loss, accuracy
