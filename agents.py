import numpy as np
import tensorflow as tf
#import tensorflow_federated as tff



class Client:
    def __init__(self, data, model, verbose=0, clip_norm=None, privacy_budget=None):
        """
        Initializes the client.
        clip_norm: Maximum norm for weight clipping.
        privacy_budget: Standard deviation of Gaussian noise to add to weights.
        """
        self.data, self.labels = data
        self.model = model
        self.verbose = verbose
        self.clip_norm = clip_norm
        self.privacy_budget = privacy_budget

    def train(self, epochs=1):
        """
        Trains the client's local model and returns the weights, with optional clipping and noise addition.
        """
        # Train the local model
        self.model.fit(self.data, self.labels, epochs=epochs, verbose=self.verbose)

        # Get the trained model's weights
        weights = self.model.get_weights()

        # Optionally apply weight clipping
        if self.clip_norm is not None:
            weights = self.clip_weights(weights)

        # Optionally add noise to the weights
        if self.privacy_budget is not None:
            weights = self.add_noise(weights)

        return weights

    def clip_weights(self, weights):
        """
        Clips the weight updates to have a maximum norm of clip_norm.
        """
        norm = np.linalg.norm([np.linalg.norm(w) for w in weights])
        if norm > self.clip_norm:
            scaling_factor = self.clip_norm / norm
            weights = [w * scaling_factor for w in weights]
        return weights

    def add_noise(self, weights):
        """
        Adds Gaussian noise to the weight updates.
        """
        noisy_weights = [w + np.random.normal(0, self.privacy_budget, w.shape) for w in weights]
        return noisy_weights


class Server:
    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_weights):
        """
        Update the global model weights using the individual clients weights.
        """
        new_weights = []
        for weights_list in zip(*client_weights):
            new_weights.append(np.mean(weights_list, axis=0))
        self.global_model.set_weights(new_weights)

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.global_model.evaluate(x_test, y_test, verbose=1)
        return loss, accuracy
