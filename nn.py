import random

import tensorflow as tf
import numpy as np


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.model = None
        self.create_model(tf.keras.Sequential())

    def create_model(self, model):
        self.model = model
        self.model.add(tf.keras.layers.Dense(self.hidden_nodes, input_shape=(self.input_nodes,), activation="sigmoid"))
        self.model.add(tf.keras.layers.Dense(self.output_nodes, activation="softmax"))

    def predict(self, input_features):
        xs = np.array(input_features).reshape(1, len(input_features))
        ys = self.model.predict(xs)
        return ys

    def mutate(self, rate):  # mutating NN weights, each weight has probability of rate to mutate
        whts = self.model.weights
        # mutated_weights = []
        ff_length = len(whts[0].numpy())  # doesn't matter what index, for all it's the same (since it's a MATRIX)
        for i in range(len(whts)):
            if i % 2 == 0:  # we only want to change weights, not the biases
                for j in range(len(whts[i].numpy())):
                    wi = whts[i][j].numpy()
                    for k in range(len(wi)):
                        if random.uniform(0, 1) <= rate:
                            wi[k] = wi[k] + random.gauss(0, 1)
                    self.model.weights[i][j].assign(wi)

    def copy(self):
        network = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        network.create_model(tf.keras.Sequential())
        network.model.set_weights(self.model.get_weights())
        return network
