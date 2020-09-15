import random

import tensorflow as tf
import numpy as np


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.model = None

    def create_model(self, model):
        self.model = model
        self.model.add(tf.keras.layers.Dense(self.hidden_nodes, input_shape=(self.input_nodes,), activation="sigmoid"))
        self.model.add(tf.keras.layers.Dense(self.output_nodes, activation="softmax"))

    def predict(self, input_features):
        xs = np.array(input_features).reshape(1, len(input_features))

        ys = self.model.predict(xs)
        return ys

    def mutate(self, rate):  # mutating NN weights with probability of rate
        weights = self.model.get_weights()
        # mutated_weights = []
        for i, weight_array in enumerate(weights):
            for j in range(len(weight_array)):
                if random.uniform(0, 1) < rate:
                    wht = self.model.weights[i].numpy()
                    wht[j] = wht[j] + random.gauss(0, 1)
                    self.model.weights[i].assign(wht)


# m = NeuralNetwork(4, 4, 2)
# sl = tf.keras.Sequential()
# m.create_model(sl)
# print(sl.weights[0][0])
# print("**********")
# print(sl.get_weights())
# print("*****************")
# m.mutate(0.2)
# print(sl.get_weights())
