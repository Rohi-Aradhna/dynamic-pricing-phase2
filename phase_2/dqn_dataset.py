import numpy as np
from sklearn.neural_network import MLPRegressor

class DQNAgent:
    def __init__(self, n_features, n_actions):
        self.model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300)
        self.n_actions = n_actions

    def train(self, states, rewards):
        self.model.fit(states, rewards)

    def predict(self, state):
        return self.model.predict([state])[0]