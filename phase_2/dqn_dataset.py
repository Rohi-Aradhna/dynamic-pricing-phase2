import numpy as np
from sklearn.neural_network import MLPRegressor

class DQNAgent:
    def __init__(self, n_features, n_actions):
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200)
        self.n_actions = n_actions
        self.memory_states = []
        self.memory_targets = []

    def train(self, states, rewards):
        # Train simple function approximator
        self.model.fit(states, rewards)

    def predict(self, state):
        return self.model.predict([state])[0]