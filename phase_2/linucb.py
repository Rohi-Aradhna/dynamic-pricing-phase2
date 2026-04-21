import numpy as np

class LinUCB:
    def __init__(self, n_arms, n_features, alpha=1.0):
        self.n_arms = n_arms
        self.alpha = alpha

        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, state):
        p = []

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]

            p_val = theta @ state + self.alpha * np.sqrt(state @ A_inv @ state)
            p.append(p_val)

        return np.argmax(p)

    def update(self, arm, state, reward):
        self.A[arm] += np.outer(state, state)
        self.b[arm] += reward * state