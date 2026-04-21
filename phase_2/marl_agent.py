import numpy as np

class SimpleMARL:
    def __init__(self, n_agents, n_arms):
        self.n_agents = n_agents
        self.n_arms = n_arms

        # Each agent has its own Q-values
        self.Q = [np.zeros(n_arms) for _ in range(n_agents)]
        self.N = [np.zeros(n_arms) for _ in range(n_agents)]

    def select_actions(self):
        actions = []
        for i in range(self.n_agents):
            if np.random.rand() < 0.1:
                actions.append(np.random.randint(self.n_arms))
            else:
                actions.append(np.argmax(self.Q[i]))
        return actions

    def update(self, actions, reward):
        for i, a in enumerate(actions):
            self.N[i][a] += 1
            self.Q[i][a] += (reward - self.Q[i][a]) / self.N[i][a]