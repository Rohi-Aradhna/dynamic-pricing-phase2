import pandas as pd
import numpy as np

class DatasetEnv:
    def __init__(self, file_path=None, data=None):

        if data is not None:
            self.data = data
        else:
            self.data = pd.read_excel(file_path, sheet_name=3)

        self.n = len(self.data)
        self.current = 0

        self.prices = sorted(self.data["action_price"].unique())

    def reset(self):
        self.current = 0
        return self.get_state()

    def get_state(self):
        row = self.data.iloc[self.current]

        # Use available features
        state = np.array([
            row["hour"],
            row["hour_sin"],
            row["hour_cos"],
            row["day_of_week"],
            row["product_score"]
        ])

        return state

    def step(self, action):
        chosen_price = self.prices[action]

        row = self.data.iloc[self.current]

        # Simulated reward logic (since real price decision not present)
        demand_factor = row["product_score"] / 5

        # Simple demand logic
        purchase_prob = max(0, 1 - chosen_price / 2000) * demand_factor

        reward = chosen_price if np.random.rand() < purchase_prob else 0

        self.current += 1
        done = self.current >= self.n

        next_state = self.get_state() if not done else None

        return next_state, reward, done