import pandas as pd
import numpy as np

class DatasetEnv:
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)
        self.n = len(self.data)
        self.current = 0

        self.prices = sorted(self.data["price"].unique())

    def reset(self):
        self.current = 0
        return self.get_state()

    def get_state(self):
        row = self.data.iloc[self.current]

        # Convert categorical
        customer_type = 1 if row["customer_type"] == "returning" else 0

        state = np.array([
            row["demand_level"],
            row["hour"],
            row["is_weekend"],
            row["inventory_level"],
            row["competitor_price"],
            customer_type
        ])

        return state

    def step(self, action):
        chosen_price = self.prices[action]

        row = self.data.iloc[self.current]

        if row["price"] == chosen_price:
            reward = row["revenue"]
        else:
            reward = 0

        self.current += 1
        done = self.current >= self.n

        next_state = self.get_state() if not done else None

        return next_state, reward, done