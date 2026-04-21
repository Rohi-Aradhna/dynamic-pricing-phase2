import numpy as np
from dataset_env import DatasetEnv
from bandits_dataset import run_bandit
from linucb import LinUCB
from dqn_dataset import DQNAgent
from marl_agent import SimpleMARL

# -----------------------------
# LOAD DATASET (FIXED)
# -----------------------------
DATA_PATH = "../data/RL_Dynamic_Pricing_Unified_Dataset.xlsx"

# -----------------------------
# 1. BASIC BANDIT (ε-Greedy)
# -----------------------------
env = DatasetEnv(DATA_PATH)

rewards, arm_counts = run_bandit(env)

print("\n=== BASIC BANDIT ===")
print("Total Reward:", np.sum(rewards))
print("Arm Allocation (%):", 100 * arm_counts / np.sum(arm_counts))


# -----------------------------
# 2. CONTEXTUAL BANDIT (LinUCB)
# -----------------------------
env = DatasetEnv(DATA_PATH)

state = env.reset()

n_arms = len(env.prices)
n_features = len(state)

agent = LinUCB(n_arms, n_features)

rewards_lin = []
arm_counts_lin = np.zeros(n_arms)

for t in range(10000):
    action = agent.select_arm(state)

    next_state, reward, done = env.step(action)

    agent.update(action, state, reward)

    rewards_lin.append(reward)
    arm_counts_lin[action] += 1

    if done:
        break

    state = next_state

print("\n=== CONTEXTUAL BANDIT (LinUCB) ===")
print("Total Reward:", np.sum(rewards_lin))
print("Arm Allocation (%):", 100 * arm_counts_lin / np.sum(arm_counts_lin))


# -----------------------------
# 3. DQN (STATE + ACTION LEARNING)

# -----------------------------
# -----------------------
# 3. Run DQN (PRODUCT-WISE)
# -----------------------

import pandas as pd

df = pd.read_excel(DATA_PATH, sheet_name=3)

print("\n=== DQN (PRODUCT-WISE PRICING) ===")

for product in df["product_category"].unique():

    df_product = df[df["product_category"] == product].reset_index(drop=True)

    env = DatasetEnv(data=df_product)   # ⚠️ IMPORTANT: your DatasetEnv must support this

    state = env.reset()

    states = []
    rewards = []

    for t in range(5000):
        action = np.random.randint(len(env.prices))

        next_state, reward, done = env.step(action)

        states.append(np.append(state, action))
        rewards.append(reward)

        if done:
            break

        state = next_state

    states = np.array(states)
    rewards = np.array(rewards)

    agent = DQNAgent(states.shape[1], len(env.prices))
    agent.train(states, rewards)

    avg_state = np.mean(states, axis=0)

    predictions = []
    for a in range(len(env.prices)):
        test_state = avg_state.copy()
        test_state[-1] = a
        predictions.append(agent.predict(test_state))

    best_price = env.prices[np.argmax(predictions)]

    
    print(f"Best Price for {product}: {best_price}")


# -----------------------------
# 4. MARL (MULTI-AGENT)
# -----------------------------
env = DatasetEnv(DATA_PATH)

marl = SimpleMARL(n_agents=2, n_arms=len(env.prices))

state = env.reset()

rewards_marl = []
arm_counts_marl = np.zeros(len(env.prices))

for t in range(10000):
    actions = marl.select_actions()

    # simulate both agents
    total_reward = 0

    for action in actions:
        _, reward, _ = env.step(action)
        total_reward += reward
        arm_counts_marl[action] += 1

    marl.update(actions, total_reward)

    rewards_marl.append(total_reward)

    if env.current >= env.n:
        break

print("\n=== MARL ===")
print("Total Reward:", np.sum(rewards_marl))
print("Arm Allocation (%):", 100 * arm_counts_marl / np.sum(arm_counts_marl))

import matplotlib.pyplot as plt
import numpy as np

# ===============================
# INPUT YOUR RESULTS HERE
# (use your printed values)
# ===============================

basic_reward = 3637992.15
linucb_reward = 2758039.59
marl_reward = 4380358.07

basic_alloc = np.array([1.32,1.01,1.69,1.13,89.33,1.03,1.16,1.17])
linucb_alloc = np.array([0.36,99.64,0,0,0,0,0,0])
marl_alloc = np.array([2.14,1.82,27.04,56.68,2.12,5.72,0.93,1.59])

# DQN outputs
dqn_prices = {
    "Electronics": 1055.56,
    "Apparel": 1055.56,
    "Books": 1055.56,
    "Sports": 866.67,
    "Home": 1055.56
}

# ===============================
# 1. TOTAL REWARD BAR CHART
# ===============================
plt.figure(figsize=(8,5))
algos = ["Basic Bandit", "LinUCB", "MARL"]
rewards = [basic_reward, linucb_reward, marl_reward]

plt.bar(algos, rewards)
plt.title("Total Reward Comparison")
plt.ylabel("Reward")
plt.show()

# ===============================
# 2. ARM DISTRIBUTION
# ===============================
prices = range(len(basic_alloc))

plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.pie(basic_alloc, labels=prices, autopct='%1.1f%%')
plt.title("Basic Bandit")

plt.subplot(1,3,2)
plt.pie(linucb_alloc, labels=prices, autopct='%1.1f%%')
plt.title("LinUCB")

plt.subplot(1,3,3)
plt.pie(marl_alloc, labels=prices, autopct='%1.1f%%')
plt.title("MARL")

plt.suptitle("Price Selection Distribution")
plt.show()

# ===============================
# 3. DQN PRODUCT-WISE PRICES
# ===============================
plt.figure(figsize=(8,5))
products = list(dqn_prices.keys())
prices = list(dqn_prices.values())

plt.bar(products, prices)
plt.title("DQN Suggested Price per Product")
plt.ylabel("Price")
plt.show()

print("\n=== PERFORMANCE SUMMARY ===")
print(f"Best Model: {'MARL' if marl_reward > linucb_reward else 'LinUCB'}")
print(f"Highest Reward: {max(rewards)}")