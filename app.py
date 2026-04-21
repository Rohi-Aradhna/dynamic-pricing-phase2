import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phase_2.dataset_env import DatasetEnv
from phase_2.bandits_dataset import run_bandit
from phase_2.linucb import LinUCB
from phase_2.dqn_dataset import DQNAgent
from phase_2.marl_agent import SimpleMARL

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Dynamic Pricing AI", layout="wide")

st.title("💰 Dynamic Pricing Intelligence System")
st.markdown("AI-based pricing using Bandits, DQN & MARL")

# ---------------------------
# LOAD DATASET (FIXED)
# ---------------------------
DATA_PATH = "data/RL_Dynamic_Pricing_Unified_Dataset.xlsx"

df = pd.read_excel(DATA_PATH, sheet_name=3)

st.success("Dataset Loaded Successfully ✅")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ---------------------------
# 1. BASIC BANDIT
# ---------------------------
st.header("📊 Basic Bandit (ε-Greedy)")

env = DatasetEnv(data=df)

rewards, arm_counts = run_bandit(env)

total_reward_bandit = np.sum(rewards)

st.metric("Total Reward", f"{total_reward_bandit:,.0f}")

fig1, ax1 = plt.subplots()
ax1.bar(range(len(arm_counts)), arm_counts)
ax1.set_title("Bandit Arm Selection")
ax1.set_xlabel("Price Arm")
ax1.set_ylabel("Selections")

st.pyplot(fig1)

# ---------------------------
# 2. LINUCB
# ---------------------------
st.header("🧠 Contextual Bandit (LinUCB)")

env = DatasetEnv(data=df)
state = env.reset()

n_arms = len(env.prices)
n_features = len(state)

agent = LinUCB(n_arms, n_features)

rewards = []
arm_counts_linucb = np.zeros(n_arms)

for _ in range(5000):
    action = agent.select_arm(state)
    next_state, reward, done = env.step(action)

    agent.update(action, state, reward)

    rewards.append(reward)
    arm_counts_linucb[action] += 1

    if done:
        break

    state = next_state

total_reward_linucb = np.sum(rewards)

st.metric("Total Reward", f"{total_reward_linucb:,.0f}")

fig2, ax2 = plt.subplots()
ax2.bar(range(len(arm_counts_linucb)), arm_counts_linucb)
ax2.set_title("LinUCB Arm Selection")
ax2.set_xlabel("Price Arm")

st.pyplot(fig2)

# ---------------------------
# 3. DQN (PRODUCT-WISE)
# ---------------------------
st.header("🤖 DQN (Product-wise Best Price)")

product_results = {}

for product in df["product_category"].unique():

    df_product = df[df["product_category"] == product]

    env = DatasetEnv(data=df_product)
    state = env.reset()

    states = []
    rewards = []

    for _ in range(3000):
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

    product_results[product] = best_price

# Display nicely
for product, price in product_results.items():
    st.write(f"**{product} → ₹{price}**")

# ---------------------------
# 4. MARL
# ---------------------------
st.header("👥 Multi-Agent RL (MARL)")

env = DatasetEnv(data=df)
state = env.reset()

marl = SimpleMARL(n_agents=2, n_actions=len(env.prices))

rewards = []
arm_counts_marl = np.zeros(len(env.prices))

for _ in range(5000):
    actions = marl.select_actions()

    action = actions[0]  # pick agent 1
    next_state, reward, done = env.step(action)

    marl.update(actions, reward)

    rewards.append(reward)
    arm_counts_marl[action] += 1

    if done:
        break

total_reward_marl = np.sum(rewards)

st.metric("Total Reward", f"{total_reward_marl:,.0f}")

fig3, ax3 = plt.subplots()
ax3.bar(range(len(arm_counts_marl)), arm_counts_marl)
ax3.set_title("MARL Price Selection")

st.pyplot(fig3)

# ---------------------------
# FINAL COMPARISON
# ---------------------------
st.header("🏆 Algorithm Comparison")

comparison = {
    "Bandit": total_reward_bandit,
    "LinUCB": total_reward_linucb,
    "MARL": total_reward_marl
}

best_algo = max(comparison, key=comparison.get)

st.success(f"Best Performing Algorithm: {best_algo} 🚀")

fig4, ax4 = plt.subplots()
ax4.bar(comparison.keys(), comparison.values())
ax4.set_title("Total Reward Comparison")

st.pyplot(fig4)