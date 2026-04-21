from dataset_env import DatasetEnv
from bandits_dataset import run_bandit
from linucb import LinUCB
import numpy as np

# Load dataset
env = DatasetEnv("../data/dynamic_pricing_realistic_dataset.xlsx")

# -----------------------
# 1. Run Basic Bandit
# -----------------------
rewards, arm_counts = run_bandit(env)

print("=== BASIC BANDIT ===")
print("Total Reward:", np.sum(rewards))
print("Arm Allocation:", 100 * arm_counts / np.sum(arm_counts))


# -----------------------
# 2. Run LinUCB
# -----------------------
env = DatasetEnv("../data/dynamic_pricing_realistic_dataset.xlsx")

state = env.reset()

n_arms = len(env.prices)
n_features = len(state)

agent = LinUCB(n_arms, n_features)

rewards = []
arm_counts = np.zeros(n_arms)

for t in range(10000):
    action = agent.select_arm(state)

    next_state, reward, done = env.step(action)

    agent.update(action, state, reward)

    rewards.append(reward)
    arm_counts[action] += 1

    if done:
        break

    state = next_state

print("\n=== CONTEXTUAL BANDIT (LinUCB) ===")
print("Total Reward:", np.sum(rewards))
print("Arm Allocation:", 100 * arm_counts / np.sum(arm_counts))

from dqn_dataset import DQNAgent

# -----------------------
# 3. Run DQN
# -----------------------
env = DatasetEnv("../data/dynamic_pricing_realistic_dataset.xlsx")

state = env.reset()

states = []
rewards = []

for t in range(10000):
    action = np.random.randint(len(env.prices))  # random exploration
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

# Evaluate best action
avg_state = np.mean(states, axis=0)

predictions = []

for a in range(len(env.prices)):
    test_state = avg_state.copy()
    test_state[-1] = a
    predictions.append(agent.predict(test_state))

dqn_best = env.prices[np.argmax(predictions)]

print("\n=== DQN ===")
print("Suggested Best Price:", dqn_best)