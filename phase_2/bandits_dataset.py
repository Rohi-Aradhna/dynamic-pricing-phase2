import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q))
    return np.argmax(Q)

def run_bandit(env, steps=10000):
    n_arms = len(env.prices)

    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)

    rewards = []
    arm_counts = np.zeros(n_arms)

    state = env.reset()

    for t in range(steps):
        action = epsilon_greedy(Q)

        next_state, reward, done = env.step(action)

        # Update same as Phase 1
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        rewards.append(reward)
        arm_counts[action] += 1

        if done:
            break

        state = next_state

    return rewards, arm_counts