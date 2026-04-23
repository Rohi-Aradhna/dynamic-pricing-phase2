import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_env import DatasetEnv
from bandits_dataset import run_bandit
from linucb import LinUCB
from dqn_dataset import DQNAgent
from marl_agent import CategoryMARL, SELLER_CATEGORIES, UNIT_COSTS

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/RL_Dynamic_Pricing_Unified_Dataset.xlsx"
N_STEPS     = 10000   # steps for bandit / linucb / old marl
N_EPISODES  = 400     # episodes for new CategoryMARL
SELLER_COLORS = {
    "Seller A": "#5b9cf6",
    "Seller B": "#e74c3c",
    "Seller C": "#2ecc71",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASIC BANDIT (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
env = DatasetEnv(DATA_PATH)
rewards_bandit, arm_counts_bandit = run_bandit(env)
print("\n=== BASIC BANDIT ===")
print("Total Reward:", np.sum(rewards_bandit))
print("Arm Allocation (%):", 100 * arm_counts_bandit / np.sum(arm_counts_bandit))

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONTEXTUAL BANDIT — LinUCB (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
env = DatasetEnv(DATA_PATH)
state = env.reset()
n_arms     = len(env.prices)
n_features = len(state)
agent_lin  = LinUCB(n_arms, n_features)
rewards_lin     = []
arm_counts_lin  = np.zeros(n_arms)

for t in range(N_STEPS):
    action = agent_lin.select_arm(state)
    next_state, reward, done = env.step(action)
    agent_lin.update(action, state, reward)
    rewards_lin.append(reward)
    arm_counts_lin[action] += 1
    if done:
        break
    state = next_state

print("\n=== CONTEXTUAL BANDIT (LinUCB) ===")
print("Total Reward:", np.sum(rewards_lin))
print("Arm Allocation (%):", 100 * arm_counts_lin / np.sum(arm_counts_lin))

# ─────────────────────────────────────────────────────────────────────────────
# 3. DQN — CATEGORY-AWARE (upgraded)
#    Each product category gets its own DQN trained on its own rows.
#    The state now includes category context (not just generic features).
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== DQN (CATEGORY-AWARE PRICING) ===")

df_full    = pd.read_excel(DATA_PATH, sheet_name=3)
categories = df_full["product_category"].unique()

dqn_prices   = {}   # best price per category
dqn_margins  = {}   # profit margin per category

for product in categories:
    df_product = df_full[df_full["product_category"] == product].reset_index(drop=True)
    env_cat    = DatasetEnv(data=df_product)
    state      = env_cat.reset()

    # Category one-hot encoding — adds context DQN can use
    cat_list  = list(categories)
    cat_index = cat_list.index(product)
    cat_onehot = np.zeros(len(cat_list))
    cat_onehot[cat_index] = 1.0

    states  = []
    rewards = []

    for t in range(5000):
        action                    = np.random.randint(len(env_cat.prices))
        next_state, reward, done  = env_cat.step(action)

        # Append category one-hot to state so DQN learns category differences
        full_state = np.append(state, cat_onehot)
        full_state = np.append(full_state, action)
        states.append(full_state)
        rewards.append(reward)

        if done:
            break
        state = next_state

    states  = np.array(states)
    rewards = np.array(rewards)

    agent_dqn = DQNAgent(states.shape[1], len(env_cat.prices))
    agent_dqn.train(states, rewards)

    # Ask DQN: for the average state of this category, which price is best?
    avg_state   = np.mean(states, axis=0)
    predictions = []
    for a in range(len(env_cat.prices)):
        test_state     = avg_state.copy()
        test_state[-1] = a                   # last dim = action index
        predictions.append(agent_dqn.predict(test_state))

    best_arm   = np.argmax(predictions)
    best_price = env_cat.prices[best_arm]
    unit_cost  = UNIT_COSTS.get(product, 280)
    margin     = ((best_price - unit_cost) / best_price * 100) if best_price > 0 else 0

    dqn_prices[product]  = best_price
    dqn_margins[product] = round(margin, 1)
    print(f"  {product:15s} → Best Price: ₹{best_price:.0f}  |  "
          f"Unit Cost: ₹{unit_cost}  |  Margin: {margin:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MARL — UPGRADED: CategoryMARL (3 agents, state-aware IQL)
#
#    WHAT CHANGED vs old SimpleMARL:
#      Old: 2 agents, shared reward, no state, no category awareness
#      New: 3 agents (Seller A/B/C), each with own state and own reward,
#           category-specific unit costs, demand split via softmax,
#           epsilon-greedy with decay
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== MARL (3-Agent Category-Aware IQL) ===")
print("Seller A → Electronics + Home")
print("Seller B → Apparel + Sports")
print("Seller C → Books")

env_marl  = DatasetEnv(DATA_PATH)
prices    = env_marl.prices

marl = CategoryMARL(
    n_arms      = len(prices),
    prices      = prices,
    n_state_bins= 5,
    lr          = 0.1,
    gamma       = 0.95,
)

# Episode-level tracking for plots
ep_total_rewards = {s: [] for s in marl.sellers}
ep_avg_prices    = {s: [] for s in marl.sellers}
ep_avg_shares    = {s: [] for s in marl.sellers}
rewards_marl_all = []

for ep in range(N_EPISODES):
    marl.reset_episode()
    ep_step_rewards = {s: [] for s in marl.sellers}
    ep_step_prices  = {s: [] for s in marl.sellers}
    ep_step_shares  = {s: [] for s in marl.sellers}

    for t in range(24):   # 24-step episode = one selling day
        actions, states  = marl.select_actions()
        rewards_step, next_states, shares, prices_step, _ = marl.step(actions, states)
        done = (t == 23)
        marl.update_agents(states, actions, rewards_step, next_states, done)

        for s in marl.sellers:
            ep_step_rewards[s].append(rewards_step[s])
            ep_step_prices[s].append(prices_step[s])
            ep_step_shares[s].append(shares[s])

    for s in marl.sellers:
        ep_total_rewards[s].append(sum(ep_step_rewards[s]))
        ep_avg_prices[s].append(np.mean(ep_step_prices[s]))
        ep_avg_shares[s].append(np.mean(ep_step_shares[s]))

    rewards_marl_all.append(sum(sum(ep_step_rewards[s]) for s in marl.sellers))

    if (ep + 1) % 100 == 0:
        print(f"  Episode {ep+1}/{N_EPISODES} | " +
              " | ".join([f"{s}: ₹{np.mean(ep_total_rewards[s][-50:]):>8,.0f}"
                          for s in marl.sellers]))

total_marl_reward = sum(sum(ep_total_rewards[s]) for s in marl.sellers)
print(f"\nTotal MARL Reward (all agents, all episodes): ₹{total_marl_reward:,.0f}")
print("Arm Allocation per agent (last 50 eps actions):")
for s in marl.sellers:
    arm_hist = marl.agents[s].actions_history[-500:]
    counts   = np.bincount(arm_hist, minlength=len(prices))
    alloc    = 100 * counts / counts.sum() if counts.sum() > 0 else counts
    print(f"  {s}: {np.round(alloc, 1)}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def smooth(arr, w=20):
    return pd.Series(arr).rolling(w, min_periods=1).mean().values

# ── Plot 1: Total Reward Comparison ──────────────────────────────────────────
plt.figure(figsize=(8, 5))
algos   = ["Basic Bandit", "LinUCB", "MARL (IQL)"]
totals  = [
    float(np.sum(rewards_bandit)),
    float(np.sum(rewards_lin)),
    float(total_marl_reward),
]
bars = plt.bar(algos, totals)
plt.bar_label(bars, labels=[f"₹{v:,.0f}" for v in totals], padding=4)
plt.title("Total Reward Comparison")
plt.ylabel("Reward (₹)")
plt.tight_layout()
plt.savefig("reward_comparison.png", dpi=150)
plt.show()

# ── Plot 2: Arm / Price Selection Distribution ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Price Selection Distribution", fontsize=13)

arm_labels = [str(i) for i in range(len(prices))]

# Basic Bandit
axes[0].pie(arm_counts_bandit + 1e-9,
            labels=arm_labels, autopct='%1.1f%%', startangle=90)
axes[0].set_title("Basic Bandit")

# LinUCB
axes[1].pie(arm_counts_lin + 1e-9,
            labels=arm_labels, autopct='%1.1f%%', startangle=90)
axes[1].set_title("LinUCB")

# MARL — combined arm usage across all 3 agents
all_marl_actions = []
for s in marl.sellers:
    all_marl_actions.extend(marl.agents[s].actions_history)
marl_arm_counts = np.bincount(all_marl_actions, minlength=len(prices))
axes[2].pie(marl_arm_counts + 1e-9,
            labels=arm_labels, autopct='%1.1f%%', startangle=90)
axes[2].set_title("MARL (3-Agent IQL)")

plt.tight_layout()
plt.savefig("price_selection_distribution.png", dpi=150)
plt.show()

# ── Plot 3: DQN Category-Wise Prices with Margin ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("DQN: Category-Aware Pricing", fontsize=13)

prods  = list(dqn_prices.keys())
p_vals = [dqn_prices[p] for p in prods]
m_vals = [dqn_margins[p] for p in prods]
costs  = [UNIT_COSTS.get(p, 280) for p in prods]

bars = axes[0].bar(prods, p_vals, color="#5b9cf6", edgecolor="white")
axes[0].bar_label(bars, labels=[f"₹{v:.0f}" for v in p_vals], padding=4)
for i, cost in enumerate(costs):
    axes[0].plot([i-0.4, i+0.4], [cost, cost], "r--", lw=1.5)
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([0],[0],color="red",ls="--",lw=1.5,label="Unit cost")])
axes[0].set_title("DQN Suggested Price per Category")
axes[0].set_ylabel("Price (₹)")
axes[0].tick_params(axis="x", rotation=15)

bars2 = axes[1].bar(prods, m_vals, color="#2ecc71", edgecolor="white")
axes[1].bar_label(bars2, labels=[f"{v:.1f}%" for v in m_vals], padding=4)
axes[1].set_title("Profit Margin per Category")
axes[1].set_ylabel("Margin (%)")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("DQN_suggested_price_product.png", dpi=150)
plt.show()

# ── Plot 4: MARL Episode Training Curves ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Category-Aware MARL: 3 Sellers × 5 Product Categories\n"
             "Seller A=Electronics+Home | Seller B=Apparel+Sports | Seller C=Books",
             fontsize=12, fontweight="bold")

# Episode rewards
ax = axes[0, 0]
for s in marl.sellers:
    ax.plot(smooth(ep_total_rewards[s], 25), color=SELLER_COLORS[s], label=s, lw=2)
ax.set_title("Episode Rewards per Seller")
ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward (₹)")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

# Price convergence
ax = axes[0, 1]
for s in marl.sellers:
    ax.plot(smooth(ep_avg_prices[s], 25), color=SELLER_COLORS[s], label=s, lw=2)
ax.set_title("Avg Price per Episode (Nash Convergence?)")
ax.set_xlabel("Episode"); ax.set_ylabel("Avg Price (₹)")
ax.legend()

# Market share
ax = axes[1, 0]
for s in marl.sellers:
    ax.plot(smooth(ep_avg_shares[s], 25), color=SELLER_COLORS[s], label=s, lw=2)
ax.axhline(1/3, color="gray", ls="--", lw=1, label="Equal split (33%)")
ax.set_title("Market Share Evolution")
ax.set_xlabel("Episode"); ax.set_ylabel("Avg Market Share")
ax.legend()

# Final category prices
ax = axes[1, 1]
cat_final_prices = {}
for s in marl.sellers:
    for cat in SELLER_CATEGORIES[s]:
        cat_hist = [
            ep_avg_prices[s][ep]
            for ep in range(len(ep_avg_prices[s]))
            if ep >= N_EPISODES - 100
        ]
        cat_final_prices[cat] = np.mean(cat_hist) if cat_hist else 0

cat_names  = list(cat_final_prices.keys())
cat_vals   = [cat_final_prices[c] for c in cat_names]
cat_colors = ["#5b9cf6","#a78bfa","#f97316","#2ecc71","#f59e0b"]
bars_cat   = ax.bar(cat_names, cat_vals, color=cat_colors, edgecolor="white")
ax.bar_label(bars_cat, labels=[f"₹{v:.0f}" for v in cat_vals], padding=3)
ax.axhline(280, color="red", ls="--", lw=1.5, label="Unit cost ₹280")
ax.set_title("Learned Price per Category\n(Last 100 episodes avg)")
ax.set_ylabel("Price (₹)")
ax.legend()
ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("marl_category_results.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== PERFORMANCE SUMMARY ===")
print(f"  Basic Bandit total reward : ₹{np.sum(rewards_bandit):>12,.0f}")
print(f"  LinUCB total reward       : ₹{np.sum(rewards_lin):>12,.0f}")
print(f"  MARL (IQL) total reward   : ₹{total_marl_reward:>12,.0f}")
print(f"\n  Best Model: MARL (IQL)" if total_marl_reward > np.sum(rewards_lin)
      else f"\n  Best Model: LinUCB")
print(f"  Highest Reward: ₹{max(np.sum(rewards_bandit), np.sum(rewards_lin), total_marl_reward):,.0f}")
print("\n  DQN Category Prices:")
for prod, price in dqn_prices.items():
    print(f"    {prod:15s} → ₹{price:.0f}  (margin {dqn_margins[prod]}%)")
