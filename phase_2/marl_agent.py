import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# marl_agent.py  —  Upgraded: State-Aware IQL (Independent Q-Learning)
#
# WHAT CHANGED vs SimpleMARL:
#   Old: 2 agents, no state, shared reward, basic ε-greedy
#   New: 3 agents, state-aware Q-table, own reward per agent,
#        epsilon decay, competitor price observation
# ─────────────────────────────────────────────────────────────────────────────

class IQLAgent:
    """
    Independent Q-Learning agent for one seller.
    Learns a Q-table over (state_bin, action) pairs.
    State is discretised into bins so we can use a table
    without needing a neural network.
    """
    def __init__(self, agent_id, n_arms, n_state_bins=5,
                 lr=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=500):
        self.agent_id    = agent_id
        self.n_arms      = n_arms
        self.n_bins      = n_state_bins
        self.lr          = lr
        self.gamma       = gamma
        self.eps         = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.steps       = 0

        # Q-table shape: (inv_bin, demand_bin, comp_price_bin, n_arms)
        # Each dimension has n_state_bins levels → small but effective
        self.Q = np.zeros((n_state_bins, n_state_bins, n_state_bins, n_arms))

        # Tracking per agent
        self.rewards_history  = []
        self.actions_history  = []
        self.prices_history   = []

    def _discretise(self, state_vec):
        """
        Convert continuous state values to bin indices.
        state_vec = [inventory_norm, demand_norm, comp_price_norm]
        Each value expected in [0, 1].
        """
        bins = np.clip(
            (np.array(state_vec) * self.n_bins).astype(int),
            0, self.n_bins - 1
        )
        return tuple(bins)

    def select_action(self, state_vec, greedy=False):
        """ε-greedy with exponential decay."""
        self.eps = self.eps_end + (1.0 - self.eps_end) * \
                   np.exp(-self.steps / self.eps_decay)
        self.steps += 1

        if not greedy and np.random.rand() < self.eps:
            return np.random.randint(self.n_arms)

        s = self._discretise(state_vec)
        return int(np.argmax(self.Q[s]))

    def update(self, state_vec, action, reward, next_state_vec, done):
        """Q-learning update rule."""
        s  = self._discretise(state_vec)
        s2 = self._discretise(next_state_vec)

        q_current = self.Q[s][action]
        q_next    = 0.0 if done else np.max(self.Q[s2])
        td_target = reward + self.gamma * q_next
        td_error  = td_target - q_current

        self.Q[s][action] += self.lr * td_error

        self.rewards_history.append(reward)
        self.actions_history.append(action)


# ─────────────────────────────────────────────────────────────────────────────
# CategoryMARL  —  3 sellers, each with assigned product categories
# ─────────────────────────────────────────────────────────────────────────────

SELLER_CATEGORIES = {
    "Seller A": ["Electronics", "Home"],
    "Seller B": ["Apparel", "Sports"],
    "Seller C": ["Books"],
}

# Unit costs per category — from dataset calibration
UNIT_COSTS = {
    "Electronics": 320,
    "Home":        260,
    "Apparel":     240,
    "Sports":      220,
    "Books":       180,
}


class CategoryMARL:
    """
    3-seller MARL where each seller specialises in product categories.

    Key upgrades over SimpleMARL:
      1. 3 agents instead of 2
      2. Each agent observes a state: [inventory, demand, competitor_price]
      3. Each agent gets its OWN reward (not shared)
      4. Demand splits via softmax across sellers (competition)
      5. Category-specific unit costs affect profit
      6. Epsilon decays — starts exploring, becomes exploitative
    """
    def __init__(self, n_arms, prices,
                 n_state_bins=5, lr=0.1, gamma=0.95):
        self.sellers  = list(SELLER_CATEGORIES.keys())
        self.n_agents = len(self.sellers)
        self.n_arms   = n_arms
        self.prices   = np.array(prices)

        # One IQL agent per seller
        self.agents = {
            s: IQLAgent(i, n_arms, n_state_bins, lr, gamma)
            for i, s in enumerate(self.sellers)
        }

        # State tracking
        self.inventories  = {s: 500 for s in self.sellers}
        self.last_prices  = {s: prices[len(prices)//2] for s in self.sellers}
        self.demand_hist  = {s: [] for s in self.sellers}

        # Episode-level logs
        self.ep_rewards      = {s: [] for s in self.sellers}
        self.ep_prices       = {s: [] for s in self.sellers}
        self.ep_shares       = {s: [] for s in self.sellers}
        self.active_cats     = {}
        self._assign_categories()

    def _assign_categories(self):
        """Randomly pick one active category per seller per episode."""
        for s in self.sellers:
            cats = SELLER_CATEGORIES[s]
            self.active_cats[s] = np.random.choice(cats)

    def _get_state(self, seller):
        """
        Build 3-value normalised state for a seller:
          [inventory_norm, demand_norm, avg_competitor_price_norm]
        """
        inv_n    = self.inventories[seller] / 500.0
        dem_n    = min(np.mean(self.demand_hist[seller][-5:]) / 50.0, 1.0) \
                   if self.demand_hist[seller] else 0.5
        # competitor price = average of other sellers' last prices
        others   = [self.last_prices[o] for o in self.sellers if o != seller]
        comp_n   = np.mean(others) / max(self.prices)
        return np.clip([inv_n, dem_n, comp_n], 0, 1)

    def _market_share(self, prices_dict):
        """
        Softmax demand split — lower price → higher share.
        β = 0.004 calibrated from dataset competitor sensitivity.
        """
        beta   = 0.004
        scores = {s: np.exp(-beta * prices_dict[s]) for s in self.sellers}
        total  = sum(scores.values())
        return {s: scores[s] / total for s in self.sellers}

    def select_actions(self):
        """Each agent picks its own action from its own state."""
        actions = {}
        states  = {}
        for s in self.sellers:
            state        = self._get_state(s)
            actions[s]   = self.agents[s].select_action(state)
            states[s]    = state
        return actions, states

    def step(self, actions, states, base_demand=40):
        """
        Simulate one market step.
        Returns per-seller rewards and info dict.
        """
        prices_dict = {s: self.prices[actions[s]] for s in self.sellers}
        shares      = self._market_share(prices_dict)

        rewards      = {}
        next_states  = {}
        units_sold   = {}

        for s in self.sellers:
            price     = prices_dict[s]
            share     = shares[s]
            cat       = self.active_cats[s]
            unit_cost = UNIT_COSTS[cat]

            # Units sold = base demand × market share + noise
            units = max(0, int(base_demand * share + np.random.normal(0, 3)))
            units = min(units, self.inventories[s])

            revenue = price * units
            profit  = (price - unit_cost) * units

            # Small penalty if pricing way above competitor
            min_comp = min(prices_dict[o] for o in self.sellers if o != s)
            penalty  = 0.05 * max(0, price - min_comp - 150)
            reward   = profit - penalty

            # Update state
            self.inventories[s] = max(0, self.inventories[s] - units)
            self.last_prices[s] = price
            self.demand_hist[s].append(units)
            units_sold[s]       = units

            rewards[s]     = reward
            next_states[s] = self._get_state(s)

            # Log
            self.ep_rewards[s].append(reward)
            self.ep_prices[s].append(price)
            self.ep_shares[s].append(share)

        return rewards, next_states, shares, prices_dict, units_sold

    def update_agents(self, states, actions, rewards, next_states, done):
        """Each agent updates its own Q-table independently."""
        for s in self.sellers:
            self.agents[s].update(
                states[s], actions[s], rewards[s],
                next_states[s], done
            )

    def reset_episode(self):
        """Reset inventories and pick new active categories."""
        self.inventories = {s: 500 for s in self.sellers}
        self.demand_hist = {s: [] for s in self.sellers}
        self._assign_categories()
