import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

st.set_page_config(page_title="Dynamic Pricing System", layout="wide")

st.title("📊 Dynamic Pricing Intelligence System")

# -----------------------------------
# MODEL FUNCTIONS
# -----------------------------------

def epsilon_greedy(prices, rewards):
    Q = np.zeros(len(prices))
    N = np.zeros(len(prices))

    for i in range(len(rewards)):
        if np.random.rand() < 0.1:
            action = np.random.randint(len(prices))
        else:
            action = np.argmax(Q)

        reward = rewards[i]

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

    return prices[np.argmax(Q)]


def run_linucb(states, rewards, prices):
    n_arms = len(prices)
    d = states.shape[1]

    A = [np.identity(d) for _ in range(n_arms)]
    b = [np.zeros(d) for _ in range(n_arms)]

    for i in range(len(states)):
        s = states[i]

        p_vals = []
        for a in range(n_arms):
            A_inv = np.linalg.inv(A[a])
            theta = A_inv @ b[a]
            p = theta @ s + 1.0 * np.sqrt(s @ A_inv @ s)
            p_vals.append(p)

        action = np.argmax(p_vals)
        reward = rewards[i]

        A[action] += np.outer(s, s)
        b[action] += reward * s

    # pick best arm
    scores = [np.linalg.norm(b[a]) for a in range(n_arms)]
    return prices[np.argmax(scores)]


def run_dqn(states, rewards, prices):
    # Simple neural network approximation
    X = states
    y = rewards

    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200)
    model.fit(X, y)

    avg_state = np.mean(states, axis=0)

    predictions = []

    for p in prices:
        state = np.append(avg_state, p)
        pred = model.predict([state])[0]
        predictions.append(pred)

    return prices[np.argmax(predictions)]


# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader("📂 Upload Online Retail Dataset", type=["csv", "xlsx"])

if uploaded_file:

    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # REQUIRED COLUMNS CHECK
    # -----------------------------------

    required_cols = ["UnitPrice", "Quantity", "InvoiceDate", "CustomerID"]

    if not all(col in df.columns for col in required_cols):
        st.error("Dataset must contain: UnitPrice, Quantity, InvoiceDate, CustomerID")
        st.stop()

    # -----------------------------------
    # CLEANING
    # -----------------------------------

    df = df.dropna(subset=required_cols)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # -----------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["hour"] = df["InvoiceDate"].dt.hour
    df["is_weekend"] = (df["InvoiceDate"].dt.weekday >= 5).astype(int)

    df["customer_type"] = df["CustomerID"].duplicated().astype(int)

    df["demand"] = df["Quantity"]
    df["revenue"] = df["UnitPrice"] * df["Quantity"]

    st.subheader("🧹 Processed Data")
    st.dataframe(df.head())

    # -----------------------------------
    # PREPARE DATA FOR MODELS
    # -----------------------------------

    # Normalize price to fixed bins (important!)
    price_bins = np.linspace(df["UnitPrice"].min(), df["UnitPrice"].max(), 5)
    df["price_bin"] = np.digitize(df["UnitPrice"], price_bins)

    prices = sorted(df["price_bin"].unique())

    states = df[["hour", "is_weekend", "customer_type", "demand"]].values
    rewards = df["revenue"].values

    # -----------------------------------
    # RUN MODELS
    # -----------------------------------

    st.subheader("⚙️ Model Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        bandit_price = epsilon_greedy(prices, rewards)
        st.metric("ε-Greedy", bandit_price)

    with col2:
        linucb_price = run_linucb(states, rewards, prices)
        st.metric("LinUCB", linucb_price)

    with col3:
        dqn_price = run_dqn(states, rewards, prices)
        st.metric("DQN", dqn_price)

    # -----------------------------------
    # USER SIMULATION
    # -----------------------------------

    st.subheader("🎯 Pricing Simulator")

    hour = st.slider("Hour", 0, 23, 12)
    is_weekend = st.selectbox("Weekend?", [0, 1])
    customer_type = st.selectbox("Customer Type", ["new", "returning"])
    demand = st.slider("Demand Level", 0.0, 10.0, 5.0)

    customer_val = 1 if customer_type == "returning" else 0

    input_state = np.array([hour, is_weekend, customer_val, demand])

    # Evaluate best price for this state
    best_prices = []

    for p in prices:
        state = np.append(input_state, p)
        score = np.sum(state)  # simple heuristic
        best_prices.append(score)

    best_price = prices[np.argmax(best_prices)]

    st.success(f"💡 Suggested Optimal Price (for given scenario): {best_price}")

    # -----------------------------------
    # SUMMARY
    # -----------------------------------

    st.subheader("📌 Summary")

    st.write("""
    - ε-Greedy: Learns a single best price overall (baseline)
    - LinUCB: Adapts price based on contextual features
    - DQN: Uses neural networks to learn complex pricing patterns
    """)