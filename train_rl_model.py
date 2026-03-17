import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# ===================== CONFIG =====================
SEED = 42
WINDOW = 24
GAMMA = 0.99
LR_POLICY = 3e-4
LR_VALUE = 5e-4
EPOCHS = 10
EPISODES_PER_EPOCH = 50
MAX_EPISODE_LEN = 128
HOLDOUT_RATIO = 0.1

DATA_PATH = "eaf_temp.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float32)

# ===================== SEED =====================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH)

if "DATETIME" not in df.columns or "TEMP" not in df.columns:
    raise ValueError("Dataset must contain DATETIME and TEMP columns")

df["DATETIME"] = pd.to_datetime(df["DATETIME"])
df = df.sort_values("DATETIME").reset_index(drop=True)

temps = df["TEMP"].astype(float).values.reshape(-1, 1)

# ===================== SPLIT =====================
split_idx = int(len(temps) * (1 - HOLDOUT_RATIO))
temps_train = temps[:split_idx]
temps_test = temps[split_idx:]

# ===================== SCALE =====================
scaler = StandardScaler()
temps_train_sc = scaler.fit_transform(temps_train)
temps_test_sc = scaler.transform(temps_test)

def make_indexable(series, window):
    return list(range(0, len(series) - window))

train_indices = make_indexable(temps_train_sc, WINDOW)
test_indices = make_indexable(temps_test_sc, WINDOW)

# ===================== ENV =====================
@dataclass
class OfflineSeriesEnv:
    series: np.ndarray
    indices: list
    window: int
    max_len: int

    def reset(self):
        self.start = random.choice(self.indices)
        self.t = 0
        self.cur = self.start
        return self.series[self.cur:self.cur + self.window].astype(np.float32).flatten()

    def step(self, action):
        true_next = self.series[self.cur + self.window][0]
        reward = - (action - true_next) ** 2

        self.cur += 1
        self.t += 1

        done = (self.cur + self.window >= len(self.series)) or (self.t >= self.max_len)

        if not done:
            obs = self.series[self.cur:self.cur + self.window].astype(np.float32).flatten()
        else:
            obs = np.zeros(self.window, dtype=np.float32)

        return obs, reward, done, {"true": true_next, "pred": action}

# ===================== MODELS =====================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = self.net(x)
        mu = self.mu(z).squeeze(-1)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, x):
        mu, std = self(x)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob

class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ===================== INIT =====================
train_env = OfflineSeriesEnv(temps_train_sc, train_indices, WINDOW, MAX_EPISODE_LEN)
test_env = OfflineSeriesEnv(temps_test_sc, test_indices, WINDOW, MAX_EPISODE_LEN)

policy = PolicyNet(WINDOW).to(DEVICE)
value = ValueNet(WINDOW).to(DEVICE)

opt_policy = optim.Adam(policy.parameters(), lr=LR_POLICY)
opt_value = optim.Adam(value.parameters(), lr=LR_VALUE)

# ===================== TRAIN =====================
def discount_rewards(rewards, gamma):
    R = 0
    out = []
    for r in reversed(rewards):
        R = r + gamma * R
        out.insert(0, R)
    return out

print("🚀 Training Started...\n")

for epoch in range(EPOCHS):
    total_rewards = []

    for _ in range(EPISODES_PER_EPOCH):
        obs = train_env.reset()
        rewards, log_probs, values = [], [], []

        for _ in range(MAX_EPISODE_LEN):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            val = value(obs_t)
            action, log_prob = policy.sample(obs_t)

            action_val = action.item()
            next_obs, reward, done, _ = train_env.step(action_val)

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(val)

            obs = next_obs
            if done:
                break

        returns = torch.tensor(discount_rewards(rewards, GAMMA), dtype=torch.float32).to(DEVICE)

        values = torch.cat(values).squeeze()
        log_probs = torch.cat(log_probs)

        advantage = returns - values.detach()

        policy_loss = -(log_probs * advantage).mean()
        value_loss = nn.functional.mse_loss(values, returns)

        opt_policy.zero_grad()
        opt_value.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        opt_policy.step()
        opt_value.step()

        total_rewards.append(sum(rewards))

    print(f"Epoch {epoch+1} | Avg Reward: {np.mean(total_rewards):.4f}")

# ===================== EVALUATE =====================
print("\n📊 Evaluating...")

preds, trues = [], []

obs = test_env.reset()

for _ in range(500):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mu, _ = policy(obs_t)

    pred = mu.item()
    next_obs, _, done, info = test_env.step(pred)

    preds.append(info["pred"])
    trues.append(info["true"])

    obs = next_obs
    if done:
        break

preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
trues = scaler.inverse_transform(np.array(trues).reshape(-1,1)).flatten()

# ===================== METRICS =====================
mse = np.mean((preds - trues)**2)
mae = np.mean(np.abs(preds - trues))

print(f"\n✅ Test MSE: {mse:.4f}")
print(f"✅ Test MAE: {mae:.4f}")

# ===================== PLOT =====================
plt.figure(figsize=(10,4))
plt.plot(trues[:200], label="True")
plt.plot(preds[:200], label="Predicted")
plt.legend()
plt.title("RL Temperature Prediction")
plt.show()

# ===================== SAVE =====================
os.makedirs("models", exist_ok=True)
torch.save(policy.state_dict(), "models/policy.pt")
torch.save(value.state_dict(), "models/value.pt")

print("\n💾 Model saved in 'models/' folder")