
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import defaultdict
import math

from Game_2048_BE import Game2048Env

def encode_state(board):
    flat = []
    for i in range(4):
        for j in range(4):
            v = board[i][j]
            flat.append(0.0 if v == 0 else np.log2(v) / 16.0)
    
    empty_count = sum(1 for i in range(4) for j in range(4) if board[i][j] == 0)
    flat.append(empty_count / 16.0)
    
    max_tile = max(max(row) for row in board)
    flat.append(0.0 if max_tile == 0 else np.log2(max_tile) / 11.0)
    
    merge_count = 0
    for i in range(4):
        for j in range(3):
            if board[i][j] != 0 and board[i][j] == board[i][j+1]:
                merge_count += 1
    for i in range(3):
        for j in range(4):
            if board[i][j] != 0 and board[i][j] == board[i+1][j]:
                merge_count += 1
    flat.append(merge_count / 24.0)
    
    monotonicity = 0.0
    for i in range(4):
        row = [board[i][j] for j in range(4) if board[i][j] != 0]
        if len(row) > 1:
            increasing = sum(1 for j in range(len(row)-1) if row[j] <= row[j+1])
            decreasing = sum(1 for j in range(len(row)-1) if row[j] >= row[j+1])
            monotonicity += max(increasing, decreasing) / (len(row) - 1)
    for j in range(4):
        col = [board[i][j] for i in range(4) if board[i][j] != 0]
        if len(col) > 1:
            increasing = sum(1 for i in range(len(col)-1) if col[i] <= col[i+1])
            decreasing = sum(1 for i in range(len(col)-1) if col[i] >= col[i+1])
            monotonicity += max(increasing, decreasing) / (len(col) - 1)
    flat.append(monotonicity / 8.0)
    
    return np.array(flat, dtype=np.float32)


class PolicyNet(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class PolicyNet256(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ValueNet256(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"ValueNet256 expects 2D input, got {x.dim()}D with shape {x.shape}")
        return self.net(x).squeeze(-1)

class PolicyNetLSTM(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, lstm_hidden=256, output_dim=4):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, output_dim)
        self.hidden = None

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        encoded = self.encoder(x)
        lstm_out, self.hidden = self.lstm(encoded, hidden)
        logits = self.head(lstm_out[:, -1, :])
        return logits

    def reset_hidden(self):
        self.hidden = None


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.ln_h = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden):
        h, c = self.cell(x, hidden)
        h = self.ln_h(h)
        c = self.ln_c(c)
        return h, c

class PolicyNetLSTM128(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, lstm_hidden=128, output_dim=4, num_layers=1):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm_cell = LayerNormLSTMCell(hidden_dim, lstm_hidden)
        self.head = nn.Linear(lstm_hidden, output_dim)
        self.hidden = None

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        encoded = self.encoder(x)
        if hidden is None:
            if self.hidden is None:
                h = torch.zeros(batch_size, self.lstm_hidden, device=x.device)
                c = torch.zeros(batch_size, self.lstm_hidden, device=x.device)
            else:
                h, c = self.hidden
                if h.size(0) != batch_size:
                    h = h[:batch_size] if h.size(0) > batch_size else torch.cat([h, torch.zeros(batch_size - h.size(0), self.lstm_hidden, device=x.device)])
                    c = c[:batch_size] if c.size(0) > batch_size else torch.cat([c, torch.zeros(batch_size - c.size(0), self.lstm_hidden, device=x.device)])
        else:
            h, c = hidden
            if h.dim() == 3:
                h = h[-1]
                c = c[-1]
        outputs = []
        for t in range(seq_len):
            h, c = self.lstm_cell(encoded[:, t, :], (h, c))
            outputs.append(h)
        
        self.hidden = (h, c)
        lstm_out = torch.stack(outputs, dim=1)
        logits = self.head(lstm_out[:, -1, :])
        return logits

    def reset_hidden(self, batch_size=1, device="cpu"):
        h_0 = torch.zeros(batch_size, self.lstm_hidden, device=device)
        c_0 = torch.zeros(batch_size, self.lstm_hidden, device=device)
        self.hidden = (h_0, c_0)
        return self.hidden

class ValueNetLSTM(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, lstm_hidden=256):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)
        self.hidden = None

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        encoded = self.encoder(x)
        lstm_out, self.hidden = self.lstm(encoded, hidden)
        value = self.head(lstm_out[:, -1, :]).squeeze(-1)
        return value

    def reset_hidden(self):
        self.hidden = None


def get_valid_actions(env, state=None):
    # Get valid actions by checking if board changes after move simulation.
    # CRITICAL: Use env.board directly, not passed state (to avoid sync issues).

    if state is None:
        board = env.board
    else:
        board = state
    
    valid_actions = []
    for action in range(4):
        board_copy = [row[:] for row in board]
        _, _, changed = env._simulate_move(board_copy, action)
        if changed:
            valid_actions.append(action)
    return valid_actions


def apply_action_mask(logits, valid_actions, device="cpu"):
    batch_size = logits.shape[0]
    action_mask = torch.zeros(batch_size, 4, dtype=torch.bool, device=device)
    
    if valid_actions is not None:
        for valid_action in valid_actions:
            action_mask[:, valid_action] = True
    else:
        action_mask.fill_(True)
    
    masked_logits = logits.clone()
    masked_logits[~action_mask] = -20.0
    return masked_logits, action_mask


def safe_entropy(logits, action_mask=None):
    eps = 1e-8
    logits = torch.clamp(logits, -20, 20)
    
    if action_mask is not None:
        masked_logits = logits.clone()
        masked_logits[~action_mask] = -20.0
        logits = masked_logits
    
    probs = torch.softmax(logits, dim=-1)
    
    if action_mask is not None:
        probs = probs * action_mask.float()
        probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)
    
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    if torch.isnan(entropy).any() or torch.isinf(entropy).any():
        return None
    
    return entropy


def select_action(policy_net, state, env=None, device="cpu", action_mask=None, use_masking=True):
    state_vec = encode_state(state)
    state_tensor = torch.from_numpy(state_vec).to(device).unsqueeze(0)

    logits = policy_net(state_tensor)
    logits = torch.clamp(logits, -20, 20)
    
    if use_masking:
        if action_mask is None:
            if env is None:
                raise ValueError("Either env or action_mask must be provided when use_masking=True")
            valid_actions = get_valid_actions(env, state)
            masked_logits, action_mask = apply_action_mask(logits, valid_actions, device=device)
        else:
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -20.0
    else:
        masked_logits = logits
        action_mask = None
    
    dist = Categorical(logits=masked_logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    
    entropy = safe_entropy(masked_logits, action_mask)
    if entropy is not None:
        entropy = entropy.item()
    else:
        entropy = 0.0
    
    return action.item(), log_prob, entropy, action_mask


def select_action_with_value(policy_net, value_net, state, env=None, device="cpu", action_mask=None, use_masking=True):
    state_vec = encode_state(state)
    state_tensor = torch.from_numpy(state_vec).to(device).unsqueeze(0)

    logits = policy_net(state_tensor)
    logits = torch.clamp(logits, -20, 20)
    
    if use_masking:
        if action_mask is None:
            if env is None:
                raise ValueError("Either env or action_mask must be provided when use_masking=True")
            valid_actions = get_valid_actions(env, state)
            masked_logits, action_mask = apply_action_mask(logits, valid_actions, device=device)
        else:
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -20.0
    else:
        masked_logits = logits
        action_mask = None
    
    dist = Categorical(logits=masked_logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    
    value = value_net(state_tensor)
    
    entropy = safe_entropy(masked_logits, action_mask)
    if entropy is not None:
        entropy = entropy.item()
    else:
        entropy = 0.0
    
    return action.item(), log_prob, value, entropy, action_mask


def select_action_lstm(policy_net, state, hidden=None, env=None, device="cpu", action_mask=None, use_masking=True):
    state_vec = encode_state(state)
    state_tensor = torch.from_numpy(state_vec).to(device).unsqueeze(0)
    
    if hidden is None:
        hidden = policy_net.hidden
    
    logits = policy_net(state_tensor, hidden)
    logits = torch.clamp(logits, -20, 20)
    
    if use_masking:
        if action_mask is None:
            if env is None:
                raise ValueError("Either env or action_mask must be provided when use_masking=True")
            valid_actions = get_valid_actions(env, state)
            masked_logits, action_mask = apply_action_mask(logits, valid_actions, device=device)
        else:
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -20.0
    else:
        masked_logits = logits
        action_mask = None
    
    dist = Categorical(logits=masked_logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    
    entropy = safe_entropy(masked_logits, action_mask)
    if entropy is not None:
        entropy = entropy.item()
    else:
        entropy = 0.0
    
    return action.item(), log_prob, entropy, action_mask, policy_net.hidden


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

class RolloutBuffer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.obs = []
        self.actions = []
        self.logp_old = []
        self.value_old = []
        self.rewards = []
        self.dones = []
        self.masks = []
        self.h0 = []
        self.c0 = []
    
    def add(self, obs, action, logp_old, value_old, reward, done, mask, h0, c0):
        self.obs.append(obs)
        self.actions.append(action)
        self.logp_old.append(logp_old)
        self.value_old.append(value_old)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)
        self.h0.append(h0)
        self.c0.append(c0)
    
    def size(self):
        return len(self.obs)
    
    def get_all(self):
        return {
            'obs': self.obs,
            'actions': self.actions,
            'logp_old': self.logp_old,
            'value_old': self.value_old,
            'rewards': self.rewards,
            'dones': self.dones,
            'masks': self.masks,
            'h0': self.h0,
            'c0': self.c0
        }

def train_reinforce(
    num_episodes=1000,
    gamma=0.99,
    lr=1e-3,
    print_every=100,
    device="cpu",
    entropy_coef=0.01,
    baseline_momentum=0.9,
    max_steps_per_episode=10000,
):
    policy_net = PolicyNet(input_dim=20).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    env = Game2048Env()
    all_scores = []
    best_score = 0
    best_tile = 0
    baseline = 0.0

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        entropy_terms = []
        steps = 0

        while not env.done and steps < max_steps_per_episode:
            action, log_prob, entropy, _ = select_action(
                policy_net, state, env=env, device=device, use_masking=True
            )
            next_state, reward, done, info = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            entropy_terms.append(entropy)

            state = next_state
            steps += 1

        score = info["score"]
        all_scores.append(score)
        best_score = max(best_score, score)
        episode_max_tile = max(max(row) for row in env.board)
        best_tile = max(best_tile, episode_max_tile)

        returns = compute_returns(rewards, gamma=gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        G_mean = returns.mean().item()
        baseline = (1 - baseline_momentum) * baseline + baseline_momentum * G_mean

        advantages = returns - baseline
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            else:
                advantages = advantages - advantages.mean()
        else:
            advantages = advantages - advantages.mean()
        log_probs_t = torch.stack(log_probs)
        pg_loss = -(log_probs_t * advantages.detach()).sum()

        entropy_bonus = torch.tensor(entropy_terms, dtype=torch.float32, device=device).mean()
        loss = pg_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % print_every == 0:
            avg_score = sum(all_scores[-print_every:]) / len(all_scores[-print_every:])
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Score: {score} | Avg({print_every}): {avg_score:.1f} | "
                f"Baseline: {baseline:.2f} | Loss: {loss.item():.3f} | "
                f"Best score: {best_score} | Best tile: {best_tile}"
            )

    print(
        f"Training complete | Best score: {best_score} | Best tile: {best_tile}"
    )
    return policy_net, all_scores


def train_actor_critic(
    num_episodes=1000,
    gamma=0.99,
    lr=1e-3,
    print_every=100,
    device="cpu",
    entropy_coef=0.01,
    value_coef=0.5,
    max_steps_per_episode=10000,
):
    policy_net = PolicyNet(input_dim=20).to(device)
    value_net = ValueNet(input_dim=20).to(device)
    optimizer = optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()), lr=lr
    )

    env = Game2048Env()
    all_scores = []
    best_score = 0
    best_tile = 0

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        values = []
        entropy_terms = []
        steps = 0

        while not env.done and steps < max_steps_per_episode:
            action, log_prob, value, entropy, _ = select_action_with_value(
                policy_net, value_net, state, env=env, device=device, use_masking=True
            )
            next_state, reward, done, info = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            entropy_terms.append(entropy)

            state = next_state
            steps += 1

        score = info["score"]
        all_scores.append(score)
        best_score = max(best_score, score)
        episode_max_tile = max(max(row) for row in env.board)
        best_tile = max(best_tile, episode_max_tile)
        returns = compute_returns(rewards, gamma=gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.stack(values)

        advantages = returns - values_t.detach()
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            else:
                advantages = advantages - advantages.mean()
        else:
            advantages = advantages - advantages.mean()
        log_probs_t = torch.stack(log_probs)
        policy_loss = -(log_probs_t * advantages.detach()).sum()
        value_loss = torch.mean(torch.clamp((values_t - returns)**2, max=1.0)) * len(values_t)
        
        entropy_bonus = torch.tensor(entropy_terms, dtype=torch.float32, device=device).mean()
        
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy_net.parameters()) + list(value_net.parameters()), 0.5
        )
        optimizer.step()

        if episode % print_every == 0:
            avg_score = sum(all_scores[-print_every:]) / len(all_scores[-print_every:])
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Score: {score} | Avg({print_every}): {avg_score:.1f} | "
                f"Policy Loss: {policy_loss.item():.3f} | Value Loss: {value_loss.item():.3f} | "
                f"Best score: {best_score} | Best tile: {best_tile}"
            )

    print(
        f"Training complete | Best score: {best_score} | Best tile: {best_tile}"
    )
    return policy_net, value_net, all_scores


def compute_gae(rewards, values, next_values, gamma=0.99, lam=0.95, device="cpu"):
    if isinstance(values, list):
        values = torch.stack([v.squeeze() if isinstance(v, torch.Tensor) else torch.tensor(v, device=device) for v in values])
    elif not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32, device=device)
    if isinstance(next_values, list):
        next_values = torch.stack([v.squeeze() if isinstance(v, torch.Tensor) else torch.tensor(v, device=device) for v in next_values])
    elif not isinstance(next_values, torch.Tensor):
        next_values = torch.tensor(next_values, dtype=torch.float32, device=device)
    if isinstance(rewards, list):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    elif not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] + gamma * next_values[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    advantages = torch.stack(advantages) if isinstance(advantages[0], torch.Tensor) else torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = advantages + values
    return advantages, returns


def train_ppo_advanced(
    num_episodes=10000,
    gamma=0.99,
    lr=5e-5,
    print_every=100,
    eval_every=1000,
    device="cpu",
    entropy_coef=4e-4,
    value_coef=0.25,
    max_steps_per_episode=3000,
    clip_epsilon=0.15,
    ppo_epochs=1,
    batch_size=256,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    use_large_network=True,
    use_lstm=False,
    lstm_hidden=256,
    checkpoint_dir="checkpoints",
    enable_burst=True,
    burst_interval=1000,
    burst_duration=200,
    burst_entropy_multiplier=2.0,
    burst_clip_epsilon=None,
):
    
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    env = Game2048Env()
    
    if use_lstm:
        policy_net = PolicyNetLSTM(input_dim=20, hidden_dim=256, lstm_hidden=lstm_hidden).to(device)
        value_net = ValueNetLSTM(input_dim=20, hidden_dim=256, lstm_hidden=lstm_hidden).to(device)
    elif use_large_network:
        policy_net = PolicyNet256(input_dim=20).to(device)
        value_net = ValueNet256(input_dim=20).to(device)
    else:
        policy_net = PolicyNet(input_dim=20).to(device)
        value_net = ValueNet(input_dim=20).to(device)
    
    optimizer = optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()), lr=lr
    )

    all_scores = []
    best_score = 0
    best_tile = 0
    training_scores = []
    eval_results = []
    ever_reached_1024 = False
    
    for episode in range(num_episodes):
        current_entropy_coef = entropy_coef
        current_clip_epsilon = clip_epsilon
        
        if enable_burst:
            burst_start = (episode // burst_interval) * burst_interval
            burst_end = burst_start + burst_duration
            if burst_start <= episode < burst_end:
                current_entropy_coef = entropy_coef * burst_entropy_multiplier
                if burst_clip_epsilon is not None:
                    current_clip_epsilon = burst_clip_epsilon
        if ever_reached_1024:
            current_entropy_coef = entropy_coef * 0.5
        
        state = env.reset()
        states = []
        actions = []
        action_masks = []
        rewards = []
        log_probs_old = []
        values = []
        old_max_tile = 0
        milestone_512_seen = False
        milestone_1024_seen = False
        milestone_2048_seen = False
        steps = 0

        if use_lstm:
            policy_net.reset_hidden()
            value_net.reset_hidden()
        
        while not env.done and steps < max_steps_per_episode:
            current_max_tile = max(max(row) for row in env.board)
            
            if use_lstm:
                action, log_prob, value, entropy, action_mask, hidden = select_action_lstm(
                    policy_net, state, hidden=policy_net.hidden, env=env, device=device, use_masking=True
                )
            else:
                action, log_prob, value, entropy, action_mask = select_action_with_value(
                    policy_net, value_net, state, env=env, device=device, use_masking=True
            )

            next_state, reward, done, info = env.step(action)
            empty_count = sum(1 for i in range(4) for j in range(4) if env.board[i][j] == 0)
            empty_cells_ratio = empty_count / 16.0
            
            monotonicity = 0.0
            for i in range(4):
                row = [env.board[i][j] for j in range(4) if env.board[i][j] != 0]
                if len(row) > 1:
                    increasing = sum(1 for j in range(len(row)-1) if row[j] <= row[j+1])
                    decreasing = sum(1 for j in range(len(row)-1) if row[j] >= row[j+1])
                    row_score = max(increasing, decreasing) / (len(row) - 1) if len(row) > 1 else 0
                    monotonicity += row_score
            for j in range(4):
                col = [env.board[i][j] for i in range(4) if env.board[i][j] != 0]
                if len(col) > 1:
                    increasing = sum(1 for i in range(len(col)-1) if col[i] <= col[i+1])
                    decreasing = sum(1 for i in range(len(col)-1) if col[i] >= col[i+1])
                    col_score = max(increasing, decreasing) / (len(col) - 1) if len(col) > 1 else 0
                    monotonicity += col_score
            monotonicity_score = monotonicity / 8.0
            
            if current_max_tile > old_max_tile:
                if current_max_tile >= 2048 and not milestone_2048_seen:
                    reward += 2.0
                    milestone_2048_seen = True
                    if not ever_reached_1024:
                        ever_reached_1024 = True
                elif current_max_tile >= 1024 and not milestone_1024_seen:
                    reward += 0.5
                    reward += 0.1 * monotonicity_score
                    milestone_1024_seen = True
                    if not ever_reached_1024:
                        ever_reached_1024 = True
                elif current_max_tile >= 512 and not milestone_512_seen:
                    reward += 0.002
                    reward += 0.001 * empty_cells_ratio
            
            if steps > 0:
                prev_empty = sum(1 for i in range(4) for j in range(4) if state[i][j] == 0) / 16.0
                delta_empty = empty_cells_ratio - prev_empty
                prev_mono = monotonicity_score
                delta_mono = monotonicity_score - prev_mono
                reward += 0.01 * delta_empty + 0.02 * delta_mono
            if not env._can_move() and not done:
                reward -= 0.1
            if done:
                reward -= 1.0
            
            states.append(state)
            actions.append(action)
            action_masks.append(action_mask)
            rewards.append(reward)
            log_probs_old.append(log_prob.detach() if isinstance(log_prob, torch.Tensor) else log_prob)
            values.append(value.detach() if isinstance(value, torch.Tensor) else value)

            old_max_tile = current_max_tile
            state = next_state
            steps += 1

        if len(rewards) == 0:
            continue

        score = info["score"]
        all_scores.append(score)
        best_score = max(best_score, score)
        episode_max_tile = max(max(row) for row in env.board)
        best_tile = max(best_tile, episode_max_tile)
        training_scores.append(score)
        values_squeezed = [v.squeeze() if isinstance(v, torch.Tensor) else torch.tensor(v, device=device) for v in values]
        next_values = values_squeezed[1:] + [torch.tensor(0.0, device=device)]
        next_values = torch.stack(next_values)
        values_t = torch.stack(values_squeezed)
        
        advantages, returns = compute_gae(
            rewards, values, next_values, gamma=gamma, lam=gae_lambda, device=device
        )
        advantages = advantages.detach()
        returns = returns.detach()
        
        adv_std = advantages.std()
        if adv_std > 1e-6:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        
        advantages = torch.clamp(advantages, -10.0, 10.0)
        states_t = torch.stack([torch.from_numpy(encode_state(s)).float().to(device) for s in states])
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        log_probs_old_detached = []
        for lp in log_probs_old:
            if isinstance(lp, torch.Tensor):
                log_probs_old_detached.append(lp.detach())
            else:
                log_probs_old_detached.append(torch.tensor(lp, device=device))
        log_probs_old_t = torch.stack(log_probs_old_detached)
        if action_masks[0] is not None:
            if isinstance(action_masks[0], torch.Tensor):
                if action_masks[0].dim() == 1:
                    batch_action_masks = torch.stack([am.squeeze(0) for am in action_masks])
                else:
                    batch_action_masks = torch.stack([am[0] for am in action_masks])
            else:
                batch_action_masks = None
        else:
            batch_action_masks = None
        
      
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        clip_fractions = []
        
      
        if batch_size is not None and len(states_t) > batch_size:
            indices = torch.randperm(len(states_t), device=device)
            states_t = states_t[indices]
            actions_t = actions_t[indices]
            log_probs_old_t = log_probs_old_t[indices]
            advantages = advantages[indices]
            returns = returns[indices]
            if batch_action_masks is not None:
                batch_action_masks = batch_action_masks[indices]
        
        for epoch in range(ppo_epochs):
            if batch_size is not None and len(states_t) > batch_size:
                for start_idx in range(0, len(states_t), batch_size):
                    end_idx = min(start_idx + batch_size, len(states_t))
                    batch_states = states_t[start_idx:end_idx]
                    batch_actions = actions_t[start_idx:end_idx]
                    batch_log_probs_old = log_probs_old_t[start_idx:end_idx]
                    batch_advantages = advantages[start_idx:end_idx]
                    batch_returns = returns[start_idx:end_idx]
                    
                    if batch_action_masks is not None:
                        batch_action_masks_batch = batch_action_masks[start_idx:end_idx]
                    else:
                        batch_action_masks_batch = None
                    
                  
                    if use_lstm:
                        logits = policy_net(batch_states.unsqueeze(1))
                    else:
                        logits = policy_net(batch_states)
                    
                    logits = torch.clamp(logits, -20, 20)

                  
                    if batch_action_masks_batch is not None:
                        masked_logits = logits.masked_fill(~batch_action_masks_batch, -20.0)
                    else:
                        masked_logits = logits
                    
                  
                    if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                        continue
                    
                    dist = Categorical(logits=masked_logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    
                  
                    ratio = torch.exp(new_log_probs - batch_log_probs_old)
                    
                  
                    policy_loss_1 = ratio * batch_advantages
                    policy_loss_2 = torch.clamp(ratio, 1 - current_clip_epsilon, 1 + current_clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                    if use_lstm:
                        values_pred = value_net(batch_states.unsqueeze(1))
                    else:
                        values_pred = value_net(batch_states)
                    value_loss = torch.mean(torch.clamp((values_pred - batch_returns)**2, max=1.0))
        
                  
                    entropy = safe_entropy(masked_logits, batch_action_masks_batch)
                    if entropy is None:
                        entropy = torch.tensor(0.0, device=device)
                    else:
                        entropy = entropy.mean()
                    
                  
                    loss = policy_loss + value_coef * value_loss - current_entropy_coef * entropy
                    
                    optimizer.zero_grad()
                    loss.backward(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(
                        list(policy_net.parameters()) + list(value_net.parameters()), max_grad_norm
                    )
                    optimizer.step()
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item() if isinstance(entropy, torch.Tensor) else entropy
                    
                  
                    clip_fraction = ((ratio < (1 - current_clip_epsilon)) | (ratio > (1 + current_clip_epsilon))).float().mean().item()
                    clip_fractions.append(clip_fraction)
            else:
              
                if use_lstm:
                    logits = policy_net(states_t.unsqueeze(1))
                else:
                    logits = policy_net(states_t)
                
                logits = torch.clamp(logits, -20, 20)
                
                if batch_action_masks is not None:
                    masked_logits = logits.masked_fill(~batch_action_masks, -20.0)
                else:
                    masked_logits = logits
                
                if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                    continue
                
                dist = Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(actions_t)
                
                ratio = torch.exp(new_log_probs - log_probs_old_t)
                
                policy_loss_1 = ratio * advantages
                policy_loss_2 = torch.clamp(ratio, 1 - current_clip_epsilon, 1 + current_clip_epsilon) * advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                if use_lstm:
                    values_pred = value_net(states_t.unsqueeze(1))
                else:
                    values_pred = value_net(states_t)
                value_loss = torch.mean(torch.clamp((values_pred - returns)**2, max=1.0))
                
                entropy = safe_entropy(masked_logits, batch_action_masks)
                if entropy is None:
                    entropy = torch.tensor(0.0, device=device)
                else:
                    entropy = entropy.mean()
                
                loss = policy_loss + value_coef * value_loss - current_entropy_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy_net.parameters()) + list(value_net.parameters()), max_grad_norm
                )
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item() if isinstance(entropy, torch.Tensor) else entropy
                
                clip_fraction = ((ratio < (1 - current_clip_epsilon)) | (ratio > (1 + current_clip_epsilon))).float().mean().item()
                clip_fractions.append(clip_fraction)

        if episode % print_every == 0:
            avg_score = sum(all_scores[-print_every:]) / len(all_scores[-print_every:])
            avg_clip_frac = sum(clip_fractions) / len(clip_fractions) if clip_fractions else 0.0
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Score: {score} | Avg({print_every}): {avg_score:.1f} | "
                f"Best: {best_score} | Tile: {best_tile} | "
                f"Policy Loss: {total_policy_loss/ppo_epochs:.4f} | Value Loss: {total_value_loss/ppo_epochs:.4f} | "
                f"Entropy: {total_entropy/ppo_epochs:.4f} | Clip Frac: {avg_clip_frac:.3f}"
            )
        
      
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            eval_scores, eval_best_tile, eval_1024_count, eval_2048_count = evaluate_policy_advanced(
                policy_net, num_episodes=100, device=device, use_lstm=use_lstm
            )
            avg_eval_score = sum(eval_scores) / len(eval_scores)
            best_eval_score = max(eval_scores)
            eval_results.append({
                'episode': episode + 1,
                'avg_score': avg_eval_score,
                'best_score': best_eval_score,
                'best_tile': eval_best_tile,
                '1024_count': eval_1024_count,
                '2048_count': eval_2048_count
            })
            print(
                f"\n=== Evaluation @ Episode {episode + 1} ===\n"
                f"Avg Score (100 ep): {avg_eval_score:.1f}\n"
                f"Best Score: {best_eval_score}\n"
                f"Best Tile: {eval_best_tile}\n"
                f"1024 Count: {eval_1024_count}/100\n"
                f"2048 Count: {eval_2048_count}/100\n"
            )
            
          
            if eval_best_tile >= best_tile:
                torch.save({
                    'policy_state_dict': policy_net.state_dict(),
                    'value_state_dict': value_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode + 1,
                    'best_tile': eval_best_tile,
                    'best_score': best_eval_score,
                }, f"{checkpoint_dir}/best_model_ep{episode+1}.pt")
    
    return policy_net, value_net, training_scores, eval_results


def evaluate_policy(policy_net, num_episodes=50, device="cpu", use_lstm=False):
    policy_net.eval()
    env = Game2048Env()
    scores = []
    best_tile = 0
    
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            if use_lstm:
                policy_net.reset_hidden()
            
            while not env.done:
                state_vec = encode_state(state)
                state_tensor = torch.from_numpy(state_vec).to(device).unsqueeze(0)
                
                if use_lstm:
                    logits = policy_net(state_tensor.unsqueeze(1))
                else:
                    logits = policy_net(state_tensor)
                
              
                valid_actions = get_valid_actions(env, state)
                masked_logits, _ = apply_action_mask(logits, valid_actions, device=device)
                
              
                action = masked_logits.argmax(dim=-1).item()
                state, _, done, info = env.step(action)
            
            score = info["score"]
            scores.append(score)
            episode_max_tile = max(max(row) for row in env.board)
            best_tile = max(best_tile, episode_max_tile)
    
    policy_net.train()
    return scores, best_tile


def evaluate_policy_advanced(policy_net, num_episodes=100, device="cpu", use_lstm=False):
    policy_net.eval()
    env = Game2048Env()
    scores = []
    best_tile = 0
    count_1024 = 0
    count_2048 = 0
    steps_to_1024_list = []
    survival_after_1024_list = []
    
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            h_t, c_t = None, None
            reached_1024 = False
            steps_to_1024 = -1
            steps_after_1024 = 0
            episode_steps = 0
            if use_lstm:
                if hasattr(policy_net, 'reset_hidden') and callable(getattr(policy_net, 'reset_hidden', None)):
                  
                    policy_net.reset_hidden(batch_size=1, device=device)
                    h_t, c_t = policy_net.hidden
                else:
                  
                    policy_net.reset_hidden()
            
            while not env.done:
                episode_steps += 1
                current_max_tile = max(max(row) for row in env.board)
                
      
                if not reached_1024 and current_max_tile >= 1024:
                    reached_1024 = True
                    steps_to_1024 = episode_steps
                
      
                if reached_1024:
                    steps_after_1024 += 1
                state_vec = encode_state(state)
                state_tensor = torch.from_numpy(state_vec).to(device).unsqueeze(0)
                
                if use_lstm:
                    if h_t is not None and c_t is not None:
                        logits = policy_net(state_tensor, (h_t, c_t))
                        h_t, c_t = policy_net.hidden
                    else:
                        logits = policy_net(state_tensor)
                else:
                    logits = policy_net(state_tensor)
                
                valid_actions = get_valid_actions(env, state)
                masked_logits, _ = apply_action_mask(logits, valid_actions, device=device)
                
      
      
                temperature = 0.9
                top_k = 3
                
      
                if len(valid_actions) >= top_k:
      
                    valid_logits = masked_logits[0, valid_actions] if isinstance(valid_actions, list) else masked_logits[0]
                    if isinstance(valid_actions, list):
      
                        valid_logits_list = [masked_logits[0, a].item() for a in valid_actions]
                        top_k_values, top_k_indices = torch.topk(torch.tensor(valid_logits_list, device=device), k=min(top_k, len(valid_actions)))
                      
                        probs = torch.softmax(top_k_values / temperature, dim=-1)
                        selected_idx = torch.multinomial(probs, 1).item()
      
                        action = valid_actions[top_k_indices[selected_idx].item()]
                    else:
                      
                        probs = torch.softmax(masked_logits[0] / temperature, dim=-1)
                        action = torch.multinomial(probs, 1).item()
                else:
                  
                    probs = torch.softmax(masked_logits[0] / temperature, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                
                state, _, done, info = env.step(action)
            
            score = info["score"]
            scores.append(score)
            episode_max_tile = max(max(row) for row in env.board)
            best_tile = max(best_tile, episode_max_tile)
            
            if episode_max_tile >= 1024:
                count_1024 += 1
                if steps_to_1024 > 0:
                    steps_to_1024_list.append(steps_to_1024)
                if steps_after_1024 > 0:
                    survival_after_1024_list.append(steps_after_1024)
            if episode_max_tile >= 2048:
                count_2048 += 1
    
    policy_net.train()
    
  
    avg_steps_to_1024 = sum(steps_to_1024_list) / len(steps_to_1024_list) if steps_to_1024_list else -1
    avg_survival_after_1024 = sum(survival_after_1024_list) / len(survival_after_1024_list) if survival_after_1024_list else -1
    
    if len(steps_to_1024_list) > 0:
        print(f"  📊 Evaluation Metrics:")
        print(f"     Steps to 1024: avg={avg_steps_to_1024:.1f}, min={min(steps_to_1024_list)}, max={max(steps_to_1024_list)}")
    if len(survival_after_1024_list) > 0:
        print(f"     Survival after 1024: avg={avg_survival_after_1024:.1f}, min={min(survival_after_1024_list)}, max={max(survival_after_1024_list)}")
    
    return scores, best_tile, count_1024, count_2048


def plot_average_score(scores, window=100, smoothing_factor=0.9, save_path=None):
      
    if len(scores) < window:
        return
    
    smoothed = []
    running_avg = scores[0]
    for score in scores:
        running_avg = smoothing_factor * running_avg + (1 - smoothing_factor) * score
        smoothed.append(running_avg)
    
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed, label='Exponentially Smoothed')
    plt.plot(scores, alpha=0.2, label='Raw')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Average Score Over Time')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_evaluation_scores(scores, window=10, smoothing_factor=0.9, save_path=None):
  
    plt.figure(figsize=(10, 6))
    plt.plot(scores, marker='o', label='Evaluation Score')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Score')
    plt.title('Evaluation Scores')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def train_reinforce_with_baseline(
    num_episodes=10000,
    gamma=0.99,
    lr=1e-4,
    print_every=100,
    eval_every=1000,
    device="cpu",
    entropy_coef=0.01,
    value_coef=0.5,
    max_steps_per_episode=3000,
    max_grad_norm=0.5,
    checkpoint_dir="checkpoints_reinforce",
    curriculum_stages=[
        (0, 3000, 128),    
        (3000, 7000, 256), 
        (7000, None, None),
    ],
):
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
      
    def get_curriculum_params(episode):
        for stage_start, stage_end, max_spawn in curriculum_stages:
            if stage_end is None:
                if episode >= stage_start:
                    return max_spawn
            else:
                if stage_start <= episode < stage_end:
                    return max_spawn
        return None
    
  
    policy_net = PolicyNet256(input_dim=20).to(device)
    value_net = ValueNet256(input_dim=20).to(device)
    optimizer = optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()), lr=lr
    )
    
    all_scores = []
    best_score = 0
    best_tile = 0
    training_scores = []
    eval_results = []
    
    for episode in range(num_episodes):
      
        max_spawn = get_curriculum_params(episode)
        env = Game2048Env(max_spawn_tile=max_spawn)
        
        state = env.reset()
        states = []
        actions = []
        action_masks = []
        rewards = []
        log_probs = []
        values = []
        steps = 0
        episode_max_tile = 0
        
        while not env.done and steps < max_steps_per_episode:
            current_max_tile = max(max(row) for row in env.board)
            episode_max_tile = max(episode_max_tile, current_max_tile)
            
          
            action, log_prob, value, entropy, action_mask = select_action_with_value(
                policy_net, value_net, state, env=env, device=device, use_masking=True
            )
            
            next_state, reward_raw, done, info = env.step(action)
            
      
      
      
      
            merge_scale = 0.1
            merge_value = reward_raw / merge_scale if reward_raw > 0 else 0
      
            reward = (merge_value / 2048.0) ** 1.5 if merge_value > 0 else 0.0
            
          
            empty_count = sum(1 for i in range(4) for j in range(4) if env.board[i][j] == 0)
            empty_cells = empty_count / 16.0
            reward = reward + empty_cells * 0.01
            
          
            if done:
                reward -= 1.0
            
            if current_max_tile >= 1024:
                reward += 1.0
            
            states.append(state)
            actions.append(action)
            action_masks.append(action_mask)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            state = next_state
            steps += 1
        
        if len(rewards) == 0:
            continue
        
        score = info["score"]
        all_scores.append(score)
        best_score = max(best_score, score)
        best_tile = max(best_tile, episode_max_tile)
        training_scores.append(score)
        returns = compute_returns(rewards, gamma=gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        states_t = torch.stack([torch.from_numpy(encode_state(s)).float().to(device) for s in states])
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        
      
        advantages = returns - values_t.detach()
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            else:
                advantages = advantages - advantages.mean()
        else:
      
            advantages = advantages - advantages.mean()
        policy_loss = -(log_probs_t * advantages.detach()).mean()
        
      
        value_loss = torch.mean(torch.clamp((values_t - returns)**2, max=1.0))
        
      
        entropy_vals = []
        for i, s in enumerate(states):
            state_vec = encode_state(s)
            state_tensor = torch.from_numpy(state_vec).to(device).unsqueeze(0)
            logits = policy_net(state_tensor)
            logits = torch.clamp(logits, -20, 20)
            
            if action_masks[i] is not None:
                masked_logits = logits.clone()
                if action_masks[i].dim() == 1:
                    masked_logits[0, ~action_masks[i]] = -20.0
                else:
                    masked_logits[0, ~action_masks[i][0]] = -20.0
            else:
                masked_logits = logits
            
            entropy = safe_entropy(masked_logits)
            if entropy is not None:
                entropy_vals.append(entropy.item())
        
        avg_entropy = sum(entropy_vals) / len(entropy_vals) if entropy_vals else 0.0
        
      
        loss = policy_loss + value_coef * value_loss - entropy_coef * avg_entropy
        
      
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy_net.parameters()) + list(value_net.parameters()), max_grad_norm
        )
        optimizer.step()
        
      
        illegal_count = 0
        total_actions = len(actions)
        for i, action in enumerate(actions):
            if action_masks[i] is not None:
                if action_masks[i].dim() == 1:
                    if not action_masks[i][action]:
                        illegal_count += 1
                else:
                    if not action_masks[i][0, action]:
                        illegal_count += 1
        illegal_pct = (illegal_count / total_actions * 100) if total_actions > 0 else 0.0
        
        if episode % print_every == 0:
            avg_score = sum(all_scores[-print_every:]) / len(all_scores[-print_every:])
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Score: {score} | Avg({print_every}): {avg_score:.1f} | "
                f"Best: {best_score} | Tile: {best_tile} | "
                f"Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f} | "
                f"Entropy: {avg_entropy:.4f} | Illegal: {illegal_pct:.1f}% | "
                f"Max Spawn: {max_spawn if max_spawn else 'Normal'}"
            )
        
      
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            eval_scores, eval_best_tile, eval_1024_count, eval_2048_count = evaluate_policy_advanced(
                policy_net, num_episodes=100, device=device, use_lstm=False
            )
            avg_eval_score = sum(eval_scores) / len(eval_scores)
            best_eval_score = max(eval_scores)
            eval_results.append({
                'episode': episode + 1,
                'avg_score': avg_eval_score,
                'best_score': best_eval_score,
                'best_tile': eval_best_tile,
                '1024_count': eval_1024_count,
                '2048_count': eval_2048_count
            })
            print(
                f"\n=== Evaluation @ Episode {episode + 1} ===\n"
                f"Avg Score (100 ep): {avg_eval_score:.1f}\n"
                f"Best Score: {best_eval_score}\n"
                f"Best Tile: {eval_best_tile}\n"
                f"1024 Count: {eval_1024_count}/100\n"
                f"2048 Count: {eval_2048_count}/100\n"
            )
            
          
            if eval_best_tile >= best_tile:
                torch.save({
                    'policy_state_dict': policy_net.state_dict(),
                    'value_state_dict': value_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode + 1,
                    'best_tile': eval_best_tile,
                    'best_score': best_eval_score,
                }, f"{checkpoint_dir}/best_model_ep{episode+1}.pt")
    
    return policy_net, value_net, training_scores, eval_results


def train_reinforce_with_baseline_lstm(
    num_episodes=10000,
    gamma=0.99,
    lr=1e-4,
    print_every=100,
    eval_every=1000,
    device="cpu",
    entropy_coef=0.01,
    value_coef=0.5,
    max_steps_per_episode=3000,
    max_grad_norm=1.0,
    checkpoint_dir="checkpoints_reinforce_lstm",
    truncated_bptt=20,
    curriculum_stages=[
        (0, 3000, 128),    
        (3000, 7000, 256), 
        (7000, None, None),
    ],
):
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    def get_curriculum_params(episode):
        for stage_start, stage_end, max_spawn in curriculum_stages:
            if stage_end is None:
                if episode >= stage_start:
                    return max_spawn
            else:
                if stage_start <= episode < stage_end:
                    return max_spawn
        return None
    
    policy_net = PolicyNetLSTM128(input_dim=20, hidden_dim=128, lstm_hidden=128).to(device)
    value_net = ValueNet256(input_dim=20).to(device)
    
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
    
    all_scores = []
    best_score = 0
    best_tile = 0
    training_scores = []
    eval_results = []
    training_metrics = []
    
    for episode in range(num_episodes):
        max_spawn = get_curriculum_params(episode)
        env = Game2048Env(max_spawn_tile=max_spawn)
        
        if episode < 3000:
            stage = 1
        elif episode < 7000:
            stage = 2
        else:
            stage = 3
        
        current_entropy_coef = entropy_coef * (0.3 + 0.7 * math.exp(-episode / 4000))
        
        policy_net.reset_hidden(batch_size=1, device=device)
        h_t, c_t = policy_net.hidden
        
        state = env.reset()
        states = []
        actions = []
        action_masks = []
        rewards = []
        log_probs = []
        values = []
        steps = 0
        episode_max_tile = 0
        reached_512_this_episode = False
        reached_1024_this_episode = False
        reached_2048_this_episode = False
        
        bptt_states = []
        bptt_actions = []
        bptt_action_masks = []
        bptt_rewards = []
        bptt_log_probs = []
        bptt_values = []
        bptt_hidden_states = []
        
        if episode == 0:
            print("Episode 0: Starting...", end="", flush=True)
        
        while not env.done and steps < max_steps_per_episode:
            current_max_tile = max(max(row) for row in env.board)
            prev_max_tile = episode_max_tile
            episode_max_tile = max(episode_max_tile, current_max_tile)
            
            if current_max_tile >= 512 and prev_max_tile < 512:
                reached_512_this_episode = True
            if current_max_tile >= 1024 and prev_max_tile < 1024:
                reached_1024_this_episode = True
            if current_max_tile >= 2048 and prev_max_tile < 2048:
                reached_2048_this_episode = True
            
            state_vec = encode_state(state)
            state_tensor = torch.from_numpy(state_vec).float().to(device).unsqueeze(0)
            
            if state_tensor.dim() != 2 or state_tensor.shape != (1, 20):
                state_tensor = state_tensor.view(1, -1)
                if state_tensor.shape[1] != 20:
                    raise ValueError(f"Invalid state tensor shape: {state_tensor.shape}, expected (1, 20)")
            
            logits = policy_net(state_tensor, (h_t, c_t))
            logits = torch.clamp(logits, -20, 20)
            
            valid_actions = get_valid_actions(env, state)
            masked_logits, action_mask = apply_action_mask(logits, valid_actions, device=device)
            
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            value = value_net(state_tensor)
            
            entropy = safe_entropy(masked_logits, action_mask)
            entropy_val = entropy.item() if entropy is not None else 0.0
            
            next_state, reward_raw, done, info = env.step(action.item())
            
            merge_scale = 0.1
            merge_value = reward_raw / merge_scale if reward_raw > 0 else 0
            if merge_value > 0:
                tile_reward = math.log2(merge_value)
                reward = tile_reward / math.log2(2048)
            else:
                reward = 0.0
            
            empty_count = sum(1 for i in range(4) for j in range(4) if env.board[i][j] == 0)
            empty_ratio = empty_count / 16.0
            if current_max_tile < 256:
                reward += empty_ratio * 0.01
            elif current_max_tile < 1024:
                reward += empty_ratio * 0.005
            else:
                reward += empty_ratio * 0.002
            
            state_vec = encode_state(env.board)
            monotonicity_score = state_vec[19]
            reward += 0.02 * monotonicity_score
            
            if done:
                episode_length_ratio = (steps + 1) / max_steps_per_episode
                reward -= 1.0 + 0.1 * episode_length_ratio
            
            if stage >= 2 and reached_512_this_episode and current_max_tile >= 512:
                reward += 0.2
            
            if stage == 3 and reached_1024_this_episode and current_max_tile >= 1024:
                reward += 1.0
            
            if stage == 3 and reached_2048_this_episode and current_max_tile >= 2048:
                reward += 5.0
            
            bptt_states.append(state)
            bptt_actions.append(action.item())
            bptt_action_masks.append(action_mask)
            bptt_rewards.append(reward)
            bptt_log_probs.append(log_prob)
            bptt_values.append(value)
            bptt_hidden_states.append((h_t.detach().clone(), c_t.detach().clone()))
            
      
            
            if episode == 0 and steps % 50 == 0:
                print(".", end="", flush=True)
            
            if len(bptt_states) >= truncated_bptt or done:
                segment_rewards = bptt_rewards
                segment_returns = compute_returns(segment_rewards, gamma=gamma)
                segment_returns = torch.tensor(segment_returns, dtype=torch.float32, device=device)
                
                if segment_returns.numel() > 1:
                    returns_std = segment_returns.std()
                    if returns_std > 1e-8:
                        segment_returns = (segment_returns - segment_returns.mean()) / (returns_std + 1e-8)
                    else:
                        segment_returns = segment_returns - segment_returns.mean()
                else:
                    segment_returns = segment_returns - segment_returns.mean()
                
                segment_states_t = torch.stack([
                    torch.from_numpy(encode_state(s)).float().to(device) for s in bptt_states
                ])
                segment_actions_t = torch.tensor(bptt_actions, dtype=torch.long, device=device)
                segment_log_probs_t = torch.stack(bptt_log_probs)
                segment_values_t = torch.stack(bptt_values)
                
                advantages = segment_returns - segment_values_t.detach()
                
                if advantages.numel() > 1:
                    adv_std = advantages.std()
                    if adv_std > 1e-8:
                        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                    else:
                        advantages = advantages - advantages.mean()
                else:
                    advantages = advantages - advantages.mean()
                
                advantages = torch.clamp(advantages, -5.0, 5.0)
                
                segment_log_probs_new = []
                segment_entropies = []
                hidden_norms = []
                
                for i, s in enumerate(bptt_states):
                    state_vec = encode_state(s)
                    state_tensor = torch.from_numpy(state_vec).to(device).unsqueeze(0).unsqueeze(1)
                    
                    h_prev, c_prev = bptt_hidden_states[i]
      
                    if h_prev.dim() == 1:
                        h_prev = h_prev.unsqueeze(0)
                    if c_prev.dim() == 1:
                        c_prev = c_prev.unsqueeze(0)
                    logits = policy_net(state_tensor, (h_prev, c_prev))
                    logits = torch.clamp(logits, -20, 20)
                    
                    if bptt_action_masks[i] is not None:
                        masked_logits = logits.clone()
                        if bptt_action_masks[i].dim() == 1:
                            masked_logits[0, ~bptt_action_masks[i]] = -20.0
                        else:
                            masked_logits[0, ~bptt_action_masks[i][0]] = -20.0
                    else:
                        masked_logits = logits
                    
                    dist = Categorical(logits=masked_logits)
                    log_prob_new = dist.log_prob(segment_actions_t[i])
                    segment_log_probs_new.append(log_prob_new)
                    
                    entropy_seg = safe_entropy(masked_logits, bptt_action_masks[i])
                    if entropy_seg is not None:
                        segment_entropies.append(entropy_seg.item())
                    
                    h_current, c_current = policy_net.hidden
                    hidden_norm = (h_current.norm().item() + c_current.norm().item()) / 2.0
                    hidden_norms.append(hidden_norm)
                
                segment_log_probs_new_t = torch.stack(segment_log_probs_new)
                
                policy_loss = -(segment_log_probs_new_t * advantages.detach()).mean()
                value_loss = torch.mean(torch.clamp((segment_values_t.squeeze(-1) - segment_returns)**2, max=1.0))
                
                avg_entropy = sum(segment_entropies) / len(segment_entropies) if segment_entropies else 0.0
                entropy_bonus = torch.tensor(avg_entropy, device=device)
                
                loss = policy_loss + value_coef * value_loss - current_entropy_coef * entropy_bonus
                
                policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                policy_optimizer.step()
                
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                value_optimizer.step()
                
                h_t = h_t.detach()
                c_t = c_t.detach()
                
      
                policy_net.hidden = (h_t, c_t)
                
                avg_hidden_norm = sum(hidden_norms) / len(hidden_norms) if hidden_norms else 0.0
                training_metrics.append({
                    'episode': episode,
                    'step': steps,
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': avg_entropy,
                    'hidden_norm': avg_hidden_norm,
                })
                
                bptt_states = []
                bptt_actions = []
                bptt_action_masks = []
                bptt_rewards = []
                bptt_log_probs = []
                bptt_values = []
                bptt_hidden_states = []
            
            state = next_state
            steps += 1
        
        if episode == 0:
            print(f"Done! (Steps: {steps})")
        
        if steps == 0:
            continue
        
        score = info["score"]
        all_scores.append(score)
        best_score = max(best_score, score)
        best_tile = max(best_tile, episode_max_tile)
        training_scores.append(score)
        
        illegal_count = 0
        total_actions = len(bptt_actions) if bptt_actions else 0
        for i, action_val in enumerate(bptt_actions if bptt_actions else actions):
            if i < len(bptt_action_masks) and bptt_action_masks[i] is not None:
                mask = bptt_action_masks[i]
                if mask.dim() == 1:
                    if not mask[action_val]:
                        illegal_count += 1
                else:
                    if not mask[0, action_val]:
                        illegal_count += 1
        illegal_pct = (illegal_count / total_actions * 100) if total_actions > 0 else 0.0
        
        latest_metrics = training_metrics[-1] if training_metrics else {}
        hidden_norm = latest_metrics.get('hidden_norm', 0.0)
        entropy_val = latest_metrics.get('entropy', 0.0)
        
        if episode == 0 or episode % print_every == 0:
            avg_score = sum(all_scores[-print_every:]) / len(all_scores[-print_every:])
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Score: {score} | Avg({print_every}): {avg_score:.1f} | "
                f"Best: {best_score} | Tile: {best_tile} | "
                f"Policy Loss: {latest_metrics.get('policy_loss', 0.0):.4f} | "
                f"Value Loss: {latest_metrics.get('value_loss', 0.0):.4f} | "
                f"Entropy: {entropy_val:.4f} | Hidden Norm: {hidden_norm:.2f} | "
                f"Illegal: {illegal_pct:.1f}% | Max Spawn: {max_spawn if max_spawn else 'Normal'}"
            )
        
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            eval_scores, eval_best_tile, eval_1024_count, eval_2048_count = evaluate_policy_advanced(
                policy_net, num_episodes=100, device=device, use_lstm=True
            )
            avg_eval_score = sum(eval_scores) / len(eval_scores)
            best_eval_score = max(eval_scores)
            eval_results.append({
                'episode': episode + 1,
                'avg_score': avg_eval_score,
                'best_score': best_eval_score,
                'best_tile': eval_best_tile,
                '1024_count': eval_1024_count,
                '2048_count': eval_2048_count
            })
            print(
                f"\nEvaluation @ Episode {episode + 1}\n"
                f"Avg Score: {avg_eval_score:.1f}\n"
                f"Best Score: {best_eval_score}\n"
                f"Best Tile: {eval_best_tile}\n"
                f"1024 Count: {eval_1024_count}/100\n"
                f"2048 Count: {eval_2048_count}/100\n"
            )
            
            if eval_best_tile >= best_tile:
                torch.save({
                    'policy_state_dict': policy_net.state_dict(),
                    'value_state_dict': value_net.state_dict(),
                    'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                    'value_optimizer_state_dict': value_optimizer.state_dict(),
                    'episode': episode + 1,
                    'best_tile': eval_best_tile,
                    'best_score': best_eval_score,
                }, f"{checkpoint_dir}/best_model_ep{episode+1}.pt")
    
    return policy_net, value_net, training_scores, eval_results, training_metrics


def train_ppo_lstm(
    num_episodes=10000,
    gamma=0.99,
    lam=0.97,
    lr=5e-5,
    ppo_epochs=3,
    clip_epsilon=0.2,
    value_clip_epsilon=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=1.0,
    rollout_len=512,
    seq_len=40,
    minibatch_size=128,
    print_every=100,
    eval_every=1000,
    device="cpu",
    max_steps_per_episode=3000,
    checkpoint_dir="checkpoints_ppo_lstm",
    curriculum_stages=[
        (0, 1000, 128),    
        (1000, 2000, 256), 
        (2000, None, None),
    ],
    seed=None,
):
    # PPO-LSTM with GAE(λ), KL guard, and sequence-aware minibatch sampling

    import os
    import random
    
  
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    def get_curriculum_params(episode):
        for stage_start, stage_end, max_spawn in curriculum_stages:
            if stage_end is None:
                if episode >= stage_start:
                    return max_spawn
            else:
                if stage_start <= episode < stage_end:
                    return max_spawn
        return None
    
  
      
      
    policy_net = PolicyNetLSTM128(input_dim=20, hidden_dim=256, lstm_hidden=256, output_dim=4, num_layers=1).to(device)
    value_net = ValueNet256(input_dim=20, hidden_dim=256).to(device)
    
      
    if device == "cuda":
        seq_len = min(seq_len, 32)
    else:
        seq_len = min(seq_len, 16)
    
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
    
    all_scores = []
    best_score = 0
    best_tile = 0
    training_scores = []
    eval_results = []
    
      
    episodes_with_1024 = set()
      
    episode_buffer_ranges = {}
    
      
    if seed is not None:
        print("Running baseline evaluation (20 episodes, stochastic with temperature=0.8, top-k=2)...")
        baseline_scores = []
        for _ in range(20):
            env = Game2048Env()
            state = env.reset()
            policy_net.reset_hidden(batch_size=1, device=device)
            done = False
            steps = 0
            info = None
            while not done and steps < max_steps_per_episode:
                state_vec = encode_state(state)
                state_tensor = torch.from_numpy(state_vec).float().to(device).unsqueeze(0).unsqueeze(1)
                with torch.no_grad():
                    logits = policy_net(state_tensor)
                    logits = torch.clamp(logits, -20, 20)
                    valid_actions = get_valid_actions(env, state)
                    if valid_actions:
                        masked_logits = logits.clone()
                        mask = torch.zeros(4, dtype=torch.bool, device=device)
                        for a in valid_actions:
                            mask[a] = True
                        masked_logits[0, ~mask] = -20.0
                        
      
                        temperature = 0.8
                        top_k = 2
      
                        if len(valid_actions) >= top_k:
                            valid_logits_list = [masked_logits[0, a].item() for a in valid_actions]
                            top_k_values, top_k_indices = torch.topk(torch.tensor(valid_logits_list, device=device), k=min(top_k, len(valid_actions)))
                          
                            probs = torch.softmax(top_k_values / temperature, dim=-1)
                            selected_idx = torch.multinomial(probs, 1).item()
      
                            action = valid_actions[top_k_indices[selected_idx].item()]
                        else:
                          
                            probs = torch.softmax(masked_logits[0] / temperature, dim=-1)
                            action = torch.multinomial(probs, 1).item()
                    else:
                        action = 0
                    state, reward, done, info = env.step(action)
                steps += 1
            score = info["score"] if info is not None else 0
            baseline_scores.append(score)
        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        print(f"Baseline (stochastic, temp=0.8, top-k=2): Avg Score = {baseline_avg:.1f}, Best = {max(baseline_scores)}")
    
    buffer = RolloutBuffer()
    
    for episode in range(num_episodes):
        max_spawn = get_curriculum_params(episode)
        env = Game2048Env(max_spawn_tile=max_spawn)
        
      
      
        if episode < 4000:
            stage = 1
        elif episode < 8000:
            stage = 2
        else:
            stage = 3
        
      
        buffer_size_before_rollout = buffer.size()
        
      
        seen_512 = False
        seen_1024 = False
        seen_2048 = False
        current_max_tile = 0
        
      
        entropy_boosted = False
        entropy_boost_step = 0
        entropy_boost_duration = 25
        
        state = env.reset()
        policy_net.reset_hidden(batch_size=1, device=device)
        h0_init, c0_init = policy_net.hidden
        h0_norm = h0_init.norm().item() if h0_init is not None else 0.0
        c0_norm = c0_init.norm().item() if c0_init is not None else 0.0
        if episode < 5:
            print(f"Episode {episode} - Hidden norm at reset: h={h0_norm:.4f}, c={c0_norm:.4f}")
        
        buffer.reset()
        done = False
        steps = 0
        episode_info = None
        
      
      
      
      
      
      
        while not done and steps < rollout_len and steps < max_steps_per_episode:
      
            if done:
                episode_info = {"score": env.score} if episode_info is None else episode_info
                break
            
            state_vec = encode_state(state)
            state_tensor = torch.from_numpy(state_vec).float().to(device).unsqueeze(0).unsqueeze(1)
            
      
            h0_step, c0_step = policy_net.hidden
            if h0_step is not None:
                h0_step = h0_step.detach().clone()
                c0_step = c0_step.detach().clone()
            
      
      
            with torch.no_grad():
                logits = policy_net(state_tensor)
      
                logits = torch.clamp(logits, -10.0, 10.0)
                value = value_net(torch.from_numpy(state_vec).float().to(device).unsqueeze(0))
                value = value.detach()
                valid_actions = get_valid_actions(env, state=None)
                action_mask = None
                if not valid_actions:
      
                    if not env._can_move():
                        done = True
                        episode_info = {"score": env.score}
                        break
                    else:
      
                        raise RuntimeError(
                            f"BUG: No valid actions but env._can_move()=True at episode {episode}, step {steps}. "
                            f"Board state may be inconsistent."
                        )
      
                NEG_INF = -20.0
                masked_logits = logits.clone()
                
      
                mask = torch.zeros(4, dtype=torch.bool, device=device)
                for a in valid_actions:
                    mask[a] = True
      
                if not mask.any():
                    raise RuntimeError(
                        f"BUG: All actions illegal at episode {episode}, step {steps}. "
                        f"This indicates a mask/env synchronization bug. "
                        f"valid_actions={valid_actions}, env._can_move()={env._can_move()}"
                    )
                
      
                masked_logits[0, ~mask] = masked_logits[0, ~mask] + NEG_INF
                
                action_mask = mask
                
      
      
                masked_logits = torch.clamp(masked_logits, -10.0, 10.0)
                
      
      
                try:
                    dist = Categorical(logits=masked_logits, validate_args=False)
                except Exception as e:
                    print(f"Error creating Categorical at episode {episode}, step {steps}: {e}")
                    print(f"  masked_logits shape: {masked_logits.shape}")
                    print(f"  masked_logits: {masked_logits}")
                    print(f"  valid_actions: {valid_actions}")
                    done = True
                    episode_info = {"score": env.score}
                    break
                
      
      
                try:
                    action = dist.sample()
                    logp_old = dist.log_prob(action)
                    logp_old = logp_old.detach()
                except (RuntimeError, ValueError) as e:
                    print(f"Error sampling action at episode {episode}, step {steps}: {e}")
                    print(f"  probs: {probs}")
      
                    action = masked_logits.argmax(dim=-1)
                    logp_old = dist.log_prob(action)
                    logp_old = logp_old.detach()
                except Exception as e:
                    print(f"Unexpected error at episode {episode}, step {steps}: {e}")
                    print(f"  probs: {probs}")
                    done = True
                    episode_info = {"score": env.score}
                    break
            
          
            next_state, reward_raw, done, info = env.step(action.item())
            episode_info = info
            
      
            if action_mask is not None:
                if not action_mask[action.item()]:
                    reward_raw -= 0.01
            
      
      
      
            merge_scale = 0.1
            if reward_raw > 0:
      
      
                merge_value = reward_raw / merge_scale
            else:
                merge_value = 0
            
      
            old_max_tile = current_max_tile
            current_max_tile = max(current_max_tile, max(max(row) for row in next_state))
            
      
      
            import math
            if merge_value > 0:
                merge_r = 0.05 * math.log2(merge_value)
            else:
                merge_r = 0.0
            
      
      
            empty_count = sum(1 for i in range(4) for j in range(4) if env.board[i][j] == 0)
            empty_ratio = empty_count / 16.0
      
            empty_r = 0.01 * math.log(1 + 16 * empty_ratio) / math.log(17)
            
      
      
            if current_max_tile >= 1024:
                empty_r += 0.02 * empty_ratio
            
      
            state_vec_next = encode_state(next_state)
            monotonicity_score = state_vec_next[19]
            mono_r = 0.005 * monotonicity_score
            
      
      
            if (not seen_512) and current_max_tile >= 512:
                seen_512 = True
            
            if (not seen_1024) and current_max_tile >= 1024:
                seen_1024 = True
      
                entropy_boosted = True
                entropy_boost_step = steps
            
            if (not seen_2048) and current_max_tile >= 2048:
                seen_2048 = True
                done = True
            
            if done and not seen_2048:
                done_penalty = -0.5
            else:
                done_penalty = 0.0
            
      
      
            base_reward = merge_r + empty_r + mono_r + done_penalty
            
      
            milestone_2048_reward = 0.0
            if seen_2048 and current_max_tile >= 2048:
                milestone_2048_reward = 2.0
            
      
            other_milestones = 0.0
            if seen_512 and current_max_tile >= 512:
                other_milestones += 0.15
            if seen_1024 and current_max_tile >= 1024:
                other_milestones += 0.5
            
      
            length_bonus = 0.0
            if current_max_tile >= 1024:
                length_bonus = 0.001 * steps
            
          
            base_reward = max(-2.0, min(2.0, base_reward + other_milestones + length_bonus))
            
      
            reward = base_reward + milestone_2048_reward
            
          
            buffer.add(
                obs=state,
                action=action.item(),
                logp_old=logp_old.item(),
                value_old=value.item(),  
                reward=reward,
                done=done,
                mask=action_mask,
                h0=h0_step,
                c0=c0_step 
            )
            
            state = next_state
            steps += 1
            prev_max_tile = current_max_tile
        
      
        rollout_max_tile = current_max_tile
        rollout_entropy_boosted = entropy_boosted
        rollout_entropy_boost_step = entropy_boost_step
        rollout_entropy_boost_duration = entropy_boost_duration
        rollout_seen_1024 = seen_1024
      
      
      
      
        
      
        if buffer.size() == 0:
      
            score = episode_info["score"] if episode_info is not None else 0
            all_scores.append(score)
            best_score = max(best_score, score)
            best_tile = max(best_tile, current_max_tile)
            training_scores.append(score)
            if done:
                env.reset()
                policy_net.reset_hidden(batch_size=1, device=device)
            continue
        
      
        buffer_size_after_rollout = buffer.size()
        
      
        if buffer_size_after_rollout > buffer_size_before_rollout:
      
            episode_buffer_ranges[episode] = (buffer_size_before_rollout, buffer_size_after_rollout)
        buffer_data = buffer.get_all()
        rewards = buffer_data['rewards']
        values = buffer_data['value_old']
        dones = buffer_data['dones']
        
      
      
        if done:
      
            next_value = 0.0
            score = episode_info["score"] if episode_info is not None else 0
        else:
      
            state_vec_final = encode_state(state)
            with torch.no_grad():
                next_value = value_net(torch.from_numpy(state_vec_final).float().to(device).unsqueeze(0)).item()
      
            score = episode_info["score"] if episode_info is not None else 0
        
      
      
        next_values = []
        for t in range(len(rewards)):
            if t == len(rewards) - 1:
      
                next_values.append(next_value)
            else:
      
                if dones[t]:
                    next_values.append(0.0)
                else:
                    next_values.append(values[t+1])
        values_t = torch.tensor(values, dtype=torch.float32, device=device)
        next_values_t = torch.tensor(next_values, dtype=torch.float32, device=device)
        advantages, returns = compute_gae(rewards, values_t, next_values_t, gamma=gamma, lam=lam, device=device)
        advantages = advantages.detach()
        returns = returns.detach()
        if advantages.numel() > 1:
      
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                print(f"Warning: NaN/Inf in advantages at episode {episode}, skipping normalization")
                advantages = advantages - advantages.mean()
            else:
      
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                else:
                    advantages = advantages - adv_mean
        else:
            advantages = advantages - advantages.mean()
        
      
        advantages = torch.clamp(advantages, -10.0, 10.0)
        obs_t = [torch.from_numpy(encode_state(o)).float().to(device) for o in buffer_data['obs']]
        actions_t = torch.tensor(buffer_data['actions'], dtype=torch.long, device=device)
        logp_old_t = torch.tensor(buffer_data['logp_old'], dtype=torch.float32, device=device)
        value_old_t = torch.tensor(buffer_data['value_old'], dtype=torch.float32, device=device)
        returns_t = returns.to(device)
        advantages_t = advantages.to(device)
      
        returns_t = returns_t.detach()
        advantages_t = advantages_t.detach()
        
      
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0
        
      
      
      
      
        base_ent = 4e-4
        
      
      
        episodes_since_burst = episode % 500
        if episodes_since_burst < 20:
      
            entropy_coef_base = base_ent * 2.0
        else:
            entropy_coef_base = base_ent
      
        avg_empty_count = 0.0
        max_tile_in_rollout = 0
        
      
        if len(buffer_data['obs']) > 0:
            for obs in buffer_data['obs']:
                empty_count = sum(1 for i in range(4) for j in range(4) if obs[i][j] == 0)
                avg_empty_count += empty_count
                max_tile_in_rollout = max(max_tile_in_rollout, max(max(row) for row in obs))
            avg_empty_count = avg_empty_count / len(buffer_data['obs'])
        
      
        rollout_max = max_tile_in_rollout
        
      
        if rollout_max >= 512 and rollout_max < 1024:
      
            entropy_coef_base = base_ent * 1.8
        
      
      
        rollout_entropy_boosted = rollout_entropy_boosted if 'rollout_entropy_boosted' in locals() else False
        rollout_entropy_boost_step = rollout_entropy_boost_step if 'rollout_entropy_boost_step' in locals() else -1
        rollout_entropy_boost_duration = rollout_entropy_boost_duration if 'rollout_entropy_boost_duration' in locals() else 25
        
      
      
        if rollout_entropy_boosted and rollout_entropy_boost_step >= 0 and rollout_max >= 1024:
      
            if len(buffer_data['obs']) > rollout_entropy_boost_duration:
      
                pass
            else:
      
                entropy_coef_base = base_ent * 2.0
        
      
      
        if max_tile_in_rollout >= 512 and avg_empty_count >= 6:
            entropy_coef_base = max(entropy_coef_base, base_ent * 2.0)
      
        elif max_tile_in_rollout >= 1024 and avg_empty_count >= 8:
            entropy_coef_base = max(entropy_coef_base, base_ent * 2.5)
        
      
        current_ent_coef = max(entropy_coef_base, base_ent)
        
        for epoch in range(ppo_epochs):
      
            T = len(obs_t)
            num_sequences = (T + seq_len - 1) // seq_len
            
      
            sequences_from_1024_episodes = []
            sequences_from_normal_episodes = []
            
            for seq_idx in range(num_sequences):
                start_idx = seq_idx * seq_len
      
                seq_episode = None
                for ep_idx, (ep_start, ep_end) in episode_buffer_ranges.items():
                    if ep_start <= start_idx < ep_end:
                        seq_episode = ep_idx
                        break
                
                if seq_episode is not None and seq_episode in episodes_with_1024:
                    sequences_from_1024_episodes.append(seq_idx)
                else:
                    sequences_from_normal_episodes.append(seq_idx)
            
      
            num_1024_samples = max(1, int(num_sequences * 0.3))
            num_normal_samples = num_sequences - num_1024_samples
            
          
            if len(sequences_from_1024_episodes) > 0:
                sampled_1024 = random.sample(sequences_from_1024_episodes, min(num_1024_samples, len(sequences_from_1024_episodes)))
            else:
                sampled_1024 = []
            
            if len(sequences_from_normal_episodes) > 0:
                sampled_normal = random.sample(sequences_from_normal_episodes, min(num_normal_samples, len(sequences_from_normal_episodes)))
            else:
                sampled_normal = sequences_from_normal_episodes[:num_normal_samples] if len(sequences_from_normal_episodes) > 0 else []
            
          
            sequence_indices = sampled_1024 + sampled_normal
            random.shuffle(sequence_indices)
            
          
            for mb_start in range(0, len(sequence_indices), minibatch_size):
                    mb_indices = sequence_indices[mb_start:mb_start + minibatch_size]
                    
                    mb_obs = []
                    mb_actions = []
                    mb_logp_old = []
                    mb_value_old = []
                    mb_returns = []
                    mb_advantages = []
                    mb_h0 = []
                    mb_c0 = []
                    mb_masks = []
                    
                    for seq_idx in mb_indices:
                        start_idx = seq_idx * seq_len
                        end_idx = min(start_idx + seq_len, T)
                        
                        mb_obs.extend(obs_t[start_idx:end_idx])
                        mb_actions.extend(actions_t[start_idx:end_idx])
                        mb_logp_old.extend(logp_old_t[start_idx:end_idx])
                        mb_value_old.extend(value_old_t[start_idx:end_idx])
                        mb_returns.extend(returns_t[start_idx:end_idx])
                        mb_advantages.extend(advantages_t[start_idx:end_idx])
                        mb_masks.extend(buffer_data['masks'][start_idx:end_idx])
                        
      
                        h0_seq = buffer_data['h0'][start_idx]
                        c0_seq = buffer_data['c0'][start_idx]
                        if h0_seq is not None:
      
                            mb_h0.append(h0_seq.detach() if isinstance(h0_seq, torch.Tensor) else h0_seq)
                            mb_c0.append(c0_seq.detach() if isinstance(c0_seq, torch.Tensor) else c0_seq)
                        else:
                            mb_h0.append(torch.zeros(1, 128, device=device))
                            mb_c0.append(torch.zeros(1, 128, device=device))
                    mb_obs_t = torch.stack(mb_obs)
                    mb_actions_t = torch.stack(mb_actions)
                    mb_logp_old_t = torch.stack(mb_logp_old)
                    mb_value_old_t = torch.stack(mb_value_old)
                    mb_returns_t = torch.stack(mb_returns)
                    mb_advantages_t = torch.stack(mb_advantages)
                    
      
                    mb_obs_seq = mb_obs_t.unsqueeze(1)
                    
                  
                    logp_new_list = []
                    value_new_list = []
                    entropy_list = []
                    
                    mb_offset = 0
                    for mb_seq_idx, seq_idx in enumerate(mb_indices):
                        start_idx = seq_idx * seq_len
                        end_idx = min(start_idx + seq_len, T)
                        seq_len_actual = end_idx - start_idx
                        
                        h0_seq = mb_h0[mb_seq_idx]
                        c0_seq = mb_c0[mb_seq_idx]
                        if h0_seq.dim() == 1:
                            h0_seq = h0_seq.unsqueeze(0)
                        if c0_seq.dim() == 1:
                            c0_seq = c0_seq.unsqueeze(0)
      
                        h0_seq = h0_seq.detach()
                        c0_seq = c0_seq.detach()
                        
                        seq_obs = mb_obs_seq[mb_offset:mb_offset + seq_len_actual]
                        seq_actions = mb_actions_t[mb_offset:mb_offset + seq_len_actual]
                        seq_masks = mb_masks[mb_offset:mb_offset + seq_len_actual]
                        
                      
      
      
                        policy_net.hidden = (h0_seq, c0_seq)
                        seq_logits = []
                        
                        for t in range(seq_len_actual):
                            obs_t_step = seq_obs[t:t+1]
                            logits_t = policy_net(obs_t_step)
                            logits_t = torch.clamp(logits_t, -10.0, 10.0)
                            
                          
                            if seq_masks[t] is not None:
                                masked_logits = logits_t.clone()
                                mask = seq_masks[t]
                                if not isinstance(mask, torch.Tensor):
                                    if isinstance(mask, (list, tuple, np.ndarray)):
                                        mask = torch.tensor(mask, dtype=torch.bool, device=device)
                                    else:
                                        continue
                                if mask.dim() == 0:
                                    mask = mask.unsqueeze(0)
                                if mask.numel() == 4:
                                    NEG_INF = -20.0
                                    if mask.any():
                                        masked_logits[0, ~mask] = masked_logits[0, ~mask] + NEG_INF
      
      
                                        masked_logits = torch.clamp(masked_logits, -10.0, 10.0)
                                    else:
      
                                        masked_logits = logits_t.clone()
                                logits_t = masked_logits
                            
                            seq_logits.append(logits_t)
                        
                        seq_logits_t = torch.cat(seq_logits, dim=0)
                        dist = Categorical(logits=seq_logits_t)
                        logp_new_seq = dist.log_prob(seq_actions)
                        entropy_seq = dist.entropy()
                        
                        logp_new_list.append(logp_new_seq)
                        entropy_list.append(entropy_seq)
                        
                      
                        seq_obs_flat = mb_obs_t[mb_offset:mb_offset + seq_len_actual]
                        value_new_seq = value_net(seq_obs_flat)
                        value_new_list.append(value_new_seq)
                        
                        mb_offset += seq_len_actual
                    
                    logp_new_t = torch.cat(logp_new_list, dim=0)
                    value_new_t = torch.cat(value_new_list, dim=0)
                    entropy_t = torch.cat(entropy_list, dim=0)
                    
                  
                    ratio = torch.exp(logp_new_t - mb_logp_old_t)
                    surr1 = ratio * mb_advantages_t
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages_t
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
      
      
                    value_pred = value_new_t
                    value_loss = torch.mean(torch.clamp((value_pred - mb_returns_t)**2, max=1.0))
                    
                  
                    entropy = entropy_t.mean()
                    
                  
                    kl = (mb_logp_old_t - logp_new_t).mean()
                    
      
                    loss = policy_loss + vf_coef * value_loss - current_ent_coef * entropy
                    
                  
                    policy_optimizer.zero_grad()
                    policy_loss.backward(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                    policy_optimizer.step()
                    
                    value_optimizer.zero_grad()
                    value_loss.backward(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                    value_optimizer.step()
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    total_kl += kl.item()
                    num_updates += 1
                    
      
      
                    if episode < 3000:
                        kl_threshold = 0.02
                    else:
                        kl_threshold = float('inf')
                    
                    if kl > kl_threshold:
                        break
            
      
      
            if episode < 3000:
                kl_threshold = 0.02
            else:
                kl_threshold = float('inf')
            
            if num_updates > 0 and total_kl / num_updates > kl_threshold:
                break
        
      
      
        all_scores.append(score)
        best_score = max(best_score, score)
        best_tile = max(best_tile, current_max_tile)
        training_scores.append(score)
      
        if done:
            env.reset()
            policy_net.reset_hidden(batch_size=1, device=device)
      
        else:
            policy_net.reset_hidden(batch_size=1, device=device)
        
        if episode % print_every == 0:
            avg_score = sum(all_scores[-print_every:]) / len(all_scores[-print_every:])
            avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0.0
            avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0.0
            avg_entropy = total_entropy / num_updates if num_updates > 0 else 0.0
            avg_kl = total_kl / num_updates if num_updates > 0 else 0.0
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Score: {score} | Avg({print_every}): {avg_score:.1f} | "
                f"Best: {best_score} | Tile: {best_tile} | "
                f"Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f} | "
                f"Entropy: {avg_entropy:.4f} | KL: {avg_kl:.4f} | "
                f"Max Spawn: {max_spawn if max_spawn else 'Normal'}"
            )
        
      
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            eval_scores, eval_best_tile, eval_1024_count, eval_2048_count = evaluate_policy_advanced(
                policy_net, num_episodes=50, device=device, use_lstm=True
            )
            avg_eval_score = sum(eval_scores) / len(eval_scores)
            best_eval_score = max(eval_scores)
            eval_results.append({
                'episode': episode + 1,
                'avg_score': avg_eval_score,
                'best_score': best_eval_score,
                'best_tile': eval_best_tile,
                '1024_count': eval_1024_count,
                '2048_count': eval_2048_count
            })
            print(
                f"\n=== Evaluation @ Episode {episode + 1} ===\n"
                f"Avg Score (50 ep): {avg_eval_score:.1f}\n"
                f"Best Score: {best_eval_score}\n"
                f"Best Tile: {eval_best_tile}\n"
                f"1024 Count: {eval_1024_count}/50\n"
                f"2048 Count: {eval_2048_count}/50\n"
            )
            
            if eval_best_tile >= best_tile:
                torch.save({
                    'policy_state_dict': policy_net.state_dict(),
                    'value_state_dict': value_net.state_dict(),
                    'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                    'value_optimizer_state_dict': value_optimizer.state_dict(),
                    'episode': episode + 1,
                    'best_tile': eval_best_tile,
                    'best_score': best_eval_score,
                }, f"{checkpoint_dir}/best_model_ep{episode+1}.pt")
    
    return policy_net, value_net, training_scores, eval_results
