import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load Data
# ─────────────────────────────────────────────────────────────────────────────

hospitals = pd.read_csv('hospitals.csv')
if hospitals.shape[0] != 17:
    raise ValueError("Expected 17 rows (0..16) in hospitals.csv.")

dist_df = pd.read_csv('distance_matrix.csv', index_col=0)
dist_mat = dist_df.values
if dist_mat.shape != (17, 17):
    raise ValueError("Expected a 17×17 distance matrix.")

demands = hospitals['demand'].values.astype(float)

DEPOT = 0
DISPOSAL = 16
VEHICLE_CAPACITY = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# 2) CVRP Environment
# ─────────────────────────────────────────────────────────────────────────────

class CVRPEnv:
    """
    CVRP environment where:
      - State = (pos, load, visited_mask)
      - Action = next node in {0..16}
      - Visiting an unvisited hospital picks up its full demand (if it fits).
      - Visiting DISPOSAL (16) unloads to zero.
      - Reward = -distance(pos→action).
      - Done when all hospitals 1..15 + depot are visited, load=0, and pos=DEPOT.
    """

    def __init__(self, dist_matrix, demands, capacity):
        self.dist = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.N = dist_matrix.shape[0]
        self.reset()

    def reset(self):
        self.pos = DEPOT
        self.load = 0.0
        self.visited = set([DEPOT])
        self.done = False
        return self._get_state()

    def _get_state(self):
        mask = np.zeros(self.N, dtype=np.float32)
        for i in self.visited:
            mask[i] = 1.0
        return (self.pos, self.load, mask)

    def step(self, action):
        if self.done:
            raise ValueError("Called step() after done=True. Call reset().")

        # If action is an unvisited hospital, ensure its demand fits entirely:
        if (action not in (DEPOT, DISPOSAL)) and (action not in self.visited):
            rem_cap = self.capacity - self.load
            if self.demands[action] > rem_cap:
                raise ValueError(
                    f"Cannot visit hospital {action}: demand {self.demands[action]:.3f} > rem {rem_cap:.3f}"
                )

        # 1) Travel cost
        d = self.dist[self.pos, action]
        cost = d if d >= 0 else 1e6

        # 2) If visiting a new hospital, pick up full demand
        if (action not in (DEPOT, DISPOSAL)) and (action not in self.visited):
            self.load += self.demands[action]
            self.visited.add(action)

        # 3) If disposal, unload
        if action == DISPOSAL:
            self.load = 0.0

        # 4) Move
        self.pos = action
        reward = -cost

        # 5) Done check: all hospitals+depot visited, pos==DEPOT, load==0
        if (len(self.visited) == 16) and (self.pos == DEPOT) and (self.load == 0.0):
            self.done = True

        return self._get_state(), reward, self.done


# ─────────────────────────────────────────────────────────────────────────────
# 3) DQN Network & Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class DQNNet(nn.Module):
    def __init__(self, input_dim=35, hidden_dim=128, output_dim=17):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# 4) State‐Encoding Helper
# ─────────────────────────────────────────────────────────────────────────────

def encode_state_for_dqn(state):
    """
    state = (pos, load, mask)
    → 35‐dim vector:
      [0..16]: one‐hot(pos)
      [17]   : load / VEHICLE_CAPACITY
      [18..34]: mask[0..16]
    """
    pos, load, mask = state
    pos_onehot = np.zeros(17, dtype=np.float32)
    pos_onehot[int(pos)] = 1.0
    load_norm = np.array([load / VEHICLE_CAPACITY], dtype=np.float32)
    return np.concatenate([pos_onehot, load_norm, mask], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 5) Greedy Rollout Using a Given Policy (policy_net)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_greedy_dqn(policy_net, env, device=torch.device("cpu")):
    """
    Run one greedy rollout under policy_net:
      1) At state (pos,load,mask), compute rem_cap = capacity−load, visited_set.
      2) If any unvisited hospital i with demands[i] ≤ rem_cap, allowed = those i.
         Else if load>0, allowed = [DISPOSAL].
         Else if pos != DEPOT, allowed = [DEPOT].
         Else: done.
      3) Among allowed, choose argmax_a Q(s,a).
      4) Travel to that action (accumulate distance), step in env.
      5) If action==DISPOSAL, record (route, load_before_unload, cost, qsum),
         then “teleport” back to depot (pos=0,load=0) for the next trip.
      6) Stop when done=True.
    Returns a list of (route, load_before_unload, cost, qsum).
    """

    state = env.reset()
    state_enc = encode_state_for_dqn(state)

    all_routes = []
    current_route = [0]
    current_cost = 0.0
    current_q_sum = 0.0

    while True:
        pos, load, visited_mask = state
        visited_set = set(np.where(visited_mask == 1.0)[0])
        rem_cap = env.capacity - load

        # 1) Build list of “fully_servable” hospitals
        servable = [
            h for h in range(1, DISPOSAL)
            if (h not in visited_set) and (demands[h] <= rem_cap)
        ]

        if servable:
            allowed = servable
        else:
            unvisited_any = [h for h in range(1, DISPOSAL) if h not in visited_set]
            if unvisited_any:
                allowed = [DISPOSAL]
            elif load > 0:
                allowed = [DISPOSAL]
            elif pos != DEPOT:
                allowed = [DEPOT]
            else:
                break

        # 2) Argmax over allowed
        st_t = torch.tensor(state_enc, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            qvals = policy_net(st_t).cpu().numpy().flatten()
        best_action = max(allowed, key=lambda a: qvals[a])
        current_q_sum += qvals[best_action]

        # 3) Accumulate travel cost
        dist = env.dist[pos, best_action]
        current_cost += dist if dist >= 0 else 1e6

        # 4) Step (with fallback for invalid)
        try:
            next_state, reward, done = env.step(best_action)
        except ValueError:
            # Fallback: if load>0, go to DISPOSAL; else go to DEPOT
            if load > 0:
                best_action = DISPOSAL
            else:
                best_action = DEPOT
            next_state, reward, done = env.step(best_action)

        next_enc = encode_state_for_dqn(next_state)
        current_route.append(best_action)

        # 5) If disposal, record and “teleport” back to depot
        if best_action == DISPOSAL:
            load_before_unload = load
            all_routes.append((current_route.copy(), load_before_unload, current_cost, current_q_sum))

            # Reset for next trip
            current_route = [0]
            current_cost = 0.0
            current_q_sum = 0.0
            env.pos = DEPOT
            env.load = 0.0
            state = env._get_state()
            state_enc = encode_state_for_dqn(state)
            continue

        # 6) If done, record last trip (if any)
        if done:
            if len(current_route) > 1:
                all_routes.append((current_route.copy(), 0.0, current_cost, current_q_sum))
            break

        state = next_state
        state_enc = next_enc

    return all_routes


# ─────────────────────────────────────────────────────────────────────────────
# 6) DQN Training Loop (“Lock‐in Best” + Plateau ε)
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn_keep_best(
        env,
        num_episodes=500,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        epsilon_plateau=100,
        target_update_freq=10,
        buffer_capacity=50000,
        eval_interval=50,
        device=torch.device("cpu")
):
    """
    Train a DQN for the CCVRP environment:
      - We use ε=1.0 for the first `epsilon_plateau` episodes, then decay
        ε ← max(epsilon_end, ε * epsilon_decay).
      - Keep track of `best_net` whenever evaluation cost < best_cost. As soon
        as we find a new best, immediately reload `best_net` into `policy_net` and
        `target_net` so we never “forget.”
      - Periodically evaluate every `eval_interval` episodes (and also at final).
    Returns (best_net, best_cost).
    """

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    policy_net = DQNNet().to(device)
    target_net = DQNNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    epsilon = epsilon_start

    best_net = DQNNet().to(device)
    best_net.load_state_dict(policy_net.state_dict())
    best_cost = float('inf')

    max_steps = 5000

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state_enc = encode_state_for_dqn(state)
        steps = 0

        while True:
            if steps >= max_steps:
                # early termination to avoid a stuck policy
                break

            # 1) ε‐greedy with plateau until `epsilon_plateau`
            if episode <= epsilon_plateau:
                action = random.randrange(env.N)
            else:
                if random.random() < epsilon:
                    action = random.randrange(env.N)
                else:
                    with torch.no_grad():
                        st_t = torch.tensor(state_enc, dtype=torch.float32, device=device).unsqueeze(0)
                        qvals = policy_net(st_t)
                        action = int(torch.argmax(qvals))

            # 2) Step (with fallback)
            try:
                next_state, reward, done = env.step(action)
            except ValueError:
                pos, load, vmask = state
                rem_cap = env.capacity - load
                visited_set = set(np.where(vmask == 1.0)[0])
                valid = []
                for j in range(env.N):
                    if j == DEPOT:
                        continue
                    if j == DISPOSAL:
                        valid.append(j)
                    elif (j not in visited_set) and (demands[j] <= rem_cap):
                        valid.append(j)
                action = random.choice(valid) if valid else DISPOSAL
                next_state, reward, done = env.step(action)

            next_enc = encode_state_for_dqn(next_state)
            replay_buffer.push(state_enc, action, reward, next_enc, done)

            state = next_state
            state_enc = next_enc
            steps += 1

            # 3) Learning step
            if len(replay_buffer) >= batch_size:
                bs, ba, br, bns, bd = replay_buffer.sample(batch_size)

                sb = torch.tensor(bs, dtype=torch.float32, device=device)
                ab = torch.tensor(ba, dtype=torch.int64, device=device).unsqueeze(1)
                rb = torch.tensor(br, dtype=torch.float32, device=device).unsqueeze(1)
                nsb = torch.tensor(bns, dtype=torch.float32, device=device)
                db = torch.tensor(bd, dtype=torch.uint8, device=device).unsqueeze(1)

                q_vals = policy_net(sb).gather(1, ab)
                with torch.no_grad():
                    nxt = target_net(nsb)
                    max_next, _ = torch.max(nxt, dim=1, keepdim=True)
                    target_q = rb + gamma * max_next * (1 - db.float())

                loss = mse_loss(q_vals, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # 4) Decay ε (only after plateau)
        if episode > epsilon_plateau:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # 5) Sync target_net every few episodes
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 6) Periodic evaluation
        if episode % eval_interval == 0 or episode == num_episodes:
            eval_env = CVRPEnv(dist_mat, demands, VEHICLE_CAPACITY)
            routes = simulate_greedy_dqn(policy_net, eval_env, device)
            cost_sum = sum(r[2] for r in routes)

            if cost_sum < best_cost:
                best_cost = cost_sum
                best_net.load_state_dict(policy_net.state_dict())

                # ─── Immediately lock in best weights ───
                policy_net.load_state_dict(best_net.state_dict())
                target_net.load_state_dict(best_net.state_dict())

            print(
                f"Episode {episode}/{num_episodes}  "
                f"Eval Cost: {cost_sum:.2f}  Best Cost: {best_cost:.2f}  Epsilon: {epsilon:.3f}"
            )

    return best_net, best_cost


# ─────────────────────────────────────────────────────────────────────────────
# 7) Main: Three Runs (500, 1000, 2000) – Verified 10×
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = CVRPEnv(dist_mat, demands, VEHICLE_CAPACITY)
    episode_counts = [50, 80, 100, 10000]

    for num_episodes in episode_counts:
        print(f"\n=== Training for {num_episodes} episodes ===")

        best_policy, best_cost = train_dqn_keep_best(
            env,
            num_episodes=num_episodes,
            batch_size=64,
            gamma=0.99,
            lr=1e-3,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9999,
            epsilon_plateau=100,  # ε=1.0 until episode 100, then decay
            target_update_freq=10,
            buffer_capacity=50000,
            eval_interval=int(num_episodes / 5) if num_episodes <= 100 else 200,  # evaluate every 100 episodes
            device=device
        )

        print(f"\nTraining complete for {num_episodes} episodes.  Best routing cost: {best_cost:.2f}")
        final_env = CVRPEnv(dist_mat, demands, VEHICLE_CAPACITY)
        final_routes = simulate_greedy_dqn(best_policy, final_env, device=device)

        print(f"Number of vehicles used: {len(final_routes)}")
        total_cost = 0.0
        for vidx, (route, load, cost, qsum) in enumerate(final_routes, start=1):
            total_cost += cost
            print(
                f" Vehicle {vidx:2d} route: {route}    Final load: {load:.3f}    Cost: {cost:.2f}    Q-sum: {qsum:.2f}")
        print(f"Total cost (all vehicles): {total_cost:.2f}")
