import numpy as np
import pandas as pd
import random

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load Data
# ─────────────────────────────────────────────────────────────────────────────

# 1.1 Read hospitals.csv (17 rows: nodes 0..16)
hospitals = pd.read_csv('hospitals.csv')
if hospitals.shape[0] != 17:
    raise ValueError("Expected 17 rows (0..16) in hospitals.csv.")

# 1.2 Read distance_matrix.csv (17×17 symmetric)
dist_df = pd.read_csv('distance_matrix.csv', index_col=0)
dist_mat = dist_df.values
N = dist_mat.shape[0]
if N != 17:
    raise ValueError("Expected a 17×17 distance matrix.")

# 1.3 Extract demands (length-17 array)
demands = hospitals['demand'].values.astype(float)

# 1.4 Constants
DEPOT = 0
DISPOSAL = 16
VEHICLE_CAPACITY = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# 2) CVRP Environment (Hospitals only; Disposal unloads but is NOT marked visited)
# ─────────────────────────────────────────────────────────────────────────────

class CVRPEnv:
    """
    A CVRP environment where:
      - Hospitals (1..15) are only marked 'visited' if their entire demand
        fits into the truck's remaining capacity at that moment.
      - Disposal (16) unloads the truck (load → 0) but is NOT marked visited.
      - The episode ends when all hospitals (1..15) + depot (0) are in visited
        AND the truck is back at the depot (0) with load = 0.
    """

    def __init__(self, dist_matrix, demands, capacity):
        self.dist = dist_matrix  # 17×17 distance array
        self.demands = demands  # length-17 demand array
        self.capacity = capacity  # 3.0
        self.N = dist_matrix.shape[0]  # Should be 17
        self.reset()

    def reset(self):
        # Start at depot, load=0, visited only={0}, done=False
        self.pos = DEPOT
        self.load = 0.0
        # We'll store {0} plus any visited hospitals. Disposal is not stored here.
        self.visited = set([DEPOT])
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        Returns (pos, load, visited_mask), where visited_mask is a length-17
        float32 array with 1.0 for visited nodes (hospitals+depot), 0.0 otherwise.
        Disposal (16) is never marked visited in 'visited'.
        """
        mask = np.zeros(self.N, dtype=np.float32)
        for i in self.visited:
            mask[i] = 1.0
        return (self.pos, self.load, mask)

    def step(self, action):
        """
        1) If done=True, error.
        2) If action is an unvisited hospital j (1..15), require demands[j] <= remaining_capacity.
           If it does not fit, raise ValueError (invalid).
        3) Compute distance from current pos → action. If distance < 0, cost = 1e6; else cost = dist.
        4) If action in {1..15} and unvisited, pick up demands[action], mark visited.add(action).
        5) If action == DISPOSAL (16), unload (load = 0). Do NOT mark disposal visited.
        6) Move to action, reward = -cost.
        7) If all hospitals 1..15 plus depot (0) are visited AND pos == DEPOT AND load == 0, done=True.
        8) Return (next_state, reward, done).
        """

        if self.done:
            raise ValueError("step() called after done=True. Please reset().")

        # 2) If action is an unvisited hospital, ensure entire demand fits:
        if (action not in (DEPOT, DISPOSAL)) and (action not in self.visited):
            remaining_capacity = self.capacity - self.load
            if self.demands[action] > remaining_capacity:
                raise ValueError(
                    f"Cannot visit hospital {action}: demand {self.demands[action]:.3f} "
                    f"exceeds remaining capacity {remaining_capacity:.3f}."
                )

        # 3) Compute cost:
        dist = self.dist[self.pos, action]
        cost = dist if dist >= 0 else 1e6

        # 4) If action is a hospital (1..15) and unvisited, pick up entire demand
        if (action not in (DEPOT, DISPOSAL)) and (action not in self.visited):
            self.load += self.demands[action]
            self.visited.add(action)

        # 5) If action == DISPOSAL, unload (load → 0). Do NOT mark disposal visited.
        if action == DISPOSAL:
            self.load = 0.0

        # 6) Move
        self.pos = action
        reward = -cost

        # 7) Done check: all hospitals + depot visited AND pos == DEPOT AND load == 0
        if (len(self.visited) == 16) and (self.pos == DEPOT) and (self.load == 0.0):
            self.done = True

        # 8) Return
        return self._get_state(), reward, self.done


# ─────────────────────────────────────────────────────────────────────────────
# 3) Tabular Q-Learning Implementation
# ─────────────────────────────────────────────────────────────────────────────

def run_q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Tabular Q-Learning for cost-minimization (use min Q in Bellman).
    env: CVRPEnv instance
    episodes: number of training episodes
    alpha: learning rate
    gamma: discount factor
    epsilon: ε for ε-greedy exploration
    Returns: Q_table (dict mapping state_key → np.array of length N).
    """

    Q = {}

    def encode_state(state):
        pos, load, mask = state
        load_key = round(load, 3)
        return (pos, load_key, bytes(mask))

    for ep in range(episodes):
        state = env.reset()
        s_key = encode_state(state)

        for _ in range(200):  # limit steps per episode
            # ε-greedy: random action with prob ε, or best argmin Q otherwise
            if (random.random() < epsilon) or (s_key not in Q):
                action = random.randrange(env.N)
            else:
                action = int(np.argmin(Q[s_key]))

            # Try to step; if it raises ValueError (invalid hospital move), pick a fallback:
            try:
                next_state, reward, done = env.step(action)
            except ValueError:
                # Build fallback valid actions
                pos, load, visited_mask = state
                visited_set = set(np.where(visited_mask == 1.0)[0])
                rem_cap = env.capacity - load

                valid = []
                for j in range(env.N):
                    if j == DEPOT:
                        continue
                    if j == DISPOSAL:
                        valid.append(j)
                    elif (j not in visited_set) and (env.demands[j] <= rem_cap):
                        valid.append(j)

                if not valid:
                    action = DISPOSAL
                else:
                    action = random.choice(valid)

                next_state, reward, done = env.step(action)

            ns_key = encode_state(next_state)

            # Initialize Q rows if unseen
            if s_key not in Q:
                Q[s_key] = np.zeros(env.N, dtype=float)
            if ns_key not in Q:
                Q[ns_key] = np.zeros(env.N, dtype=float)

            # Bellman update (cost minimization, so use min over Q(next_state, ·))
            old_val = Q[s_key][action]
            Q[s_key][action] = old_val + alpha * (
                    reward + gamma * np.min(Q[ns_key]) - old_val
            )

            state = next_state
            s_key = ns_key

            if done:
                break

    return Q


# ─────────────────────────────────────────────────────────────────────────────
# 4) Simulate Greedy Policy (Capacity-Aware / Full-Demand Only), with final load
# ─────────────────────────────────────────────────────────────────────────────

def simulate_greedy(Q_table, env):
    """
    Simulate one run using the trained Q_table.

    Returns: list of (route, final_load, total_cost)
    """

    def encode_state(state):
        pos, load, mask = state
        load_key = round(load, 3)
        return (pos, load_key, bytes(mask))

    state = env.reset()
    s_key = encode_state(state)

    all_routes = []
    current_route = [0]
    current_cost = 0.0

    while True:
        pos, load, visited_mask = state
        visited_set = set(np.where(visited_mask == 1.0)[0])
        rem_cap = env.capacity - load

        # Fully servable hospitals
        fully_servable = [
            j for j in range(1, DISPOSAL)
            if (j not in visited_set) and (env.demands[j] <= rem_cap)
        ]

        if fully_servable:
            allowed = fully_servable
        else:
            unvisited_any = [j for j in range(1, DISPOSAL) if j not in visited_set]
            if unvisited_any:
                allowed = [DISPOSAL]
            elif load > 0:
                allowed = [DISPOSAL]
            elif env.pos != DEPOT:
                allowed = [DEPOT]
            else:
                break

        # Select action
        if s_key in Q_table:
            q_vals = Q_table[s_key]
            action = min(allowed, key=lambda j: q_vals[j])
            q_val =+ q_vals[action]
        else:
            action = random.choice(allowed)
            q_val = + q_vals[action]

        # Add travel cost
        dist = env.dist[env.pos, action]
        current_cost += dist if dist >= 0 else 1e6

        # Step
        next_state, reward, done = env.step(action)
        current_route.append(action)

        # Disposal: finalize and reset
        if action == DISPOSAL:
            final_load = load
            all_routes.append((current_route.copy(), final_load, current_cost, q_val))
            current_route = [0]
            current_cost = 0.0
            env.pos = DEPOT
            env.load = 0.0
            state = env._get_state()
            s_key = encode_state(state)
            continue

        if done:
            if not (len(current_route) == 2 and current_route[0] == 0 and current_route[1] == 0):
                all_routes.append((current_route.copy(), 0.0, current_cost))
            break

        state = next_state
        s_key = encode_state(state)

    return all_routes


# ─────────────────────────────────────────────────────────────────────────────
# 5) Main: Train Q-Learning & Extract Routes (including final_load)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for e in range(5):
        m = 999
        final = None
        for i in range(int(100000 / 10 ** e)):
            # (A) Instantiate environment
            full_env = CVRPEnv(dist_mat, demands, VEHICLE_CAPACITY)

            # (B) Train Tabular Q-Learning with α=0.1, γ=0.9, ε=0.1
            #print("Training Tabular Q-Learning on CCVRP...")
            Q_table = run_q_learning(
                full_env,
                episodes=10 ** e,
                alpha=0.1,
                gamma=0.9,
                epsilon=0.0
            )
            #print("Training complete. Number of distinct states learned:", len(Q_table))

            # (C) Simulate the learned policy greedily (with capacity & full-demand checks)
            routes_with_loads = simulate_greedy(Q_table, full_env)

            # (D) Print out each vehicle’s route and its final load, along with total vehicle count
            # (D) Print out each vehicle’s route, load, and cost
            #print("\nNumber of vehicles used:", len(routes_with_loads))
            t_cost = 0
            for v_idx, (route, final_load, total_cost, q_value) in enumerate(routes_with_loads, start=1):
                t_cost += total_cost
                #print(f" Vehicle {v_idx:2d} route: {route}    "
                      #f"Final load: {final_load:.3f}    Cost: {total_cost:.2f}      Q-value: {q_value}")
            #print(f"Total cost: {t_cost}")
            if t_cost <= m:
                m = t_cost
                final = routes_with_loads
        for v_idx, (route, final_load, total_cost, q_value) in enumerate(final, start=1):
            print(f" Vehicle {v_idx:2d} route: {route}    "
            f"Final load: {final_load:.3f}    Cost: {total_cost:.2f}      Q-value: {q_value}")
        print(f"Total cost: {m}")
