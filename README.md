# CVRP Solver with Tabular Q-Learning and Deep Q-Network (DQN)

This repository provides implementations of **Tabular Q-Learning** and **Deep Q-Network (DQN)** to solve a variant of the **Capacitated Vehicle Routing Problem (CVRP)** — tailored for medical waste collection, using real hospital demand and distance data.

---

## 📁 Files Included

- `main_tabular_q.py` – Runs tabular Q-learning
- `dpn_adn.py` – Runs DQN-based reinforcement learning
- `hospitals.csv` – Hospital nodes and demand data
- `distance_matrix.csv` – 17×17 symmetric distance matrix

---

## 📦 Requirements

Create a virtual environment and install the following:

```bash
pip install -r requirements.txt
````

**`requirements.txt`** should include:

```txt
numpy
pandas
torch
```

If using GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 🧠 How It Works

### Tabular Q-Learning (`main_tabular_q.py`)

* Uses a simple lookup table `Q[state][action]`
* Trained over episodes with an epsilon-greedy policy
* Output:

  * Number of vehicles used
  * Each vehicle’s route
  * Final load
  * Total cost
  * Q-value sum per route (optional)

### DQN (`dpn_adn.py`)

* Uses a neural network to generalize Q-values
* Handles more complex state spaces (load, visited mask, etc.)
* Includes:

  * Replay buffer
  * Target network
  * Evaluation every N episodes
* Output:

  * Same as tabular Q-learning
  * Tracks best episode cost and Q-value quality

---

## 🚀 Running the Code

### Run Tabular Q-Learning

```bash
python main_tabular_q.py
```

### Run DQN (with GPU if available)

```bash
python dpn_adn.py
```

---

## 📈 Output Format

After training, you’ll see output like:

```
Training complete. Best routing cost: 130.78
Number of vehicles used: 5
 Vehicle  1 route: [0, 2, 1, 13, 9, 16]    Final load: 2.951    Cost: 22.63    Q-sum: -22.45
 ...
Total cost (all vehicles): 130.78
```

* `route`: nodes visited (0 = depot, 16 = disposal)
* `Final load`: how full the truck was before unloading
* `Cost`: total distance traveled by that vehicle
* `Q-sum`: total Q-values used in that route (DQN only)

---

## 📊 Data Format

### `hospitals.csv`

```csv
id,demand,x,y
0,0.0,0,0
1,1.2,15,20
...
16,0.0,50,50
```

### `distance_matrix.csv`

* 17×17 symmetric matrix of distances between nodes

---

## 🧾 Notes

* Truck capacity: `3.0`
* Depot node: `0`
* Disposal node: `16`
* Problem ends when all hospitals are served and truck is back at depot

---

## 🧑‍🎓 Educational Purpose

This project is a teaching tool to demonstrate:

* Reinforcement learning applied to logistics
* How Q-learning scales with table vs neural net
* Practical implementation with real-world style constraints

---

## 🏷 License

MIT License – feel free to use or adapt this code with attribution.

---

## 🙏 Acknowledgements

Based on methods described in:

> “Enhanced Vehicle Routing for Medical Waste Management via Hybrid Deep Reinforcement Learning and Optimization Algorithms” (2024)

```
