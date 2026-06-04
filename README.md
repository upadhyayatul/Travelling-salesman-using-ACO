# Traveling Salesman Problem (TSP) Using Ant Colony Optimization (ACO)

This repository contains a clean, modular, and optimized Python implementation of the **Ant Colony Optimization (ACO)** algorithm applied to the **Traveling Salesman Problem (TSP)**. 

The application utilizes **Object-Oriented Programming (OOP)**, provides mathematically correct probability selection (avoiding deadlocks), implements symmetric pheromone updates, and features a **dynamic, non-blocking real-time visualization** using Matplotlib.

---

## 📌 Problem Overview

The **Traveling Salesman Problem (TSP)** is a classic NP-hard optimization problem:
> Given a list of cities and the distances between each pair, what is the shortest possible route that visits each city exactly once and returns to the origin city?

Since exact solutions are computationally intractable (exponential time complexity) for large sets of cities, metaheuristics like **Ant Colony Optimization (ACO)** are used to find high-quality, near-optimal solutions in a fraction of the time.

---

## 🐜 How Ant Colony Optimization (ACO) Works

ACO is a nature-inspired metaheuristic based on the foraging behavior of real ants:
1. Ants wander randomly searching for food.
2. Upon finding food, they return to the colony while depositing a chemical trail called **pheromone**.
3. Other ants detect the pheromone and are probabilistically drawn to follow paths with higher pheromone concentrations.
4. Shorter paths are completed faster, resulting in faster pheromone accumulation, creating a positive feedback loop.
5. Over time, pheromones **evaporate**, preventing the algorithm from getting trapped in local optima.

---

## 🧮 Mathematical Formulation

### 1. Transition Probability
The probability $P_{ij}^k$ of ant $k$ transitioning from city $i$ to city $j$ is computed **only over the set of unvisited cities** ($\text{Allowed}_k$):

$$P_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in \text{Allowed}_k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta}$$

Where:
* $\tau_{ij}$ is the pheromone level on the edge connecting city $i$ and city $j$.
* $\eta_{ij} = \frac{1}{d_{ij}}$ is the visibility (heuristic desirability), representing the inverse of the distance between the two cities.
* $\alpha$ controls the influence of pheromone history.
* $\beta$ controls the influence of visibility (distance).

### 2. Pheromone Evaporation and Deposit
At the end of each iteration (after all ants have completed their tours), pheromone levels on all edges are evaporated, and new pheromones are deposited:

$$\tau_{ij} \leftarrow (1 - \rho)\tau_{ij} + \sum_{k=1}^{m} \Delta \tau_{ij}^k$$

Where:
* $\rho$ is the pheromone evaporation rate ($0 < \rho < 1$).
* $m$ is the total number of ants.
* $\Delta \tau_{ij}^k$ is the amount of pheromone deposited by ant $k$ if it traversed edge $(i, j)$:

$$\Delta \tau_{ij}^k = \begin{cases} 
\frac{Q}{L_k} & \text{if ant } k \text{ used edge } (i, j) \\
0 & \text{otherwise}
\end{cases}$$

Here, $L_k$ is the total tour length of ant $k$, and $Q$ is a constant parameter.

---

## 🏗️ Project Structure & Architecture

The project has been restructured into a modular Python package layout:

```text
Travelling-salesman-using-ACO/
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    └── travelling_salesman/
        ├── __init__.py
        ├── core.py          # Contains City, Ant, and AntColonyOptimizer classes
        └── main.py          # Entrypoint script (configuration, run loop, visualization)
```

The core codebase is divided into modular components:
1. **`City`** (in `core.py`): Represents a city's coordinates and name, and calculates the Euclidean distance to other cities.
2. **`Ant`** (in `core.py`): Manages state for an individual ant and implements the transition rules to construct a complete tour.
3. **`AntColonyOptimizer`** (in `core.py`): Coordinates the entire optimization process. It computes the distance/visibility matrices, maintains the pheromone matrix, runs iterations, evaporates/deposits pheromones, and tracks the global best solution.
4. **`main.py`**: Configures the simulation parameters, generates cities, runs the optimization loop, and orchestrates the Matplotlib GUI drawing.

---

## 🎨 Visualization Features

The project includes an interactive Matplotlib visualization that updates dynamically at the end of each iteration:
* 🔴 **Red Nodes**: Represent the cities.
* 🟢 **Green Edges**: Represent pheromone trails. The thickness and opacity of the green lines correspond to the relative strength of the pheromone on that connection.
* 🔵 **Blue Tour**: Represents the best path found so far.

---

## ⚙️ Configuration & Hyperparameters

You can tune the hyperparameters at the top of the `src/travelling_salesman/main.py` script:

```python
ALPHA = 1.0        # Pheromone importance
BETA = 2.0         # Heuristic desirability (visibility) importance
RHO = 0.2          # Pheromone evaporation rate
Q = 100.0          # Pheromone deposit constant
NUM_ANTS = 15      # Number of ants per generation
ITERATIONS = 50    # Number of optimization generations
```

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have Python 3 and the required libraries installed:
```bash
pip install -r requirements.txt
```

### 🏃 Running the Code
Execute the script directly from your terminal:
```bash
python src/travelling_salesman/main.py
```


---

## 📊 Sample Output
```text
Initializing cities...
Number of cities: 9
  City 0: CITY_1(12, 26)
  City 1: CITY_2(48, 92)
  City 2: CITY3(86, 92)
  ...

Starting optimization simulation...

------------------ Optimization Complete ------------------
Optimal path length: 271.95 km
Optimal path found:
CITY_5 -> CITY_1 -> CITY_8 -> CITY_7 -> CITY_9 -> CITY_4 -> CITY_3 -> CITY_2 -> CITY_6 -> CITY_5
-----------------------------------------------------------
```
