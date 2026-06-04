import numpy as np
import math 
import matplotlib.pyplot as plt
import time

# =====================================================================
# CONFIGURATION & PARAMETERS
# =====================================================================
# List of available cities
CITIES_LIST = [
    'CITY_1', 'CITY_2', 'CITY3', 'CITY4', 'CITY5', 
    'CITY6', 'CITY7', 'CITY8', 'CITY9', 'CITY10'
]

# Number of cities to select (randomly between 5 and 10 for a meaningful problem size)
NUM_CITIES = np.random.randint(5, 11)
SELECTED_CITIES = CITIES_LIST[:NUM_CITIES]

# ACO Hyperparameters
ALPHA = 1.0        # Pheromone importance parameter
BETA = 2.0         # Heuristic/visibility importance parameter
RHO = 0.2          # Pheromone evaporation rate (0 < RHO < 1)
Q = 100.0          # Pheromone constant deposited by ants
NUM_ANTS = 15      # Number of ants in each iteration
ITERATIONS = 50    # Number of optimization generations/iterations

# Coordinate bounds for random city generation
MIN_COORD = 5
MAX_COORD = 95

# =====================================================================
# CLASS DEFINITIONS
# =====================================================================

class City:
    """Represents a city with a name and 2D coordinates."""
    def __init__(self, name: str, x: float, y: float):
        self.name = name
        self.x = x
        self.y = y

    def distance_to(self, other: 'City') -> float:
        """Calculates the Euclidean distance to another city."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return f"{self.name}({self.x}, {self.y})"


class Ant:
    """Represents a single ant that constructs a TSP tour."""
    def __init__(self, start_city_idx: int):
        self.start_city_idx = start_city_idx
        self.current_city_idx = start_city_idx
        self.tour = [start_city_idx]
        self.tour_length = 0.0

    def reset(self, start_city_idx: int):
        """Resets the ant state for a new iteration."""
        self.start_city_idx = start_city_idx
        self.current_city_idx = start_city_idx
        self.tour = [start_city_idx]
        self.tour_length = 0.0

    def select_next_city(self, pheromone: np.ndarray, eta: np.ndarray, alpha: float, beta: float) -> int:
        """
        Chooses the next city to visit using standard ACO probabilistic transition rules.
        Only unvisited cities are considered, eliminating the potential for deadlocks.
        """
        n = pheromone.shape[0]
        unvisited = [i for i in range(n) if i not in self.tour]
        
        # If all cities visited, return to start
        if not unvisited:
            return self.start_city_idx

        # Compute transition probabilities for unvisited neighbors
        probabilities = []
        for city_idx in unvisited:
            tau = pheromone[self.current_city_idx][city_idx]
            visibility = eta[self.current_city_idx][city_idx]
            # Transition formula weight: (tau^alpha) * (eta^beta)
            probabilities.append((tau ** alpha) * (visibility ** beta))

        total_weight = sum(probabilities)
        if total_weight == 0:
            # Fallback to uniform distribution if pheromones are zeroed out
            probabilities = [1.0 / len(unvisited)] * len(unvisited)
        else:
            probabilities = [p / total_weight for p in probabilities]

        # Select next city using roulette-wheel choice
        return np.random.choice(unvisited, p=probabilities)

    def construct_tour(self, pheromone: np.ndarray, eta: np.ndarray, alpha: float, beta: float, distance_matrix: np.ndarray):
        """Builds a complete tour through all cities and returns to the start."""
        n = pheromone.shape[0]
        while len(self.tour) < n:
            next_city = self.select_next_city(pheromone, eta, alpha, beta)
            self.tour_length += distance_matrix[self.current_city_idx][next_city]
            self.tour.append(next_city)
            self.current_city_idx = next_city
        
        # Complete the loop: return to starting city
        self.tour_length += distance_matrix[self.current_city_idx][self.start_city_idx]
        self.tour.append(self.start_city_idx)


class AntColonyOptimizer:
    """Manages the ACO simulation, pheromones, and optimization loop."""
    def __init__(self, cities: list[City], num_ants: int, alpha: float, beta: float, rho: float, Q: float):
        self.cities = cities
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.num_cities = len(cities)
        
        # Precompute distance and visibility (1/distance) matrices
        self.distance_matrix = np.zeros((self.num_cities, self.num_cities))
        self.eta = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dist = self.cities[i].distance_to(self.cities[j])
                    self.distance_matrix[i][j] = dist
                    self.eta[i][j] = 1.0 / dist
                else:
                    self.distance_matrix[i][j] = 0.0
                    self.eta[i][j] = 0.0

        # Initialize pheromones to 1.0 on all edges
        self.pheromone = np.ones((self.num_cities, self.num_cities))
        
        # Best tour tracking
        self.global_best_tour = None
        self.global_best_distance = float('inf')

    def run_iteration(self) -> tuple[list[int], float]:
        """Runs a single generation/iteration of ants constructing tours."""
        # Initialize ants at random starting cities
        ants = [Ant(start_city_idx=np.random.randint(0, self.num_cities)) for _ in range(self.num_ants)]
        
        # 1. Construct tours for all ants in parallel/sequence
        for ant in ants:
            ant.construct_tour(self.pheromone, self.eta, self.alpha, self.beta, self.distance_matrix)
            
            # Keep track of global best solution
            if ant.tour_length < self.global_best_distance:
                self.global_best_distance = ant.tour_length
                self.global_best_tour = list(ant.tour)

        # 2. Evaporate pheromones on all edges
        self.pheromone *= (1.0 - self.rho)
        
        # 3. Deposit new pheromones symmetrically along the constructed tours
        for ant in ants:
            contribution = self.Q / ant.tour_length
            for idx in range(len(ant.tour) - 1):
                i = ant.tour[idx]
                j = ant.tour[idx + 1]
                self.pheromone[i][j] += contribution
                self.pheromone[j][i] += contribution
                
        # Enforce a small lower bound for pheromones to keep paths active
        self.pheromone = np.maximum(self.pheromone, 1e-4)
        
        return self.global_best_tour, self.global_best_distance


# =====================================================================
# MAIN SIMULATION RUNNER
# =====================================================================

if __name__ == '__main__':
    print("Initializing cities...")
    # Generate coordinates for selected cities
    cities = []
    x_coords = []
    y_coords = []
    for name in SELECTED_CITIES:
        x = np.random.randint(MIN_COORD, MAX_COORD)
        y = np.random.randint(MIN_COORD, MAX_COORD)
        cities.append(City(name, x, y))
        x_coords.append(x)
        y_coords.append(y)

    print(f"Number of cities: {NUM_CITIES}")
    for idx, c in enumerate(cities):
        print(f"  City {idx}: {c}")

    # Initialize the ACO optimizer
    optimizer = AntColonyOptimizer(
        cities=cities,
        num_ants=NUM_ANTS,
        alpha=ALPHA,
        beta=BETA,
        rho=RHO,
        Q=Q
    )

    print("\nStarting optimization simulation...")
    
    # Set up matplotlib for interactive, non-blocking plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 7))
    
    best_distances_history = []
    
    for i in range(1, ITERATIONS + 1):
        # Run one iteration of the optimizer
        best_tour, best_distance = optimizer.run_iteration()
        best_distances_history.append(best_distance)
        
        # Clear the plot axis for redraw
        ax.clear()
        
        # 1. Draw pheromone trails in green (opacity represents strength)
        max_pher = np.max(optimizer.pheromone)
        for u in range(NUM_CITIES):
            for v in range(u + 1, NUM_CITIES):
                p_val = optimizer.pheromone[u][v]
                line_alpha = min(0.6 * (p_val / max_pher), 0.6)  # Caps opacity at 0.6
                if line_alpha > 0.02:
                    ax.plot(
                        [cities[u].x, cities[v].x], 
                        [cities[u].y, cities[v].y], 
                        color='green', 
                        alpha=line_alpha, 
                        linewidth=1.2 + 2.0 * (p_val / max_pher)
                    )

        # 2. Draw the best tour in bright blue
        for idx in range(len(best_tour) - 1):
            c1 = cities[best_tour[idx]]
            c2 = cities[best_tour[idx + 1]]
            ax.plot([c1.x, c2.x], [c1.y, c2.y], color='#1f77b4', linewidth=2.5, zorder=3)
            
        # 3. Draw cities as red dots
        ax.scatter(x_coords, y_coords, color='red', s=60, edgecolors='black', zorder=4)
        
        # 4. Annotate city names
        for idx, city in enumerate(cities):
            ax.text(city.x + 1.5, city.y + 1.5, city.name, fontsize=10, fontweight='bold', zorder=5)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title(f"Ant Colony Optimization for TSP\nIteration {i}/{ITERATIONS} | Best Distance: {best_distance:.2f} km")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")
        ax.grid(True, linestyle='--', alpha=0.5)

        # Draw frame and pause slightly to animate
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.08)

    # Turn off interactive plotting mode so the final figure stays open
    plt.ioff()
    
    # Print results to console
    print("\n------------------ Optimization Complete ------------------")
    print(f"Optimal path length: {optimizer.global_best_distance:.2f} km")
    path_names = [cities[idx].name for idx in optimizer.global_best_tour]
    print("Optimal path found:")
    print(" -> ".join(path_names))
    print("-----------------------------------------------------------")
    
    # Display the final static plot
    plt.show()
