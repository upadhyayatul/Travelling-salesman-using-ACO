import numpy as np
import math

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
        
        # 1. Construct tours for all ants
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
