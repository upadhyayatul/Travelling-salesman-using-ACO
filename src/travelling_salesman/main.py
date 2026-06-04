import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Ensure the parent directory/current folder is in sys.path to resolve imports cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core import City, AntColonyOptimizer

# =====================================================================
# CONFIGURATION & PARAMETERS
# =====================================================================
# List of available cities
CITIES_LIST = [
    'CITY_1', 'CITY_2', 'CITY_3', 'CITY_4', 'CITY_5', 
    'CITY_6', 'CITY_7', 'CITY_8', 'CITY_9', 'CITY_10'
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
# MAIN RUNNER
# =====================================================================

def main():
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

if __name__ == '__main__':
    main()
