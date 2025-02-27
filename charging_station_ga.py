import numpy as np
import random

# -------------------------------
# Step 1: Define Problem Parameters
# -------------------------------

NUM_LOCATIONS = 10  # Example: 10 candidate charging locations
NUM_STATIONS = 4    # Example: Install 4 stations
POP_SIZE = 200       # Number of individuals in population
MUTATION_RATE = 0.2 # Mutation probability
GENERATIONS = 200   # Max generations

# Candidate locations (x, y coordinates, installation cost)
candidate_locations = [
    (5, 10, 3000), (10, 15, 3500), (15, 10, 4000), (20, 5, 2500), (25, 10, 3000),
    (30, 15, 4500), (35, 20, 5000), (40, 25, 5500), (45, 30, 6000), (50, 35, 6500)
]

# Demand points (x, y coordinates)
demand_points = [(8, 12), (12, 18), (18, 8), (22, 7), (28, 12),
                 (33, 17), (38, 22), (43, 27), (48, 32), (53, 37)]

BUDGET = 20000  # Maximum allowed cost
SERVICE_RADIUS = 10  # Maximum distance a station can serve

# -------------------------------
# Step 2: Helper Functions
# -------------------------------

# Function to calculate distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Compute coverage for a given selection of stations
def compute_coverage(selected_stations):
    covered_demand_points = set()
    
    for i, station in enumerate(candidate_locations):
        if selected_stations[i] == 1:  # If the station is selected
            for demand in demand_points:
                if euclidean_distance(station[:2], demand) <= SERVICE_RADIUS:
                    covered_demand_points.add(demand)
    
    return len(covered_demand_points)

# Compute total cost of a given selection
def compute_cost(selected_stations):
    total_cost = sum(candidate_locations[i][2] for i in range(len(candidate_locations)) if selected_stations[i] == 1)
    return total_cost

# Fitness function: maximize coverage while minimizing cost
def fitness_function(solution):
    coverage = compute_coverage(solution)
    cost = compute_cost(solution)

    # Apply penalty if cost exceeds budget
    if cost > BUDGET:
        return -1  # Penalize infeasible solutions
    return coverage - (cost / 10000)  # Normalize cost

# -------------------------------
# Step 3: Genetic Algorithm Components
# -------------------------------

# Generate initial population
def generate_population():
    return [np.random.choice([0, 1], size=NUM_LOCATIONS, p=[0.6, 0.4]).tolist() for _ in range(POP_SIZE)]

# Selection: Tournament Selection
def tournament_selection(population, fitness_scores, k=3):
    selected = random.sample(list(zip(population, fitness_scores)), k)
    return max(selected, key=lambda x: x[1])[0]  # Return best individual

# Crossover: One-Point Crossover
def crossover(parent1, parent2):
    point = random.randint(1, NUM_LOCATIONS - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation: Bit-flip mutation
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]  # Flip bit
    return individual

# -------------------------------
# Step 4: Running the Genetic Algorithm
# -------------------------------

def genetic_algorithm():
    population = generate_population()

    for generation in range(GENERATIONS):
        # Evaluate fitness of all individuals
        fitness_scores = [fitness_function(ind) for ind in population]

        # Select best solutions for reproduction
        new_population = []
        for _ in range(POP_SIZE // 2):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        # Replace old population with new one
        population = new_population

        # Track best solution
        best_solution = max(population, key=fitness_function)
        best_fitness = fitness_function(best_solution)

        print(f"Generation {generation+1}: Best Fitness = {best_fitness}, Coverage = {compute_coverage(best_solution)}, Cost = {compute_cost(best_solution)}")

    # Return best found solution
    return best_solution

# Run the genetic algorithm
best_solution = genetic_algorithm()
print("Best Charging Station Placement:", best_solution)
