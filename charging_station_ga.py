import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Define Synthetic City Data
# -------------------------------

NUM_LOCATIONS = 10   # Number of candidate charging station locations
POP_SIZE = 100       # Number of individuals in the population
MUTATION_RATE = 0.1  # Mutation probability
GENERATIONS = 100    # Max generations
SERVICE_RADIUS = 10  # Maximum distance a station can serve
BUDGET = 20000       # Max total budget for stations

# Generate exactly NUM_LOCATIONS unique candidate locations
candidate_locations = set()
while len(candidate_locations) < NUM_LOCATIONS:
    candidate_locations.add((random.randint(0, 50), random.randint(0, 50), random.randint(3000, 6000)))
candidate_locations = list(candidate_locations)  # Convert to list for indexing

# Generate 50 random demand points (scattered across the city)
demand_points = [(random.randint(0, 50), random.randint(0, 50)) for _ in range(50)]

# -------------------------------
# Step 2: Helper Functions
# -------------------------------

def euclidean_distance(p1, p2):
    """Compute distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_coverage(selected_stations):
    """Count demand points covered by selected stations."""
    if len(selected_stations) != len(candidate_locations):
        raise ValueError(f"Mismatch: {len(selected_stations)} stations but {len(candidate_locations)} locations.")

    covered_demand_points = set()
    for i, station in enumerate(candidate_locations):
        if selected_stations[i] == 1:  # If station is selected
            for demand in demand_points:
                if euclidean_distance(station[:2], demand) <= SERVICE_RADIUS:
                    covered_demand_points.add(demand)
    return len(covered_demand_points)

def compute_cost(selected_stations):
    """Calculate total installation cost."""
    return sum(candidate_locations[i][2] for i in range(len(candidate_locations)) if selected_stations[i] == 1)

def fitness_function(solution):
    """Evaluate solution based on coverage and cost."""
    coverage = compute_coverage(solution)
    cost = compute_cost(solution)

    if cost > BUDGET:
        return -1  # Penalize over-budget solutions
    return (2 * coverage) - (cost / 20000)  # Adjust weights for better optimization

# -------------------------------
# Step 3: Genetic Algorithm Components
# -------------------------------

def generate_population():
    """Ensure all individuals are binary vectors of fixed length."""
    return [np.random.choice([0, 1], size=NUM_LOCATIONS, p=[0.7, 0.3]).tolist() for _ in range(POP_SIZE)]

def tournament_selection(population, fitness_scores, k=3):
    """Select best individual from k randomly chosen ones."""
    selected = []
    for _ in range(POP_SIZE):
        contenders = random.sample(list(zip(population, fitness_scores)), k)
        best = max(contenders, key=lambda x: x[1])[0]
        selected.append(best)
    return selected

def crossover(parent1, parent2):
    """Perform one-point crossover with length validation."""
    if len(parent1) != NUM_LOCATIONS or len(parent2) != NUM_LOCATIONS:
        raise ValueError(f"Parent length mismatch: {len(parent1)} vs {len(parent2)}, expected {NUM_LOCATIONS}")

    point = random.randint(1, NUM_LOCATIONS - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual):
    """Apply bit-flip mutation."""
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]  # Flip bit
    return individual

# -------------------------------
# Step 4: Run the Genetic Algorithm with Convergence Tracking
# -------------------------------

def genetic_algorithm():
    """Run the GA to find optimal station placement and track convergence."""
    population = generate_population()
    best_fitness_over_time = []  # Store best fitness per generation

    for generation in range(GENERATIONS):
        fitness_scores = [fitness_function(ind) for ind in population]

        # Track best solution fitness for convergence
        best_fitness = max(fitness_scores)
        best_fitness_over_time.append(best_fitness)

        # Selection
        parents = tournament_selection(population, fitness_scores)

        # Crossover and mutation
        offspring = []
        for i in range(0, POP_SIZE, 2):
            parent1 = parents[i]
            parent2 = parents[(i+1) % POP_SIZE]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([mutate(child1), mutate(child2)])

        # Replace population
        population = offspring[:POP_SIZE]

        print(f"Generation {generation+1}: Best Fitness = {best_fitness:.2f}")

    # Find final best solution
    final_fitness_scores = [fitness_function(ind) for ind in population]
    best_idx = np.argmax(final_fitness_scores)
    best_solution = population[best_idx]

    print("\nOptimal Charging Station Locations:")
    for i, selected in enumerate(best_solution):
        if selected == 1:
            print(f"Station {i+1}: {candidate_locations[i]}")

    return best_fitness_over_time  # Return fitness history for plotting

# Run the genetic algorithm
fitness_history = genetic_algorithm()

# -------------------------------
# Step 5: Plot the Convergence Graph
# -------------------------------
plt.plot(fitness_history)
plt.xlabel("Generations")
plt.ylabel("Best Fitness Score")
plt.title("Genetic Algorithm Convergence")
plt.grid(True)
plt.show()
