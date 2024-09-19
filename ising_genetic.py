import numpy as np
from numba import float64, int64, njit, void


@njit([float64(float64)])
def mplogp(x: np.float64):
    if x <= 0:
        return 0
    return -x * np.log(x)


@njit([int64[:, :](int64)])
def create_random_state(n: int):
    """Returns a random matrix of spins up and down

    Args:
        n (int): Size of the matrix

    Returns:
        np.ndarray: Matrix of shape (n,n) and random values in +1/-1.
    """
    return 2 * np.random.randint(0, 2, (n, n)) - 1


@njit([float64(int64, int64, int64[:, :])])
def compute_sum_neighbors_si_sj_per_site(i: int, j: int, lattice: np.ndarray):
    """Computes the Hamiltonian at a lattice site i,j given a lattice state as well as parameters values.

    Args:
        i (int): row of lattice site
        j (int): column of lattice site
        lattice (ndarray): lattice of spins in a given state

    Returns:
        float64: Value of the Hamiltonian at site i,j for this configuration.
    """
    n = lattice.shape[0]
    si = lattice[i, j]
    neighbors = [((i + 1) % n, j), ((i - 1) % n, j), (i, (j + 1) % n), (i, (j - 1) % n)]
    term_sum_neighbors = np.float64(0.0)
    for neighbor in neighbors:
        term_sum_neighbors += si * lattice[neighbor]
    Hi = -term_sum_neighbors / 2
    return Hi


@njit([float64(int64[:, :])])
def compute_sum_si(lattice: np.ndarray):
    """Computes the magnetization for a given lattice configuration

    Args:
        lattice (ndarray): Configuration of up and down spins

    Returns:
        float64: Total magnetization of the configuration
    """
    return np.sum(lattice)


@njit([float64(int64[:, :])])
def compute_sum_neighbors_si_sj(lattice: np.ndarray):
    """Computes the Hamiltonian given a lattice state as well as parameters values using the site method.

    Args:
        lattice (ndarray): lattice of spins in a given state.

    Returns:
        float64: Value of the Hamiltonian for this configuration.
    """
    n = lattice.shape[0]
    H = np.float64(0.0)
    for i in range(n):
        for j in range(n):
            H += compute_sum_neighbors_si_sj_per_site(i, j, lattice)
    return H


@njit([float64(int64[:, :], float64, float64)])
def compute_entropy_per_site(lattice: np.ndarray, magnetization: np.float64, spin_correlation: np.float64):
    """Computes an approximation of the entropy per site
    (see https://doi.org/10.1016/S0378-4371(02)01327-4 and https://doi.org/10.1142/S0217984905008153 )
    Note that in those papers, S_i take values +- 1/2 instead of +- 1.

    Args:
        lattice (np.ndarray): Spin configuration.
        mag_per_site (np.float64): Average magnetization.
        spin_corr_per_site (np.float64): Average two-spin correlation.

    Returns:
        float: Entropy per site
    """
    n_spins = lattice.size
    z = 4  # Nearest neighbours
    m = magnetization / 2 / n_spins
    c = spin_correlation / 4 / n_spins
    sigma_1 = mplogp(m + 1 / 2) + mplogp(-m + 1 / 2)
    sigma_2 = mplogp(m + c + 1 / 4) + mplogp(-m + c + 1 / 4) + 2 * mplogp(-c + 1 / 4)
    return (z / 2) * sigma_2 - (z - 1) * sigma_1


@njit([float64[:](int64[:, :], float64, float64)])
def compute_thermodynamics(lattice: np.ndarray, beta: np.float64, h: np.float64):
    """Compute the thermodynamics (energy per site, magnetization per site, free energy per site used as an objective function).

    Args:
        lattice (np.ndarray): Spin configuration.
        beta (np.float64): Inverse temperature
        h (np.float64): External coupling

    Returns:
        np.ndarray: Objective function to minimize.
    """
    n_spins = lattice.size
    hamiltonian_term_si_sj = compute_sum_neighbors_si_sj(lattice)
    magnetization = compute_sum_si(lattice)

    spin_correlation = -(2 / 4) * hamiltonian_term_si_sj
    entropy_per_site = compute_entropy_per_site(lattice, magnetization, spin_correlation)

    internal_energy = hamiltonian_term_si_sj - h * magnetization

    return np.array([internal_energy, magnetization, internal_energy - entropy_per_site * n_spins / beta])


@njit([void(int64[:, :, :], int64, int64, int64, int64)])
def cross_entropy(states: np.ndarray, idx_parent_1: int, idx_parent_2: int, idx_child_1: int, idx_child_2: int):
    """This function creates two children arrays from two parent arrays by combining their sublocks of random size s and n - s.

    Args:
        states (np.ndarray): Numpy array of states for each state. Should be of shape (n_states + 2 * n_children,n,n).
        idx_parent_1 (int): Index of the first parent. Should be lower than n_states.
        idx_parent_2 (int): Index of the second parent. Should be lower than n_states.
        idx_child_1 (int): Index of the first child. Should be greater than n_states.
        idx_child_2 (int): Index of the second child. Should be greater than n_states.
    """
    n = states.shape[1]
    s = np.random.randint(0, n)

    states[idx_child_1] = states[idx_parent_1].copy()
    states[idx_child_1, :s, s:] = states[idx_parent_2, :s, s:].copy()
    states[idx_child_1, s:, :s] = states[idx_parent_2, s:, :s].copy()

    states[idx_child_2] = states[idx_parent_2].copy()
    states[idx_child_2, :s, s:] = states[idx_parent_1, :s, s:].copy()
    states[idx_child_2, s:, :s] = states[idx_parent_1, s:, :s].copy()


@njit([void(int64[:, :, :], int64, float64)])
def mutate_state(states: np.ndarray, idx_child: int, mutation_rate: float):
    """Applies random mutations (spin flips) to a state with probability r (in-place).

    Args:
        states (np.ndarray): Lattice of spins
        idx_child (int): Index of the child to mutate.
        mutation_rate (float): Probability of mutation for a given spin
    """
    n = states.shape[1]
    mutation_matrix = 2 * (np.random.rand(n, n) > mutation_rate) - 1
    states[idx_child] *= mutation_matrix


@njit([float64[:](float64[:], float64)])
def compute_fitness(free_energies, beta):
    """Computes the fitness of each population member

    Args:
        free_energies (np.ndarray): Array of free energies
        beta (float): temperature
    """
    F_min = np.min(free_energies)
    # G_min = -1 / beta
    F_max = np.max(free_energies)
    # G_max = 0
    fitness = np.exp(-2 * (free_energies - F_min) / (F_max - F_min))
    return fitness


@njit([void(float64[:], int64[:, :, :], float64[:, :])])
def sort_by_fitness(fitness: np.ndarray, states: np.ndarray, thermodynamics: np.ndarray):
    """Sort in place states and thermodynamics arrays using the fitness array.

    Args:
        fitness (np.ndarray): Numpy array of fitness for each state. Should be of shape (n_states + 2 * n_children,).
        states (np.ndarray): Numpy array of states for each state. Should be of shape (n_states + 2 * n_children,n,n).
        thermodynamics (np.ndarray): Numpy array of thermodynamics for each state. Should be of shape (n_states + 2 * n_children,3).
    """
    idx_sort = np.argsort(fitness)[::-1]
    fitness[:] = fitness[idx_sort]
    states[:, :, :] = states[idx_sort, :, :]
    thermodynamics[:, :] = thermodynamics[idx_sort, :]


@njit([void(int64[:, :], int64[:], int64[:, :, :], float64[:, :], float64, float64, float64)])
def evolve_population(
    idx_parents_pairs: np.ndarray,
    list_idx_surplus: np.ndarray,
    states: np.ndarray,
    thermodynamics: np.ndarray,
    mutation_rate: np.float64,
    beta: np.float64,
    h: np.float64,
):
    """_summary_

    Args:
        idx_parents_pairs (np.ndarray): List of indices corresponding to possible parents. Should be of shape (n_states,)
        list_idx_surplus (np.ndarray): List of indices corresponding to possible children. Should be of shape (n_children,)
        states (np.ndarray): Numpy array of states for each state. Should be of shape (n_states + 2 * n_children,n,n).
        thermodynamics (np.ndarray): Numpy array of thermodynamics for each state. Should be of shape (n_states + 2 * n_children,3).
        mutation_rate (np.float64): Mutation rate for children.
        beta (np.float64): Inverse temperature.
        h (np.float64): External field coupling.
    """
    # Generate all children
    for j in range(idx_parents_pairs.shape[0]):
        # Choose two parents close to each other in fitness
        idx_parent_1 = idx_parents_pairs[j, 0]
        idx_parent_2 = idx_parents_pairs[j, 1]
        idx_child_1 = list_idx_surplus[2 * j]
        idx_child_2 = list_idx_surplus[2 * j + 1]

        # Create children states from them
        cross_entropy(states, idx_parent_1, idx_parent_2, idx_child_1, idx_child_2)

        # Mutate children
        mutate_state(states, idx_child_1, mutation_rate)
        mutate_state(states, idx_child_2, mutation_rate)

        # Compute thermodynamics
        thermodynamics[list_idx_surplus[2 * j]] = compute_thermodynamics(states[idx_child_1], beta, h)
        thermodynamics[list_idx_surplus[2 * j + 1]] = compute_thermodynamics(states[idx_child_2], beta, h)


def genetic_algorithm(
    n: int, beta: float, h: float, n_states=1_000, n_children=100, n_cycles=1_000, mutation_rate=0.01
):
    """Runs a genetic algorithm on a random population to find the Ising ground state at a given temperature

    Args:
        n (int): Number of spins in one dimension (total spins N = n^2).
        beta (float): Inverse temperature.
        h (float): External field coupling.
        n_states (_type_, optional): Size of population. Defaults to 1000.
        n_children (int, optional): Number of pairs children at each cycle. Defaults to 100.
        n_cycles (_type_, optional): Number of cycles of evolution. Defaults to 1_000.
        mutation_rate (float, optional): Mutation rate. Defaults to 0.01.

    Returns:
        dict: Dictionary containing a list of spin configuration and observables, as well as input parameters.
        Keys are "population", "energies", "magnetizations", "n", "J", "beta", "h", "n_states", "n_children", "n_cycles", "mutation_rate".
    """
    # Convert inputs into numpy floats
    h = np.float64(h)
    beta = np.float64(beta)

    # Total number of parents and children
    n_tot = n_states + 2 * n_children

    # List of indices for our population
    list_idx = np.arange(n_states)
    list_idx_tot = np.arange(n_tot)
    list_idx_surplus = np.arange(n_states, n_tot)

    # Initial population building
    states = np.array([create_random_state(n) for _ in list_idx_tot])
    # Thermodynamics is an array whose rows corresponding to states and columns are in order energy, magnetization, free energy
    thermodynamics = np.array([compute_thermodynamics(lattice, beta, h) for lattice in states])
    fitness = compute_fitness(thermodynamics[:, 2], beta)
    # We sort the initial population
    sort_by_fitness(fitness, states, thermodynamics)

    for _ in range(n_cycles):
        # We implement a roulette wheel selection
        total_fitness = np.sum(fitness[:n_states])
        relative_fitness_parents = fitness[:n_states] / total_fitness
        idx_parents_pairs = np.random.choice(list_idx, (n_children, 2), replace=False, p=relative_fitness_parents)
        # Generate all children
        evolve_population(idx_parents_pairs, list_idx_surplus, states, thermodynamics, mutation_rate, beta, h)

        # After generating the children, sort by custom comparison (energies with random Boltzmann mixing)
        fitness = compute_fitness(thermodynamics[:, 2], beta)
        sort_by_fitness(fitness, states, thermodynamics)

    # We end up with a population of lowest energy states. We can return the minimum of these or do statistics on the final population
    energies = thermodynamics[:n_states, 0]
    fenergies = thermodynamics[:n_states, 2]
    fitness = fitness[:n_states]
    population = states[:n_states]
    magnetizations = thermodynamics[:n_states, 1]
    return {
        "population": population,
        "free_energy": fenergies,
        "energy": energies,
        "entropy": (energies - fenergies) * beta,
        "magnetization": magnetizations,
        "free_energy_per_site": fenergies / n**2,
        "energy_per_site": energies / n**2,
        "entropy_per_site": (energies - fenergies) * beta / n**2,
        "magnetization_per_site": magnetizations / n**2,
        "fitness": fitness,
        "n": n,
        "beta": beta,
        "h": h,
        "n_states": n_states,
        "n_children": n_children,
        "n_cycles": n_cycles,
        "mutation_rate": mutation_rate,
    }
