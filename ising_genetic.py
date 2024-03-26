import functools

import numba as nb
import numpy as np
from numba import jit, njit


def create_random_state(n: int):
    """Returns a random matrix of spins up and down

    Args:
        n (int): Size of the matrix

    Returns:
        np.ndarray: Matrix of shape (n,n) and random values in +1/-1.
    """
    return 2 * np.random.randint(0, 2, (n, n)) - 1


@njit
def compute_hamiltonian_term(i: int, j: int, lattice: np.ndarray):
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
    # Remove the extra factor of 4 due to duplicate counting of pairs
    Hi = -term_sum_neighbors
    return Hi


@njit
def compute_magnetization(lattice: np.ndarray):
    """Computes the magnetization for a given lattice configuration

    Args:
        lattice (ndarray): Configuration of up and down spins

    Returns:
        float: Average magnetization of the configuration
    """
    N = np.int64(lattice.size)
    return np.sum(lattice) / N


@njit
def compute_hamiltonian_from_site(lattice: np.ndarray):
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
            H += compute_hamiltonian_term(i, j, lattice)
    return H / n**2 / 2


@njit
def compute_entropy_per_site(lattice: np.ndarray, mag_per_site: np.float64, spin_corr_per_site: np.float64):
    """Computes an approximation of the entropy per site
    (see https://doi.org/10.1016/S0378-4371(02)01327-4 and https://doi.org/10.1142/S0217984905008153 )
    Note that in those papers, S_i take values +- 1/2 instead of +- 1.

    Args:
        lattice (np.ndarray): Spin configuration
        mag_per_site (np.float64): Average magnetization per site
        spin_corr_per_site (np.float64): Average two-spin correlation

    Returns:
        float: Entropy per site
    """
    n = lattice.size
    z = 4  # Nearest neighbours
    m = mag_per_site / 2
    c = spin_corr_per_site / 4
    sigma_1 = mplogp(m + 1 / 2) + mplogp(-m + 1 / 2)
    sigma_2 = mplogp(m + c + 1 / 4) + mplogp(-m + c + 1 / 4) + 2 * mplogp(-c + 1 / 4)
    return (z / 2) * sigma_2 - (z - 1) * sigma_1


@njit
def compute_gibbs_free_energy_per_site(lattice: np.ndarray, beta: np.float64, h: np.float64):
    n = lattice.shape[0]
    hamiltonian_term = compute_hamiltonian_from_site(lattice)
    mag_per_site = compute_magnetization(lattice)
    spin_cross_per_site = -(2 / 4) * hamiltonian_term
    entropy_per_site = compute_entropy_per_site(lattice, mag_per_site, spin_cross_per_site)
    internal_energy = hamiltonian_term - h * mag_per_site
    return np.array([internal_energy, mag_per_site, internal_energy - entropy_per_site / beta])


@njit
def cross_entropy(lattice_1: np.ndarray, lattice_2: np.ndarray):
    """This function creates two children arrays from two parent arrays by combining their sublocks of random size s and n - s.

    Args:
        lattice_1 (np.ndarray): Parent array 1
        lattice_2 (np.ndarray): Parent array 2

    Returns:
        tuple(np.ndarray, np.ndarray): Children arrays
    """
    n = lattice_1.shape[0]
    s = np.random.randint(0, n)

    r1 = np.zeros((n, n))
    r2 = np.zeros((n, n))

    r1[:s, :] = lattice_1[:s, :]
    r1[s:, :] = lattice_2[s:, :]

    r2[:s, :] = lattice_2[:s, :]
    r2[s:, :] = lattice_1[s:, :]

    return r1, r2


@njit
def mutate_state(lattice: np.ndarray, r: float):
    """Applies random mutations (spin flips) to a state with probability r (in-place).

    Args:
        lattice (np.ndarray): Lattice of spins
        r (float): Probability of mutation for a given spin
    """
    n = lattice.shape[0]
    mutation_matrix = 2 * (np.random.rand(n, n) > r) - 1
    lattice *= mutation_matrix


@njit
def mplogp(x: np.float64):
    if x < 0:
        print(x)
        raise ValueError(f"Error in probability when computing entropy x = {x}")
    if x == 0:
        return 0
    return -x * np.log(x)


def bundle_state(lattice: np.ndarray, beta: np.float64, h: np.float64):
    """Given a spin configuration and parameters, bundles the state, energy and partition function term inside a tuple.

    Args:
        lattice (np.ndarray): Spin configuration
        beta (float): Inverse temperature
        h (float): External coupling

    Returns:
        tuple: Bundle of information for a given population member
    """
    thermodynamics = compute_gibbs_free_energy_per_site(lattice, beta, h)
    energy = thermodynamics[0]
    mag = thermodynamics[1]
    gibbs_energy = thermodynamics[2]
    return (energy, mag, gibbs_energy, 0.0, lattice)


def compute_fitness(states, beta):
    """Computes the fitness of each population member

    Args:
        states (np.ndarray): Structured array of states
        beta (float): temperature
    """
    G_vals = states["gibbs_energy"]
    G_min = np.min(G_vals)
    # G_min = -1 / beta
    G_max = 0
    # G_max = 0
    fitness = np.exp((G_vals - G_min) / (G_max - G_min))
    states["fitness"] = fitness


def compare_states(state_1, state_2):
    """Custom comparison of two states. State_1 is declared smaller than state_2 if its energy is smaller
    or through random acceptance comparing to the Boltzmann weight.

    Args:
        state_1 (tuple(float, np.ndarray)): State 1
        state_2 (tuple(float, np.ndarray)): State 2

    Returns:
        Int: Returns 1 if state_1 > state_2, -1 otherwise and 0 if equal.
    """
    deltabE = state_1[0] - state_2[0]
    r = np.random.rand()
    if deltabE == 0:
        return 0
    elif deltabE < 0:
        return -1
    else:
        if r < np.exp(-deltabE):
            return -1
        else:
            return 1


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

    # List of indices for our population
    list_idx = np.arange(n_states)

    # Structured array to store the population and their properties
    dtype = np.dtype(
        [
            ("energy", np.float64),
            ("magnetization", np.float64),
            ("gibbs_energy", np.float64),
            ("fitness", np.float64),
            ("state", np.int64, (n, n)),
        ]
    )
    states_energies = np.array([], dtype=dtype)

    # Initial population building
    for i in range(n_states):
        state = create_random_state(n)
        states_energies = np.append(states_energies, np.array([bundle_state(state, beta, h)], dtype=dtype))
    # We sort the initial population
    # states_energies = np.array(sorted(list(states_energies), key=functools.cmp_to_key(compare_states)))
    compute_fitness(states_energies, beta)
    states_energies = np.sort(states_energies, order="fitness")
    for i in range(n_cycles):
        # We implement a roulette wheel selection
        total_fitness = np.sum(states_energies["fitness"])
        relative_fitness = states_energies["fitness"] / total_fitness
        idx_parents_pairs = [
            np.random.choice(list_idx, 2, replace=False, p=relative_fitness) for j in range(n_children)
        ]
        # Generate all children
        for j in range(n_children):
            # Choose two parents close to each other in fitness
            m1 = idx_parents_pairs[j][0]
            m2 = idx_parents_pairs[j][1]
            parent_1 = states_energies[m1]
            parent_2 = states_energies[m2]

            # Create children states from them
            child_1, child_2 = cross_entropy(parent_1[-1], parent_2[-1])

            # Mutate children
            mutate_state(child_1, mutation_rate)
            mutate_state(child_2, mutation_rate)

            # Add children to the population
            states_energies = np.append(
                states_energies,
                np.array(
                    [
                        bundle_state(child_1, beta, h),
                        bundle_state(child_2, beta, h),
                    ],
                    dtype=dtype,
                ),
            )

        # After generating the children, sort by custo comparison (energies with random Boltzmann mixing)
        # states_energies = np.array(sorted(list(states_energies), key=functools.cmp_to_key(compare_states)))
        compute_fitness(states_energies, beta)
        states_energies = np.sort(states_energies, order="fitness")
        # Drop the highest energy states from the population
        states_energies = states_energies[:n_states]

    # We end up with a population of lowest energy states. We can return the minimum of these or do statistics on the final population
    energies = states_energies["energy"]
    benergies = beta * energies
    fenergies = states_energies["gibbs_energy"]
    fitness = states_energies["fitness"]
    population = states_energies["state"]
    magnetizations = states_energies["magnetization"]
    return {
        "population": population,
        "energies": energies,
        "benergies": benergies,
        "gibbs_energies": fenergies,
        "fitness": fitness,
        "magnetizations": magnetizations,
        "n": n,
        "beta": beta,
        "h": h,
        "n_states": n_states,
        "n_children": n_children,
        "n_cycles": n_cycles,
        "mutation_rate": mutation_rate,
    }
