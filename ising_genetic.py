import functools

import numba as nb
import numpy as np
from numba import jit, njit
from scipy.special import binom


def create_random_state(n: int):
    """Returns a random matrix of spins up and down

    Args:
        n (int): Size of the matrix

    Returns:
        np.ndarray: Matrix of shape (n,n) and random values in +1/-1.
    """
    return 2 * np.random.randint(0, 2, (n, n)) - 1


@njit
def compute_hamiltonian_term(i: int, j: int, lattice: np.ndarray, h: np.float64):
    """Computes the Hamiltonian at a lattice site i,j given a lattice state as well as parameters values.

    Args:
        i (int): row of lattice site
        j (int): column of lattice site
        lattice (ndarray): lattice of spins in a given state.
        h (float64): External field coupling.

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
    term_sum_neighbors = term_sum_neighbors / 2
    Hi = -term_sum_neighbors - h * si
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
def compute_hamiltonian_from_site(lattice: np.ndarray, h: np.float64):
    """Computes the Hamiltonian given a lattice state as well as parameters values using the site method.

    Args:
        lattice (ndarray): lattice of spins in a given state.
        h (float64): External field coupling.

    Returns:
        float64: Value of the Hamiltonian for this configuration.
    """
    n = lattice.shape[0]
    H = np.float64(0.0)
    for i in range(n):
        for j in range(n):
            H += compute_hamiltonian_term(i, j, lattice, h)
    return H


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


def compute_entropy(lattice: np.ndarray):
    """Computes the microcanonical entropy by computing the number of microstate configurations

    Args:
        lattice (np.ndarray): Spin configuration

    Returns:
        float: Microcanonical entropy
    """
    n = lattice.size
    number_ones = np.sum(lattice == 1)
    p = binom(n, number_ones) / 2**n
    return p * np.log(p)


def bundle_state(lattice: np.ndarray, beta: np.float64, h: np.float64):
    """Given a spin configuration and parameters, bundles the state, energy and partition function term inside a tuple.

    Args:
        lattice (np.ndarray): Spin configuration
        beta (float): Inverse temperature
        h (float): External coupling

    Returns:
        tuple: Bundle of information for a given population member
    """
    energy = compute_hamiltonian_from_site(lattice, h)
    benergy = beta * energy
    return (energy, benergy, lattice)


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
    h64 = np.float64(h)
    beta64 = np.float64(beta)

    # Structured array to store the population and their properties
    dtype = np.dtype([("energy", np.float64), ("benergy", np.float64), ("state", np.int64, (n, n))])
    states_energies = np.array([], dtype=dtype)

    # Initial population building
    for i in range(n_states):
        state = create_random_state(n)
        states_energies = np.append(states_energies, np.array([bundle_state(state, beta64, h64)], dtype=dtype))
    # We sort the initial population
    states_energies = np.array(sorted(list(states_energies), key=functools.cmp_to_key(compare_states)))

    for i in range(n_cycles):
        # Generate all children
        for j in range(n_children):
            # Choose two parents close to each other in fitness
            m1 = 2 * j
            m2 = 2 * (j + 1)
            parents = states_energies[m1:m2]

            # Create children states from them
            child_1, child_2 = cross_entropy(parents[0][-1], parents[1][-1])

            # Mutate children
            mutate_state(child_1, mutation_rate)
            mutate_state(child_2, mutation_rate)

            # Add children to the population
            states_energies = np.append(
                states_energies,
                np.array(
                    [
                        bundle_state(child_1, beta64, h64),
                        bundle_state(child_2, beta64, h64),
                    ],
                    dtype=dtype,
                ),
            )

        # After generating the children, sort by custo comparison (energies with random Boltzmann mixing)
        states_energies = np.array(sorted(list(states_energies), key=functools.cmp_to_key(compare_states)))
        # Drop the highest energy states from the population
        states_energies = states_energies[:n_states]

    # We end up with a population of lowest energy states. We can return the minimum of these or do statistics on the final population
    energies = states_energies["energy"]
    benergies = states_energies["benergy"]
    population = states_energies["state"]
    magnetizations = [compute_magnetization(p) for p in population]
    return {
        "population": population,
        "energies": energies,
        "benergies": benergies,
        "magnetizations": magnetizations,
        "n": n,
        "beta": beta64,
        "h": h64,
        "n_states": n_states,
        "n_children": n_children,
        "n_cycles": n_cycles,
        "mutation_rate": mutation_rate,
    }
