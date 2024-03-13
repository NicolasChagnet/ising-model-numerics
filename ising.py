import numba as nb
import numpy as np
from numba import jit, njit


def create_lattice(n, state=0):
    """Creates a numpy lattice of N spins.

    Args:
        n (int): Number of spins in one dimension (total spins N = n^2).
        state (int, optional): Defines the initial state. Accepted values are 0 (hot, random state) or +1/-1 (all up, down resp.). Defaults to 0.

    Returns:
        ndarray: Numpy ndarray of spins (+1 or -1 at every site) with shape (n,n).
    """
    match state:
        case 1:
            return np.ones((n, n), dtype=np.int64)
        case -1:
            return -1 * np.ones((n, n), dtype=np.int64)
        case _:
            return np.random.choice(np.array([-1, 1], dtype=np.int64), (n, n))

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

@jit
def compute_magnetization(lattice: np.ndarray):
    """Computes the magnetization for a given lattice configuration

    Args:
        lattice (ndarray): Configuration of up and down spins

    Returns:
        float: Average magnetization of the configuration
    """
    N = np.int64(lattice.size)
    return np.sum(lattice) / N

def compute_deltaH(spin: tuple, lattice: np.ndarray, h: np.float64):
    n = lattice.shape[0]
    i, j = spin
    si = lattice[spin]
    neighbors = [((i + 1) % n, j), ((i - 1) % n, j), (i, (j + 1) % n), (i, (j - 1) % n)]
    sum_nn = sum([lattice[nn] for nn in neighbors])
    deltaH = (-2) * (-(si) * sum_nn - si * h)
    return deltaH

def compute_deltaM(spin: tuple, lattice: np.ndarray):
    N = lattice.size
    return (- 2 * lattice[spin] / N)

def update_lattice(spin: tuple, lattice: np.ndarray, flips: list, energies: list, magnetizations: list, i: int,
                   n: int, beta: np.float64, h: np.float64):
    """Update step for the Metropolic MC algorithm (in-place)

    Args:
        lattice (ndarray): List of lattices to update
        flips (list): List of flips to update and whether they are accepted
        energies (list): List of energies to update
        magnetizations (list): List of magnetizations to update
        i (int): index of current iteration
        n (int): Length of the square lattice
        h (float): External coupling
    """
    # Compute the new energy and the energy difference
    deltaH = compute_deltaH(spin, lattice, h)
    energy_flipped = energies[i - 1] + deltaH
    mag_flipped = magnetizations[i - 1] + compute_deltaM(spin, lattice)

    # We now decide whether to accept the change or not
    accept = True
    if deltaH > 0:
        # If the energy variation is positive, we only accept under a probability threshold given by Boltzmannian weight
        probability_threshold = np.exp(-beta * deltaH)
        accept = np.random.rand() < probability_threshold

    # If accepted, we flip the sign
    if accept:
        lattice[spin] *= -1
        magnetizations[i] = mag_flipped
        energies[i] = energy_flipped
    else:
        magnetizations[i] = magnetizations[i - 1]
        energies[i] = energies[i - 1]
    flips[i] = (accept, spin)


def monte_carlo_metropolis(n, beta: float, h: float, max_steps=100, initial_state=0, method="random"):
    """General Monte Carlo simulation of Ising spins for a given number of spins, inverse temperature, spin coupling and external coupling.

    Args:
        n (int): Number of spins in one dimension (total spins N = n^2).
        beta (float): Inverse temperature.
        h (float): External field coupling.
        max_steps (int): Maximal number of updates (in volume size units). Defaults to 1000.
        initial_state (int, optional): Defines the initial state. Accepted values are 0 (hot, random state) or +1/-1. Defaults to 0.
        method (str, optional): Method to use ("random", "sweep"). Defaults to sweep.

    Returns:
        dict: Dictionary containing a list of spin configuration and observables, as well as input parameters.
        Keys are "lattices", "energies", "magnetizations", "n", "J", "beta", "h", "max_steps".
    """
    # Convert inputs into numpy floats
    h64 = np.float64(h)
    beta64 = np.float64(beta)

    # Store outputs of the program
    loop_steps = max_steps * n**2
    flips = [None for i in range(loop_steps)]
    energies = [None for i in range(loop_steps)]
    magnetizations = [None for i in range(loop_steps)]

    # We start by initializing the lattice
    lattice_init = create_lattice(n, state=initial_state)
    lattice = lattice_init.copy()
    energies[0] = compute_hamiltonian_from_site(lattice, h64)
    magnetizations[0] = compute_magnetization(lattice)

    for i in range(max_steps):
        match method:
            case "random":
                for j in range(n ** 2):
                    if i == 0 and j == 0:
                        continue
                    # Choose a random spin
                    rand_spin = (
                        int(np.random.random() * n),
                        int(np.random.random() * n)
                    )
                    update_lattice(rand_spin, lattice, flips, energies, magnetizations, i * n**2 + j, n, beta64, h64)
            case "sweep":
                for u in range(n):
                    for v in range(n):
                        if i == 0 and u == 0 and v == 0:
                            continue
                        update_lattice((u, v), lattice, flips, energies, magnetizations, i * n**2 + u * n + v, n, beta64, h64)
            case _:
                raise ValueError("Unknown method selected!")

    benergies = [beta64 * energy for energy in energies]

    object_return = {
        "lattice": lattice,
        "lattice_init": lattice_init,
        "flips": flips,
        "benergies": benergies,
        "magnetizations": magnetizations,
        "n": n,
        "beta": beta64,
        "h": h64,
        "number_steps": loop_steps,
        "time": np.arange(1, loop_steps + 1)
    }

    return object_return
