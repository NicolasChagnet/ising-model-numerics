import functools

import numba as nb
import numpy as np
import numpy.random as rand
from numba import float64, int64, jit, njit, prange, void


@njit([int64[:, :](int64)])
def create_random_state(n):
    """Returns a random matrix of spins up and down

    Args:
        n (int): Size of the matrix

    Returns:
        np.ndarray: Matrix of shape (n,n) and random values in +1/-1.
    """
    return 2 * rand.randint(0, 2, (n, n)) - 1


@njit([int64[:, :](int64, int64)])
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
            return create_random_state(n)


@njit([float64(int64, int64, int64[:, :], float64)])
def compute_hamiltonian_term(x: int, y: int, lattice: np.ndarray, h: np.float64):
    """Computes the Hamiltonian at a lattice site i,j given a lattice state as well as parameters values.

    Args:
        x (int): row of lattice site
        y (int): column of lattice site
        lattice (ndarray): lattice of spins in a given state.
        h (float64): External field coupling.

    Returns:
        float64: Value of the Hamiltonian at site i,j for this configuration.
    """
    n = lattice.shape[0]
    si = lattice[x, y]
    neighbors = [((x + 1) % n, y), ((x - 1) % n, y), (x, (y + 1) % n), (x, (y - 1) % n)]
    term_sum_neighbors = np.float64(0.0)
    for neighbor in neighbors:
        term_sum_neighbors += si * lattice[neighbor]
    # Remove the extra factor of 4 due to duplicate counting of pairs
    term_sum_neighbors = term_sum_neighbors / 2
    Hi = -term_sum_neighbors - h * si
    return Hi


@njit([float64(int64[:, :], float64)])
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
    for x in range(n):
        for y in range(n):
            H += compute_hamiltonian_term(x, y, lattice, h)
    return H


@njit([float64(int64[:, :])])
def compute_magnetization(lattice: np.ndarray):
    """Computes the magnetization for a given lattice configuration

    Args:
        lattice (ndarray): Configuration of up and down spins

    Returns:
        float: Average magnetization of the configuration
    """
    N = np.int64(lattice.size)
    return np.sum(lattice) / N


@njit(
    [float64(int64, int64, int64[:, :], float64)],
)
def compute_deltaH(x, y, lattice: np.ndarray, h: np.float64):
    """Compute the variation of the Hamiltonian given a spin flip

    Args:
        x (int): Index of spin to flip (row)
        y (int): Index of spin to flip (column)
        lattice (np.ndarray): Spin configuration
        h (float): External coupling

    Returns:
        float: Hamiltonian change
    """
    n = lattice.shape[0]
    si = lattice[x, y]
    neighbors = [((x + 1) % n, y), ((x - 1) % n, y), (x, (y + 1) % n), (x, (y - 1) % n)]
    sum_nn = sum([lattice[nn] for nn in neighbors])
    deltaH = (-2) * (-(si) * sum_nn - si * h)
    return deltaH


@njit(
    [float64(int64, int64, int64[:, :])],
)
def compute_deltaM(x: int, y: int, lattice: np.ndarray):
    """Compute the variation of the magnetization given a spin flip

    Args:
        x (int): Index of spin to flip (row)
        y (int): Index of spin to flip (column)
        lattice (np.ndarray): Spin configuration

    Returns:
        float: Magnetization change
    """
    N = lattice.size
    return -2 * lattice[x, y] / N


@njit(
    [void(int64, int64, int64[:, :], float64[:], float64[:], int64, int64, float64, float64)],
)
def update_lattice(
    x: int,
    y: int,
    lattice: np.ndarray,
    energies: list,
    magnetizations: list,
    i: int,
    n: int,
    beta: np.float64,
    h: np.float64,
):
    """Update step for the Metropolic MC algorithm (in-place)

    Args:
        x (int): Index of spin to flip (row)
        y (int): Index of spin to flip (column)
        lattice (ndarray): List of lattices to update
        energies (list): List of energies to update
        magnetizations (list): List of magnetizations to update
        i (int): index of current iteration
        n (int): Length of the square lattice
        h (float): External coupling
    """
    # Since the initial state is manually set, we make here a rule to trivially exit
    if i == 0:
        return None
    # Compute the new energy and the energy difference
    deltaH = compute_deltaH(x, y, lattice, h)
    energy_flipped = energies[i - 1] + deltaH
    mag_flipped = magnetizations[i - 1] + compute_deltaM(x, y, lattice)

    # We now decide whether to accept the change or not
    accept = True
    if deltaH > 0:
        # If the energy variation is positive, we only accept under a probability threshold given by Boltzmannian weight
        probability_threshold = np.exp(-beta * deltaH)
        accept = rand.rand() < probability_threshold

    # If accepted, we flip the sign
    if accept:
        lattice[x, y] *= -1
        magnetizations[i] = mag_flipped
        energies[i] = energy_flipped
    else:
        magnetizations[i] = magnetizations[i - 1]
        energies[i] = energies[i - 1]


@njit([void(int64, int64[:, :], float64, float64, float64[:], float64[:])])
def random_method(i, lattice, beta64, h64, energies, magnetizations):
    """This method picks a random spin N = n^2 times and updates the lattice

    Args:
        i (int): Index of Monte-Carlo step
        lattice (np.ndarray): _description_
        beta64 (np.float64): _description_
        h64 (np.float64): _description_
        flips (list): _description_
        energies (list): _description_
        magnetizations (list): _description_
    """
    n = lattice.shape[0]
    x, y = (int(rand.rand() * n), int(rand.rand() * n))
    update_lattice(x, y, lattice, energies, magnetizations, i, n, beta64, h64)


@njit([int64[:, :](int64)])
def get_mask_even(n):
    """Returns an (n,n) matrix filled with 0,1 in a checkerboard pattern (0 on top-left cell)

    Args:
        n (int): Shape of matrix
    Returns:
        np.ndarray: Mask matrix
    """
    mask_basic = np.indices((n, n)).sum(axis=0) % 2
    return mask_basic


@njit([int64[:, :](int64)])
def get_mask_odd(n):
    """Returns an (n,n) matrix filled with 0,1 in a checkerboard pattern (1 on top-left cell)

    Args:
        n (int): Shape of matrix
    Returns:
        np.ndarray: Mask matrix
    """
    mask_basic = np.indices((n, n)).sum(axis=0) % 2
    return 1 - mask_basic


def checkerboard(i, lattice, beta64, h64, energies, magnetizations, mask_even, mask_odd):
    """This method updates the N spins in two batches of N/2 in a checkerboard pattern allowing for parallelization

    Args:
        i (int): Index of Monte-Carlo step
        lattice (np.ndarray): _description_
        beta64 (np.float64): _description_
        h64 (np.float64): _description_
        energies (list): _description_
        magnetizations (list): _description_
    """
    # We extract the checkerboard sublattice matching the parity of the iteration counter
    n = lattice.shape[0]
    sublattice = (i - 1) % 2
    mask = mask_even if sublattice == 0 else mask_odd
    # We compute the energy and magnetization variations for each spin flips
    sum_nn_terms = (
        np.roll(lattice, 1, axis=0)
        + np.roll(lattice, -1, axis=0)
        + np.roll(lattice, 1, axis=1)
        + np.roll(lattice, -1, axis=1)
    )
    neighbour_contributions = -lattice * (sum_nn_terms) - h64 * lattice
    delta_contributions_flipped = -2 * neighbour_contributions
    mag_flipped = -2 * lattice / n**2

    # Generate random flips and compare to boltzmann weights
    random_flips = rand.rand(n, n)
    boltzmann_weights = np.exp(-beta64 * delta_contributions_flipped)
    random_mask = random_flips < boltzmann_weights

    accepted_flips = mask * ((delta_contributions_flipped < 0) | ((delta_contributions_flipped > 0) & random_mask))

    lattice *= 1 - 2 * accepted_flips
    magnetizations[i] = magnetizations[i - 1] + np.sum(mag_flipped * accepted_flips)
    energies[i] = energies[i - 1] + np.sum(delta_contributions_flipped * accepted_flips)


def monte_carlo_metropolis(
    n: int, beta: float, h: float, max_steps=100, initial_state=0, method="checkerboard", store_history=False
):
    """General Monte Carlo simulation of Ising spins for a given number of spins, inverse temperature, spin coupling and external coupling.

    Args:
        n (int): Number of spins in one dimension (total spins N = n^2).
        beta (float): Inverse temperature.
        h (float): External field coupling.
        max_steps (int): Maximal number of updates (in volume size units). Defaults to 1000.
        initial_state (int, optional): Defines the initial state. Accepted values are 0 (hot, random state) or +1/-1. Defaults to 0.
        method (str, optional): Method to use ("random", "checkerboard"). Defaults to checkerboard.

    Returns:
        dict: Dictionary containing a list of spin configuration and observables, as well as input parameters.
        Keys are "lattices", "energies", "magnetizations", "n", "J", "beta", "h", "max_steps".
    """
    allowed_methods = ["random", "checkerboard"]
    if method not in allowed_methods:
        raise ValueError(f"Unknown method! Allowed values are: {allowed_methods}")
    if method == "checkerboard" and (n % 2) == 1:
        raise ValueError("The length of the lattice in checkerboard method should be an even integer!")
    # Convert inputs into numpy floats
    h64 = np.float64(h)
    beta64 = np.float64(beta)
    # Define method specific variables
    match method:
        case "random":
            loop_steps = max_steps * n**2 + 1
        case "checkerboard":
            loop_steps = 2 * max_steps + 1
            mask_even, mask_odd = get_mask_even(n), get_mask_odd(n)
    # Store outputs of the program
    energies = np.zeros(loop_steps, dtype=np.float64)
    magnetizations = np.zeros(loop_steps, dtype=np.float64)
    lattice_history = np.zeros((loop_steps, n, n))

    # We start by initializing the lattice
    lattice_init = create_lattice(n, state=initial_state)
    lattice = lattice_init.copy()
    lattice_history[0] = lattice_init.copy()
    energies[0] = compute_hamiltonian_from_site(lattice, h64)
    magnetizations[0] = compute_magnetization(lattice)

    for i in range(1, loop_steps):
        match method:
            case "random":
                random_method(i, lattice, beta64, h64, energies, magnetizations)
            case "checkerboard":
                checkerboard(i, lattice, beta64, h64, energies, magnetizations, mask_even, mask_odd)
        if store_history:
            lattice_history[i] = lattice.copy()

    benergies = [beta64 * energy for energy in energies]

    object_return = {
        "lattice": lattice,
        "lattice_init": lattice_init,
        "benergies": benergies,
        "magnetizations": magnetizations,
        "n": n,
        "beta": beta64,
        "h": h64,
        "number_steps": loop_steps,
        "time": np.arange(1, loop_steps + 1),
        "lattice_history": lattice_history,
    }

    return object_return
