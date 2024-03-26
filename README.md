# Ising model

The Ising model in 2D models $N$ up and down spins according to the Hamiltonian

$$
    \beta H = - J \sum_{\langle ij \rangle} s_i s_j - h \sum_i s_i
$$

We are interested in computing the magnetization $\langle m \rangle = \frac{1}{N}\sum_i \langle s_i \rangle$, the specific heat $c_v = \frac{\beta^2}{N} \Bigl( \langle E^2 \rangle - \langle E \rangle^2\Bigr)$ and the magnetic susceptibility $\chi = \beta N \Bigl( \langle m^2 \rangle - \langle m \rangle^2\Bigr)$.

The critical value for the temperature should be $\beta_c \simeq 0.441$.

# Monte-Carlo method

In the MC method, we randomly flip spins and accept/reject the flip depending on a sampling scheme (Metropolis, Swendsen-Wang, Wolff). In short, the method will go as follows:

- Model $N$ spins in volume $V$ (square lattice, $N = n^2$) with periodic boundary conditions.
- Repeat the following loop for $M$ steps:
    
    1. Pick a random site $(i,j)$
    2. Flip the spin and compute the associated $\Delta H$
        - If $\Delta H <0$, accept the flip
        - Otherwise, accept with propability $p = e^{- \beta \Delta H}$

# Genetic Algorithm

In the GA method, we create a random population of states ($n \times n matrices$) with size $M$ on which we apply the following steps for $X$ cycles:
    
1. Create $Y$ children from random pairs of parents (their probability of choice is proportional to the fitness):
    - To create children, we divide each parent in two blocks using a random divider $s$ (so we have two sets of two blocks,of size $s \times n$ and $(n - s) \times m$). We cross-combine these blocks to create two new matrices.
    - We mutate each child by randomly flipping some of their spins per the mutation rate.
2. Add children to the population, compute their fitness score. The population is now of size $M + Y$.
3. Sort the population w.r.t. the fitness score and cutoff the $Y$ poorest members. The population is back to size $M$.

In order to include thermal fluctuations, we need to compute the Gibbs free energy $G = U - T S$ where $U$ is the internal energy and $S$ the entropy. The internal energy of a given configuration is obtained by looking computing its Hamiltonian. The entropy is more delicate: is it a statistical property and usually requires the computation of the entire partition function. Instead, we approximate it by its cumulants following these two papers \[[1](https://doi.org/10.1142/S0217984905008153), [2](https://doi.org/10.1016/S0378-4371%2802%2901327-4)\].