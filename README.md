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