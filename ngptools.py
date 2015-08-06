"""
ngptools.py implements functions useful for sampling from nested Gaussian
processes.

Currently, this is specialized to (m, n) = (2, 1) nested GPs.
"""

import numpy as np
import sstools as ss

def _Z(delta, sigU, sigA):
    """
    Observation matrix for state space model. In this case, simply picks out
    the first entry of the state vector, which is U.
    """
    return np.array([1.0, 0, 0])

def _G(delta, sigU, sigA, approx=False):
    """
    State transition matrix. Defined in (5) of Zhu and Dunson.
    """
    gmat = np.eye(3)
    gmat[0, 1] = delta
    gmat[1, 2] = delta
    if not approx:
        gmat[0, 2] = delta**2 / 2

    return gmat

def _W(delta, sigU, sigA, approx=False):
    """
    Covariance of state disturbances. Defined in (5) of Zhu and Dunson.
    """
    ssu = sigU**2
    ssa = sigA**2

    if approx:
        wmat = delta * np.diag([ssu, ssa])
    else:
        wmat = np.empty((3, 3))
        wmat[0, 0] = (delta**3 / 3) * ssu + (delta**5 / 20) * ssa
        wmat[0, 1] = (delta**2 / 2) * ssu + (delta**4 / 8) * ssa
        wmat[0, 2] = (delta**3 / 6) * ssa
        wmat[1, 0] = wmat[0, 1]
        wmat[1, 1] = delta * ssu + (delta**3 / 3) * ssa
        wmat[1, 2] = (delta**2 / 2) * ssa
        wmat[2, 0] = wmat[0, 2]
        wmat[2, 1] = wmat[1, 2]
        wmat[2, 2] = delta * ssa

    return wmat

def _assemble_matrices(dims, delta, sigeps, sigU, sigA, sigmu, sigalpha, 
    approx=False):
    """
    Take nested GP parameters and return matrices suitable for feeding into
    state space model.

    dims is a tuple (Np, Nm, Nr) of state space dimensions:
        (observation, state, noise)
    """
    Nt = delta.size  # number of time points
    Np, Nm, Nr = dims

    if approx:
        Nr = 2

    # allocate arrays:
    Z = _Z(0, sigU, sigA).reshape(1, -1)
    H = np.array(sigeps).reshape(Np, Np)
    T = np.empty((Nt, Nm, Nm))
    if approx:
        R = np.zeros((Nm, Nr))
        R[1:] = np.eye(2)
    else:
        R = np.eye(Nr)
    Q = np.empty((Nt, Nr, Nr))

    for t in range(Nt):
        T[t] = _G(delta[t], sigU, sigA, approx)
        Q[t] = _W(delta[t], sigU, sigA, approx)

    a_init = np.zeros(Nm)
    P_init = np.diag([sigmu, sigmu, sigalpha])

    return Z, H, T, R, Q, a_init, P_init

def generate(dims, delta, sigeps, sigU, sigA, sigmu, sigalpha, approx=False):
    """
    Generate nGP data according to the state space model. Uses matrices
    constructed in _assemble_matrices.

    dims is a tuple (Np, Nm, Nr) of state space dimensions:
        (observation, state, noise)
    """
    Nt = delta.size
    Np, Nm, Nr = dims

    if approx:
        Nr = 2

    Z, H, T, R, Q, a_init, P_init = _assemble_matrices(dims, delta,
        sigeps, sigU, sigA, sigmu, sigalpha, approx)

    alpha = np.empty((Nt, Nm))
    y = np.empty((Nt, Np))

    alpha[0] = np.random.multivariate_normal(a_init, P_init)

    for t in range(Nt):
        eps = np.random.multivariate_normal(np.zeros(Np), H)
        eta = np.random.multivariate_normal(np.zeros(Nr), Q[t])
        y[t] = Z.dot(alpha[t]) + eps
        if t + 1 < Nt:
            alpha[t + 1] = T[t].dot(alpha[t]) + R.dot(eta)

    return y, alpha

def sample(y, Nsamples, dims, delta, sigeps, sigU, sigA, sigmu, sigalpha, 
    approx=False):
    """
    Sample Nsamples times from the nGP posterior, given observations y.
    """

    Nt = delta.size
    Np, Nm, _ = dims

    Z, H, T, R, Q, a_init, P_init = _assemble_matrices(dims, delta,
        sigeps, sigU, sigA, sigmu, sigalpha, approx)

    alpha_samples = np.empty((Nsamples, Nt, Nm))

    for idx in range(Nsamples):
        alpha_samples[idx] = ss.simulate(y, a_init, P_init, Z, H, T, R, Q)

    return alpha_samples

