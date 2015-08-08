module SStools

import Distributions.MvNormal

export kalman_filter, fast_state_smoother, simulate

"""
Perform Kalman filtering on the data y.
Conventions are as in Durbin and Koopman (2012).
Relevant dimensions are:
- Nt: number of time points
- Np: dimension of observation space
- Nm: dimension of state space
- Nr: dimension of state noise covariance

Parameters:
- a0: mean of prior on states: (Nm, 1) or (Nm,)
- P0: variance of prior on states: (Nm, Nm)
- Z: either a (Np, Nm, Nt) array or a (Np, Nm) array
- H: either a (Np, Np, Nt) array or a (Np, Np) array
- T: either a (Nm, Nm, Nt) array or a (Nm, Nm) array
- R: either a (Nm, Nr, Nt) array or a (Nm, Nr) array
- Q: either a (Nr, Nr, Nt) array or a (Nr, Nr) array

Returns:
- v: (Np, Nt) vector of residuals at each time
- K: (Nm, Np, Nt) Kalman gain matrix
- Finv: (Np, Np, Nt) inverse of prediction variance matrix
"""
function kalman_filter(y, a0, P0, Z, H, T, R, Q)
    Np, Nt = size(y)
    Nm = size(P0, 1)

    # preallocate arrays
    a = Array(Float64, Nm, Nt)
    P = Array(Float64, Nm, Nm, Nt)
    v = Array(Float64, Np, Nt)
    Finv = Array(Float64, Np, Np, Nt)
    K = Array(Float64, Nm, Np, Nt)

    # initialize
    a[:, 1] = a0;
    P[:, :, 1] = P0;

    # iterate
    for t in 1:Nt
        local a_t = a[:, t]
        local P_t = P[:, :, t]
        local y_t = y[:, t]
        local Z_t = Z[:, :, min(t, size(Z, 3))]
        local H_t = H[:, :, min(t, size(H, 3))]
        local T_t = T[:, :, min(t, size(T, 3))]
        local R_t = R[:, :, min(t, size(R, 3))]
        local Q_t = Q[:, :, min(t, size(Q, 3))]

        v_t = y_t - Z_t * a_t
        F_t = Z_t * P_t * Z_t' + H_t
        Finv_t = inv(F_t)
        K_t = T * P_t * Z_t' * Finv_t
        L_t = T_t - K_t * Z_t
        
        if t < Nt
            a[:, t + 1] = T * a_t + K_t * v_t
            P[:, :, t + 1] = T * P_t * L_t' + R_t * Q_t * R_t'
        end

        v[:, t] = v_t
        K[:, :, t] = K_t
        Finv[:, :, t] = Finv_t
    end

    return v, K, Finv
end

"""
Runs an efficient smoothing algorithm to get posterior mean of state
vector.

Parameters are as in kalman_filter.
"""
function fast_state_smoother(v, K, Finv, a0, P0, Z, T, R, Q)
    # infer dimensions
    Np, Nt = size(v)
    Nm = size(P0, 1) 

    # preallocate
    r0 = 0.
    r = Array(Float64, Nm, Nt)
    r[:, end] = 0
    alpha = Array(Float64, Nm, Nt);

    # iterate backward
    for t in Nt:-1:1
        local Finv_t = Finv[:, :, t]
        local v_t = v[:, t]
        local K_t = K[:, :, t]
        local r_t = r[:, t]
        local Z_t = Z[:, :, min(t, size(Z, 3))]
        
        local u = Finv_t * v_t - K_t' * r_t
        local thisr = Z_t' * u + T' * r_t
        
        if t - 1 > 0
            r[:, t - 1] = thisr
        else
            r0 = thisr
        end
    end

    # run model forward
    alpha[:, 1] = a0 + P0 * r0

    for t in 1:(Nt - 1)
        local T_t = T[:, :, min(t, size(T, 3))]
        local R_t = R[:, :, min(t, size(R, 3))]
        local Q_t = Q[:, :, min(t, size(Q, 3))]
        local RQR = R_t * Q_t * R_t'

        alpha[:, t + 1] = T_t * alpha[:, t] + RQR * r[:, t]
    end

    return alpha
end

"""
Draw a sample of the state trajectory from the smoothed posterior.
"""
function simulate(y, a0, P0, Z, H, T, R, Q)
    # get dimensions
    Np, Nt = size(y)
    Nm = size(a0, 1)
    Nr = size(Q, 1)

    # preallocate
    alpha_plus = Array(Float64, Nm, Nt)
    y_plus = Array(Float64, Np, Nt)

    # simulate data
    alpha_plus[:, 1] = rand(MvNormal(a0, P0))

    for t in 1:Nt

        local Z_t = Z[:, :, min(t, size(Z, 3))]
        local H_t = H[:, :, min(t, size(H, 3))]
        local T_t = T[:, :, min(t, size(T, 3))]
        local R_t = R[:, :, min(t, size(R, 3))]
        local Q_t = Q[:, :, min(t, size(Q, 3))]

        # draw disturbances
        ϵ = rand(MvNormal(H_t))
        y_plus[:, t] = Z_t * alpha_plus[:, t] + ϵ

        if t < Nt
            η = rand(MvNormal(Q_t))
            alpha_plus[:, t + 1] = T_t * alpha_plus[:, t] + R_t * η
        end
    end

    # calculate smoothed means:

    # actual data
    v, K, Finv = kalman_filter(y, a0, P0, Z, H, T, R, Q)
    alpha_hat = fast_state_smoother(v, K, Finv, a0, P0, Z, T, R, Q)

    # simulated data
    v, K, Finv = kalman_filter(y_plus, a0, P0, Z, H, T, R, Q)
    alpha_hat_plus = fast_state_smoother(v, K, Finv, a0, P0, Z, T, R, Q)

    # combine
    alpha_draw = alpha_plus - alpha_hat_plus + alpha_hat

    return alpha_draw
end

end  # module