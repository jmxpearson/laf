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
        local a_t = slice(a, :, t)
        local P_t = slice(P, :, :, t)
        local y_t = slice(y, :, t)
        local Z_t = ndims(Z) < 3 ? Z : slice(Z, :, :, t)
        local H_t = ndims(H) < 3 ? H : slice(H, :, :, t)
        local T_t = ndims(T) < 3 ? T : slice(T, :, :, t)
        local R_t = ndims(R) < 3 ? R : slice(R, :, :, t)
        local Q_t = ndims(Q) < 3 ? Q : slice(Q, :, :, t)

        v_t = y_t - Z_t * a_t
        F_t = Z_t * P_t * Z_t' + H_t
        Finv_t = inv(F_t)
        K_t = T_t * P_t * Z_t' * Finv_t
        L_t = T_t - K_t * Z_t
        
        if t < Nt
            a[:, t + 1] = T_t * a_t + K_t * v_t
            P[:, :, t + 1] = T_t * P_t * L_t' + R_t * Q_t * R_t'
        end

        v[:, t] = v_t
        K[:, :, t] = K_t
        Finv[:, :, t] = Finv_t
    end

    return v, K, Finv, a, P
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
        local Finv_t = slice(Finv, :, :, t)
        local v_t = slice(v, :, t)
        local K_t = slice(K, :, :, t)
        local r_t = slice(r, :, t)
        local Z_t = ndims(Z) < 3 ? Z : slice(Z, :, :, t)
        local T_t = ndims(T) < 3 ? T : slice(T, :, :, t)
        
        local u = Finv_t * v_t - K_t' * r_t
        local thisr = Z_t' * u + T_t' * r_t
        
        if t - 1 > 0
            r[:, t - 1] = thisr
        else
            r0 = thisr
        end
    end

    # run model forward
    alpha[:, 1] = a0 + P0 * r0

    for t in 1:(Nt - 1)
        local T_t = ndims(T) < 3 ? T : slice(T, :, :, t)
        local R_t = ndims(R) < 3 ? R : slice(R,  :, :, t)
        local Q_t = ndims(Q) < 3 ? Q : slice(Q,  :, :, t)
        local RQR = R_t * Q_t * R_t'

        alpha[:, t + 1] = T_t * alpha[:, t] + RQR * r[:, t]
    end

    return alpha
end

"""
Draw a sample of the state trajectory from the smoothed posterior.
"""
function simulate(y, a0, P0, Z, H, T, R, Q; interleaved=true)
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
        local Z_t = ndims(Z) < 3 ? Z : slice(Z, :, :, t)
        local T_t = ndims(T) < 3 ? T : slice(T, :, :, t)
        local R_t = ndims(R) < 3 ? R : slice(R, :, :, t)
        # random number generators should take copies, not views?
        local H_t = ndims(H) < 3 ? H : H[:, :, t]
        local Q_t = ndims(Q) < 3 ? Q : Q[:, :, t]

        # draw disturbances
        local ϵ = rand(MvNormal(H_t))
        y_plus[:, t] = Z_t * alpha_plus[:, t] + ϵ

        if t < Nt
            η = rand(MvNormal(Q_t))
            alpha_plus[:, t + 1] = T_t * alpha_plus[:, t] + R_t * η
        end
    end

    # calculate smoothed means:
    if interleaved
        # actual data
        v, K, Finv, a, P = interleaved_kalman_filter(y, a0, P0, Z, H, T, R, Q)
        alpha_hat = interleaved_state_smoother(v, K, Finv, a0, P0, Z, T, R, Q)

        # simulated data
        v, K, Finv, a, P = interleaved_kalman_filter(y_plus, a0, P0, Z, H, T, R, Q)
        alpha_hat_plus = interleaved_state_smoother(v, K, Finv, a0, P0, Z, T, R, Q)

    else
        # actual data
        v, K, Finv, a, P = kalman_filter(y, a0, P0, Z, H, T, R, Q)
        alpha_hat = fast_state_smoother(v, K, Finv, a0, P0, Z, T, R, Q)

        # simulated data
        v, K, Finv, a, P = kalman_filter(y_plus, a0, P0, Z, H, T, R, Q)
        alpha_hat_plus = fast_state_smoother(v, K, Finv, a0, P0, Z, T, R, Q)
    end

    # combine
    alpha_draw = alpha_plus - alpha_hat_plus + alpha_hat

    return alpha_draw
end

"""
Like Kalman filter, but works by transforming a vector-valued 
observation into a sequence of scalar observations.
"""
function interleaved_kalman_filter(y, a0, P0, Z, H, T, R, Q)
    # test for diagonality of H
    # H (minus its diagonals) should be the 0 matrix
    if ndims(H) == 2
        assert(H == diagm(diag(H)))
    end

    Np, Nt = size(y)
    Nm = size(P0, 1)

    # preallocate
    a = Array(Float64, Nm, Np + 1, Nt)
    P = Array(Float64, Nm, Nm, Np + 1, Nt)
    v = Array(Float64, Np, Nt)
    Finv = Array(Float64, Np, Nt)
    K = Array(Float64, Nm, Np, Nt)

    # initialize
    a[:, 1, 1] = a0
    P[:, :, 1, 1] = P0

    for t in 1:Nt, i in 1:Np
        local Z_t = ndims(Z) < 3 ? Z : slice(Z, :, :, t)
        local H_t = ndims(H) < 3 ? H : slice(H, :, :, t)
        local T_t = ndims(T) < 3 ? T : slice(T, :, :, t)
        local R_t = ndims(R) < 3 ? R : slice(R, :, :, t)
        local Q_t = ndims(Q) < 3 ? Q : slice(Q, :, :, t)
        z = squeeze(Z_t[i, :], 1)  # now a column vector

        v[i, t] = y[i, t] - dot(z, a[:, i, t])
        F = dot(z, P[:, :, i, t] * z) + H_t[i, i]
        Finv[i, t] = 1. / F
        if F != 0
            K[:, i, t] = P[:, :, i, t] * z * Finv[i, t]
        else
            K[:, i, t] = 0
        end
        
        a[:, i + 1, t] = a[:, i, t] + K[:, i, t] * v[i, t]
        P[:, :, i + 1, t] = P[:, :, i, t] - F * (K[:, i, t] * K[:, i, t]')
        
        if i == Np && t < Nt
            a[:, 1, t + 1] = T_t * a[:, Np + 1, t]
            P[:, :, 1, t + 1] = T_t * P[:, :, Np + 1, t] * T_t' + R_t * Q_t * R_t'
        end
    end

    return v, K, Finv, a, P
end


function interleaved_state_smoother(v, K, Finv, a0, P0, Z, T, R, Q)
    # infer dimensions
    Np, Nt = size(v)
    Nm = size(P0, 1) 

    # preallocate
    r = Array(Float64, Nm, Np + 1, Nt)

    # initialize
    r[:, end, end] = 0

    # iterate backward
    for t in Nt:-1:1, i in (Np + 1):-1:2
        ii = i - 1  # handles offset between r and Z/K/F indices

        local Z_t = ndims(Z) < 3 ? Z : slice(Z, :, :, t)
        L = eye(Nm) - K[:, ii, t] * Z_t[ii, :] 
        z = squeeze(Z_t[ii, :], 1)  # now a column vector
        
        r[:, i - 1, t] = z * (v[ii, t] * Finv[ii, t]) + L' * r[:, i, t]
        
        if i == 2 && t > 1
            # NOTE: t - 1 BELOW !!!
            local T_t = ndims(T) < 3 ? T : slice(T, :, :, t - 1)
            r[:, Np + 1, t - 1] = T_t' * r[:, 1, t]
        end
    end

    # initialize again
    α = Array(Float64, Nm, Nt)
    α[:, 1] = a0 + P0 * r[:, 1, 1]

    for t in 1:(Nt - 1)
        local T_t = ndims(T) < 3 ? T : slice(T, :, :, t)
        local R_t = ndims(R) < 3 ? R : slice(R, :, :, t)
        local Q_t = ndims(Q) < 3 ? Q : slice(Q, :, :, t)
        local RQR = R_t * Q_t * R_t'

        α[:, t + 1] = T_t * α[:, t] + RQR * r[:, 1, t + 1]
    end

    return α
end

end  # module