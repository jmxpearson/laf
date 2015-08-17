module NGPtools

import Distributions.MvNormal
import SStools.simulate

export assemble_matrices, generate, sample

"""
Observation matrix for state space model. In this case, simply picks out
the first entry of the state space vector, which is U.
"""
function Z(Np)
    Zmat = sparse(1:Np, 1:Np, 1, Np, 3Np)
    return Zmat
end


"""
State transition matrix. Defined in (5) of Zhu and Dunson. Order of entries
is all state variables, followed by all derivatives, then all latents.
"""
function G(Np, δ, σU, σA; approx=false)
    gmat = eye(3 * Np)
    gmat += diagm(δ * ones(2Np), Np)
    if approx
        gmat += diagm(δ^2/2 * ones(Np), 2Np)
    end

    return sparse(gmat)
end


"""
Covariance of state disturbances. Defined in (5) of Zhu and Dunson. Order of
entries is all state variables, followed by all derivatives, then all latents.
"""
function W(Np, δ, varU, varA; approx=false)
    if approx
        wmat = δ * spdiagm(repeat([varU, varA], inner=[Np]))
    else
        wmat = zeros(Float64, 3Np, 3Np)
        for idx in 1:Np
            wmat[idx, idx] = (δ^3 / 3)varU + (δ^5 / 20)varA
            wmat[idx, Np + idx] = (δ^2 / 2)varU + (δ^4 / 8)varA
            wmat[idx, 2Np + idx] = (δ^3 / 6)varA
            wmat[Np + idx, idx] = wmat[idx, Np + idx]
            wmat[Np + idx, Np + idx] = δ * varU + (δ^3 / 3)varA
            wmat[Np + idx, 2Np + idx] = (δ^2 / 2)varA
            wmat[2Np + idx, idx] = wmat[idx, 2Np + idx]
            wmat[2Np + idx, Np + idx] = wmat[Np + idx, 2Np + idx]
            wmat[2Np + idx, 2Np + idx] = δ * varA
        end
    end

    return wmat
end


"""
Take nested GP parameters and return matrices suitable for feeding into
state space model.

Np is the dimension of the observation vector
δ is a vector of differences between observation times. If observations are 
uniformly spaced, δ can be a scalar or a length-1 vector.
"""
function assemble_matrices(Np, δ, σϵ, σU, σA, σμ, σα; approx=false)
    Nt = length(δ)  # number of time points

    # calculate dimensions
    # Note: for an observation vector of size Np, we introduce 3 latent
    # variables per observation and 2 or 3 noise variables (depending on 
    # approximation)
    Nm = 3 * Np
    if approx
        Nr = 2 * Np
    else
        Nr = 3 * Np
    end

    # preallocate arrays:
    H = σϵ^2 * eye(Np)
    T = Array(Float64, Nm, Nm, Nt)
    Q = Array(Float64, Nr, Nr, Nt)

    if approx
        R = sparse([2:3:Nm ; 3:3:Nm], [1:2:Nr ; 2:2:Nr], 1., Nm, Nr)
    else
        R = speye(Float64, Nm)
    end

    for t in 1:Nt
        T[:, :, t] = G(Np, δ[t], σU, σA, approx=approx)
        Q[:, :, t] = W(Np, δ[t], σU^2, σA^2, approx=approx)
    end

    if Nt == 1
        T = sparse(squeeze(T, 3))
        Q = squeeze(Q, 3)
        if approx
            Q = sparse(Q)
        end
    end

    a0 = zeros(Nm)
    P0 = diagm(repeat([σμ, σμ, σα], inner=[Np]))

    return Z(Np), H, T, R, Q, a0, P0
end


"""
Generate nGP data according to the state space model. Uses matrices
constructed in assemble_matrices.

Np is the dimension of the observation vector
δ is a spacing between observations
"""
function generate(Np, Nt, δ, σϵ, σU, σA, σμ, σα; approx=false)

    Z, H, T, R, Q, a0, P0 = assemble_matrices(Np, δ, σϵ, σU, σA, σμ, σα, approx=approx)

    Nm = size(Z, 2)

    α = Array(Float64, Nm, Nt)
    y = Array(Float64, Np, Nt)

    # initialize
    α[:, 1] = rand(MvNormal(full(a0), full(P0)))

    # simulate
    for t in 1:Nt
        ϵ = rand(MvNormal(H))
        local Q_t = ndims(Q) < 3 ? Q : Q[:, :, t]
        η = rand(MvNormal(Q_t))
        y[:, t] = Z * α[:, t] + ϵ

        if t < Nt
            local T_t = ndims(T) < 3 ? T : slice(T, :, :, t)
            α[:, t + 1] = T_t * α[:, t] + R * η
        end
    end

    return y, α
end


"""
Sample Ns times from the nGP posterior, given observations y.
Np is the dimension of the observation vector
"""
function _sample(y, Ns, Nt, a0, P0, Z, H, T, R, Q)
    Nm = size(Z, 2)

    α_samples = Array(Float64, Nm, Nt, Ns)

    for idx in 1:Ns
        α_samples[:, :, idx] = simulate(y, a0, P0, Z, H, T, R, Q)
    end

    return α_samples

end

"""
Version where spacing is uniform, so number of time points must be specified 
explicitly.
"""
function sample(y, Ns, Np, Nt, δ::Float64, σϵ, σU, σA, σμ, σα; approx=false)
    Z, H, T, R, Q, a0, P0 = assemble_matrices(Np, δ, σϵ, σU, σA, σμ, σα, approx=approx)
    _sample(y, Ns, Nt, a0, P0, Z, H, T, R, Q)
end


"""
Version where spacing is non-uniform, so δ is a vector and number of time
points follows from that.
"""
function sample(y, Ns, Np, δ::Vector{Any}, σϵ, σU, σA, σμ, σα; approx=false)
    Nt = length(δ)
    Z, H, T, R, Q, a0, P0 = assemble_matrices(Np, δ, σϵ, σU, σA, σμ, σα, approx=approx)
    _sample(y, Ns, Nt, a0, P0, Z, H, T, R, Q)
end


"""
Version where observation model (Z and H) is specified externally.
"""
function sample(y, Ns, Np, Z::Matrix{Any}, H::Matrix{Any}, δ::Float64, 
    σϵ, σU, σA, σμ, σα; approx=false)
    _, _, T, R, Q, a0, P0 = assemble_matrices(Np, δ, σϵ, σU, σA, σμ, σα, approx=approx)
    _sample(y, Ns, Nt, a0, P0, Z, H, T, R, Q)
end

end  # module