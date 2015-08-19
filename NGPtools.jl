module NGPtools

import Distributions.MvNormal
import SStools.simulate

export assemble_matrices, generate, sample

"""
Observation matrix for state space model. In this case, we assume that the
Np components of the observation vector each track the state of the same 
index (more specifically, its U component). All other terms are ignored.
Np is the dimension of the observation vector.
Ns is the number of underlying states.
"""
function Z(Np, Ns)
    maxind = min(3Ns, Np)
    Zmat = sparse(1:maxind, 1:maxind, 1, Np, 3Ns)
    return Zmat
end


"""
State transition matrix. Defined in (5) of Zhu and Dunson. Order of entries
is all state variables, followed by all derivatives, then all latents.
Ns is the dimension of the underlying state space.
"""
function G(Ns, δ; approx=false)
    gmat = eye(3 * Ns)
    gmat += diagm(δ * ones(2Ns), Ns)
    if approx
        gmat += diagm(δ^2/2 * ones(Ns), 2Ns)
    end

    return sparse(gmat)
end


"""
Covariance of state disturbances. Defined in (5) of Zhu and Dunson. Order of
entries is all state variables, followed by all derivatives, then all latents.
"""
function W(Ns, δ, varU::Array{Float64}, varA::Array{Float64}; approx=false)
    if approx
        wmat = δ * spdiagm([varU ; varA])
    else
        wmat = zeros(Float64, 3Ns, 3Ns)
        for idx in 1:Ns
            wmat[idx, idx] = (δ^3 / 3)varU[idx] + (δ^5 / 20)varA[idx]
            wmat[idx, Ns + idx] = (δ^2 / 2)varU[idx] + (δ^4 / 8)varA[idx]
            wmat[idx, 2Ns + idx] = (δ^3 / 6)varA[idx]
            wmat[Ns + idx, idx] = wmat[idx, Ns + idx]
            wmat[Ns + idx, Ns + idx] = δ * varU[idx] + (δ^3 / 3)varA[idx]
            wmat[Ns + idx, 2Ns + idx] = (δ^2 / 2)varA[idx]
            wmat[2Ns + idx, idx] = wmat[idx, 2Ns + idx]
            wmat[2Ns + idx, Ns + idx] = wmat[Ns + idx, 2Ns + idx]
            wmat[2Ns + idx, 2Ns + idx] = δ * varA[idx]
        end
    end

    return wmat
end

function W(Ns, δ, varU::Array{Float64}, varA::Float64; approx=false)
    return W(Ns, δ, varU, varA * ones(Ns); approx=approx)
end

function W(Ns, δ, varU::Float64, varA::Array{Float64}; approx=false)
    return W(Ns, δ, varU * ones(Ns), varA; approx=approx)
end

function W(Ns, δ, varU::Float64, varA::Float64; approx=false)
    return W(Ns, δ, varU * ones(Ns), varA * ones(Ns); approx=approx)
end


"""
Take nested GP parameters and return matrices suitable for feeding into
state space model.

Np is the dimension of the observation vector
Ns is the number of underlying states
δ is a vector of differences between observation times. If observations are 
uniformly spaced, δ can be a scalar or a length-1 vector.
"""
function assemble_matrices(Np, Ns, δ, σϵ, σU, σA, σμ, σα; approx=false)
    Nt = length(δ)  # number of time points

    # calculate dimensions
    # Note: for an observation vector of size Np, we introduce 3 latent
    # variables per observation and 2 or 3 noise variables (depending on 
    # approximation)
    Nm = 3 * Ns
    if approx
        Nr = 2 * Ns
    else
        Nr = 3 * Ns
    end

    # preallocate arrays:
    H = σϵ^2 * eye(Ns)
    T = Array(Float64, Nm, Nm, Nt)
    Q = Array(Float64, Nr, Nr, Nt)

    if approx
        R = sparse([2:3:Nm ; 3:3:Nm], [1:2:Nr ; 2:2:Nr], 1., Nm, Nr)
    else
        R = speye(Float64, Nm)
    end

    for t in 1:Nt
        T[:, :, t] = G(Ns, δ[t], approx=approx)
        Q[:, :, t] = W(Ns, δ[t], σU.^2, σA.^2, approx=approx)
    end

    if Nt == 1
        T = sparse(squeeze(T, 3))
        Q = squeeze(Q, 3)
        if approx
            Q = sparse(Q)
        end
    end

    a0 = zeros(Nm)
    P0 = diagm(repeat([σμ, σμ, σα], inner=[Ns]))

    return Z(Np, Ns), H, T, R, Q, a0, P0
end


"""
Generate nGP data according to the state space model. Uses matrices
constructed in assemble_matrices.

Np is the dimension of the observation vector
Ns is the number of underlying states
δ is a spacing between observations
"""
function generate(Np, Ns, Nt, δ, σϵ, σU, σA, σμ, σα; approx=false)

    Z, H, T, R, Q, a0, P0 = assemble_matrices(Np, Ns, δ, σϵ, σU, σA, σμ, σα, approx=approx)

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
"""
function _sample(y, Ns, a0, P0, Z, H, T, R, Q; interleaved=true)
    Nt = size(y, 2)
    Nm = size(Z, 2)

    α_samples = Array(Float64, Nm, Nt, Ns)

    for idx in 1:Ns
        α_samples[:, :, idx] = simulate(y, a0, P0, Z, H, T, R, Q, interleaved=interleaved)
    end

    return α_samples

end

"""
Version where spacing is uniform, so number of time points must be specified 
explicitly.
"""
function sample(y, Nsamples, Nstates, δ, σϵ, σU, σA, σμ, σα; approx=false)
    Np = size(y, 1)
    Z, H, T, R, Q, a0, P0 = assemble_matrices(Np, Nstates, δ, σϵ, σU, σA, σμ, σα, approx=approx)
    _sample(y, Nsamples, a0, P0, Z, H, T, R, Q)
end


"""
Version where observation model (Z and H) is specified externally.
"""
function sample(y, Nsamples, Z::Array{Float64}, H::Array{Float64}, δ::Float64, 
    σU, σA, σμ, σα; approx=false, interleaved=true)
    Nstates = convert(Int64, size(Z, 2) / 3)
    Np = size(y, 1)
    _, _, T, R, Q, a0, P0 = assemble_matrices(Np, Nstates, δ, 0, σU, σA, σμ, σα, approx=approx)
    _sample(y, Nsamples, a0, P0, Z, H, T, R, Q; interleaved=interleaved)
end

end  # module