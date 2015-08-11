module NGPtools

import Distributions.MvNormal
import SStools.simulate

export assemble_matrices, generate, sample

"""
Observation matrix for state space model. In this case, simply picks out
the first entry of the state space vector, which is U.
"""
function Z(δ, σU, σA)
    return reshape([1., 0., 0.], 1, 3, 1)::Array{Float64}
end


"""
State transition matrix. Defined in (5) of Zhu and Dunson.
"""
function G(δ, σU, σA, approx=false)
    gmat = eye(3)
    gmat[1, 2] = δ
    gmat[2, 3] = δ
    if !approx
        gmat[1, 3] = δ^2 / 2
    end

    return gmat
end

"""
Covariance of state disturbances. Defined in (5) of Zhu and Dunson.
"""
function W(δ, varU, varA, approx=false)
    if approx
        wmat = δ * diagm([varU, varA])
    else
        wmat = Array(Float64, 3, 3)
        wmat[1, 1] = (δ^3 / 3)varU + (δ^5 / 20)varA
        wmat[1, 2] = (δ^2 / 2)varU + (δ^4 / 8)varA
        wmat[1, 3] = (δ^3 / 6)varA
        wmat[2, 1] = wmat[1, 2]
        wmat[2, 2] = δ * varU + (δ^3 / 3)varA
        wmat[2, 3] = (δ^2 / 2)varA
        wmat[3, 1] = wmat[1, 3]
        wmat[3, 2] = wmat[2, 3]
        wmat[3, 3] = δ * varA
    end

    return wmat
end


"""
Take nested GP parameters and return matrices suitable for feeding into
state space model.

dims is a tuple (Np, Nm, Nr) of state space dimensions:
    (observation, state, noise)
"""
function assemble_matrices(dims, δ, σϵ, σU, σA, σμ, σα, approx=false)
    Nt = length(δ)  # number of time points
    Np, Nm, Nr = dims

    if approx
        Nr = 2
    end

    # allocate arrays:
    H = σϵ * ones(1, 1)
    T = Array(Float64, Nm, Nm, Nt)

    if approx
        R = zeros(Nm, Nr)
        R[2:end, :] = eye(2)
    else
        R = eye(Nr)
    end

    Q = Array(Float64, Nr, Nr, Nt)

    for t in 1:Nt
        T[:, :, t] = G(δ[t], σU, σA, approx)
        Q[:, :, t] = W(δ[t], σU^2, σA^2, approx)
    end

    a0 = zeros(Nm)
    P0 = diagm([σμ, σμ, σα])

    return Z(0, σU, σA), H, T, R, Q, a0, P0
end


"""
Generate nGP data according to the state space model. Uses matrices
constructed in assemble_matrices.

dims is a tuple (Np, Nm, Nr) of state space dimensions:
    (observation, state, noise)
"""
function generate(dims, δ, σϵ, σU, σA, σμ, σα, approx=false)
    Nt = length(δ)

    Np, Nm, Nr = dims

    if approx
        Nr = 2
    end

    Z, H, T, R, Q, a0, P0 = assemble_matrices(dims, δ, σϵ, σU, σA, σμ, σα, approx)

    α = Array(Float64, Nm, Nt)
    y = Array(Float64, Np, Nt)

    # initialize
    α[:, 1] = rand(MvNormal(a0, P0))

    # simulate
    for t in 1:Nt
        ϵ = H * randn()
        η = rand(MvNormal(Q[:, :, t]))
        y[:, t] = Z * α[:, t] + ϵ

        if t < Nt
            α[:, t + 1] = T[:, :, t] * α[:, t] + R * η
        end
    end

    return y, α
end


"""
Sample Ns times from the nGP posterior, given observations y.
"""
function sample(y, Ns, dims, δ, σϵ, σU, σA, σμ, σα, approx=false)
    Nt = length(δ)
    Np, Nm, _ = dims

    Z, H, T, R, Q, a0, P0 = assemble_matrices(dims, δ, σϵ, σU, σA, σμ, σα, approx)

    α_samples = Array(Float64, Nm, Nt, Ns)

    for idx in 1:Ns
        α_samples[:, :, idx] = simulate(y, a0, P0, Z, H, T, R, Q)
    end

    return α_samples

end


end  # module