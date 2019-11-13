# Aliases (types)
const FloatVector  = Array{Float64,1};
const FloatMatrix  = Array{Float64,2};
const FloatArray   = Array{Float64};
const SymMatrix    = Symmetric{Float64,Array{Float64,2}};
const DiagMatrix   = Diagonal{Float64,Array{Float64,1}};
const JVector{T}   = Array{Union{Missing, T}, 1};
const JArray{T, N} = Array{Union{Missing, T}, N};

#=
--------------------------------------------------------------------------------------------------------------------------------
Kalman structures
--------------------------------------------------------------------------------------------------------------------------------
=#

abstract type KalmanSettings end

"""
    ImmutableKalmanSettings(...)

Define an immutable structure that includes all the Kalman filter and smoother inputs.

# Model
The state space model used below is,

``Y_{t} = B*X_{t} + e_{t}``

``X_{t} = C*X_{t-1} + v_{t}``

Where ``e_{t} ~ N(0, R)`` and ``v_{t} ~ N(0, V)``.

# Arguments
- `Y`: Observed measurements (`nxT`)
- `B`: Measurement equations' coefficients
- `R`: Covariance matrix of the measurement equations' error terms
- `C`: Transition equations' coefficients
- `V`: Covariance matrix of the transition equations' error terms
- `X0`: Mean vector for the states at time t=0
- `P0`: Covariance matrix for the states at time t=0
- `n`: Number of series
- `T`: Number of observations
- `m`: Number of latent states
- `compute_loglik`: Boolean (true for computing the loglikelihood in the Kalman filter)
- `store_history`: Boolean (true to store the history of the filter and smoother)
"""
struct ImmutableKalmanSettings <: KalmanSettings
    Y::Union{FloatMatrix, JArray{Float64,2}}
    B::FloatMatrix
    R::SymMatrix
    C::FloatMatrix
    V::SymMatrix
    X0::FloatVector
    P0::SymMatrix
    n::Int64
    T::Int64
    m::Int64
    compute_loglik::Bool
    store_history::Bool
end

# ImmutableKalmanSettings constructor
function ImmutableKalmanSettings(Y::Union{FloatMatrix, JArray{Float64,2}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, V::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B,2);
    X0 = zeros(m);
    P0 = Symmetric(reshape((I-kron(C, C))\V[:], m, m));

    # Return ImmutableKalmanSettings
    return ImmutableKalmanSettings(Y, B, R, C, V, X0, P0, n, T, m, compute_loglik, store_history);
end

# ImmutableKalmanSettings constructor
function ImmutableKalmanSettings(Y::Union{FloatMatrix, JArray{Float64,2}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, V::SymMatrix, X0::FloatVector, P0::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B,2);

    # Return ImmutableKalmanSettings
    return ImmutableKalmanSettings(Y, B, R, C, V, X0, P0, n, T, m, compute_loglik, store_history);
end

"""
    MutableKalmanSettings(...)

Define a mutable structure identical in shape to ImmutableKalmanSettings.

See the docstring of ImmutableKalmanSettings for more information.
"""
mutable struct MutableKalmanSettings <: KalmanSettings
    Y::Union{FloatMatrix, JArray{Float64,2}}
    B::FloatMatrix
    R::SymMatrix
    C::FloatMatrix
    V::SymMatrix
    X0::FloatVector
    P0::SymMatrix
    n::Int64
    T::Int64
    m::Int64
    compute_loglik::Bool
    store_history::Bool
end

# MutableKalmanSettings constructor
function MutableKalmanSettings(Y::Union{FloatMatrix, JArray{Float64,2}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, V::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B,2);
    X0 = zeros(m);
    P0 = Symmetric(reshape((I-kron(C, C))\V[:], m, m));

    # Return MutableKalmanSettings
    return MutableKalmanSettings(Y, B, R, C, V, X0, P0, n, T, m, compute_loglik, store_history);
end

# MutableKalmanSettings constructor
function MutableKalmanSettings(Y::Union{FloatMatrix, JArray{Float64,2}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, V::SymMatrix, X0::FloatVector, P0::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B,2);

    # Return MutableKalmanSettings
    return MutableKalmanSettings(Y, B, R, C, V, X0, P0, n, T, m, compute_loglik, store_history);
end

"""
    KalmanStatus(...)

Define a mutable structure to manage the status of the Kalman filter and smoother.

# Arguments
- `t`: Current point in time
- `loglik`: Loglikelihood
- `X_prior`: Latest a-priori X
- `X_post`: Latest a-posteriori X
- `P_prior`: Latest a-priori P
- `P_post`: Latest a-posteriori P
- `history_X_prior`: History of a-priori X
- `history_X_post`: History of a-posteriori X
- `history_P_prior`: History of a-priori P
- `history_P_post`: History of a-posteriori P
"""
mutable struct KalmanStatus
    t::Int64
    loglik::Union{Float64, Nothing}
    X_prior::Union{FloatVector, Nothing}
    X_post::Union{FloatVector, Nothing}
    P_prior::Union{SymMatrix, Nothing}
    P_post::Union{SymMatrix, Nothing}
    history_X_prior::Union{Array{FloatVector,1}, Nothing}
    history_X_post::Union{Array{FloatVector,1}, Nothing}
    history_P_prior::Union{Array{SymMatrix,1}, Nothing}
    history_P_post::Union{Array{SymMatrix,1}, Nothing}
end

# KalmanStatus constructors
KalmanStatus() = KalmanStatus(0, [nothing for i=1:9]...);


#=
--------------------------------------------------------------------------------------------------------------------------------
UC models: structures
--------------------------------------------------------------------------------------------------------------------------------
=#

abstract type UCSettings end

"""
    ARIMASettings(...)

Define an immutable structure to manage ARIMA specifications.

# Arguments
- `Y_levels`: Observed measurements (`nxT`) - in levels
- `Y`: Observed measurements (`nxT`) - differenced and demeaned
- `d`: Degree of differencing
- `p`: Order of the autoregressive model
- `q`: Order of the moving average model
"""
struct ARIMASettings <: UCSettings
    Y_levels::Union{FloatMatrix, JArray{Float64,2}}
    Y::Union{FloatMatrix, JArray{Float64,2}}
    μ::Float64
    r::Int64
    d::Int64
    p::Int64
    q::Int64
end

# ARIMASettings constructor
function ARIMASettings(Y_levels::Union{FloatMatrix, JArray{Float64,2}}, d::Int64, p::Int64, q::Int64)

    # Differenciate data
    Y = copy(Y_levels);

    if d > 0
        for i=1:d
            Y = diff(Y, dims=2);
        end
    end

    μ = mean(Y);

    # Demean data
    Y = demean(Y);

    # ARIMASettings
    return ARIMASettings(Y_levels, Y, μ, max(p, q+1), d, p, q);
end
