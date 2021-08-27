# Aliases (types)
const IntVector    = Array{Int64,1};
const IntMatrix    = Array{Int64,2};
const IntArray     = Array{Int64};
const FloatVector  = Array{Float64,1};
const FloatMatrix  = Array{Float64,2};
const FloatArray   = Array{Float64};
const JVector{T}   = Array{Union{Missing, T}, 1};
const JMatrix{T}   = Array{Union{Missing, T}, 2};
const JArray{T, N} = Array{Union{Missing, T}, N};
const DiagMatrix   = Diagonal{Float64,Array{Float64,1}};
const SymMatrix    = Symmetric{Float64,Array{Float64,2}};

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

``X_{t} = C*X_{t-1} + D*U_{t}``

where ``e_{t} ~ N(0_{nx1}, R)`` and ``U_{t} ~ N(0_{mx1}, Q)``.

# Arguments
- `Y`: Observed measurements (`nxT`)
- `B`: Measurement equations' coefficients
- `R`: Covariance matrix of the measurement equations' error terms
- `C`: Transition equations' coefficients
- `D`: Transition equations' coefficients associated to the error terms
- `Q`: Covariance matrix of the transition equations' error terms
- `X0`: Mean vector for the states at time t=0
- `P0`: Covariance matrix for the states at time t=0
- `DQD`: Covariance matrix of D*U_{t} (i.e., D*Q*D')
- `n`: Number of series
- `T`: Number of observations
- `m`: Number of latent states
- `compute_loglik`: Boolean (true for computing the loglikelihood in the Kalman filter)
- `store_history`: Boolean (true to store the history of the filter and smoother)
"""
struct ImmutableKalmanSettings <: KalmanSettings
    Y::Union{FloatMatrix, JMatrix{Float64}}
    B::FloatMatrix
    R::SymMatrix
    C::FloatMatrix
    D::FloatMatrix
    Q::SymMatrix
    X0::FloatVector
    P0::SymMatrix
    DQD::SymMatrix
    n::Int64
    T::Int64
    m::Int64
    compute_loglik::Bool
    store_history::Bool
end

#=
ImmutableKalmanSettings constructors
=#

function ImmutableKalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, Q::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B, 2);
    D = Matrix(1.0I, m, m);
    X0 = zeros(m);
    P0 = solve_discrete_lyapunov(C, Q);

    # Return ImmutableKalmanSettings
    return ImmutableKalmanSettings(Y, B, R, C, D, Q, X0, P0, Q, n, T, m, compute_loglik, store_history);
end

function ImmutableKalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B, 2);
    X0 = zeros(m);
    DQD = Symmetric(D*Q*D');
    P0 = solve_discrete_lyapunov(C, DQD);

    # Return ImmutableKalmanSettings
    return ImmutableKalmanSettings(Y, B, R, C, D, Q, X0, P0, DQD, n, T, m, compute_loglik, store_history);
end

function ImmutableKalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatVector, P0::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B, 2);
    DQD = Symmetric(D*Q*D');

    # Return ImmutableKalmanSettings
    return ImmutableKalmanSettings(Y, B, R, C, D, Q, X0, P0, DQD, n, T, m, compute_loglik, store_history);
end

"""
    MutableKalmanSettings(...)

Define a mutable structure identical in shape to ImmutableKalmanSettings.

See the docstring of ImmutableKalmanSettings for more information.
"""
mutable struct MutableKalmanSettings <: KalmanSettings
    Y::Union{FloatMatrix, JMatrix{Float64}}
    B::FloatMatrix
    R::SymMatrix
    C::FloatMatrix
    D::FloatMatrix
    Q::SymMatrix
    X0::FloatVector
    P0::SymMatrix
    DQD::SymMatrix
    n::Int64
    T::Int64
    m::Int64
    compute_loglik::Bool
    store_history::Bool
end

#=
MutableKalmanSettings constructors
=#

function MutableKalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, Q::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B, 2);
    D = Matrix(1.0I, m, m);
    X0 = zeros(m);
    P0 = solve_discrete_lyapunov(C, Q);

    # Return MutableKalmanSettings
    return MutableKalmanSettings(Y, B, R, C, D, Q, X0, P0, Q, n, T, m, compute_loglik, store_history);
end

function MutableKalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B, 2);
    X0 = zeros(m);
    DQD = Symmetric(D*Q*D');
    P0 = solve_discrete_lyapunov(C, DQD);

    # Return MutableKalmanSettings
    return MutableKalmanSettings(Y, B, R, C, D, Q, X0, P0, DQD, n, T, m, compute_loglik, store_history);
end

function MutableKalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatVector, P0::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

    # Compute default value for missing parameters
    n, T = size(Y);
    m = size(B, 2);
    DQD = Symmetric(D*Q*D');

    # Return MutableKalmanSettings
    return MutableKalmanSettings(Y, B, R, C, D, Q, X0, P0, DQD, n, T, m, compute_loglik, store_history);
end

abstract type KalmanStatus end

"""
    OnlineKalmanStatus(...)

Define a mutable structure to handle minimal Kalman filter output. 

The Kalman filter history is not stored. This makes it ideal for online filtering problems. However, it is incompatible with the kalman smoother implementation.

# Arguments
- `t`: Current point in time
- `loglik`: Loglikelihood
- `X_prior`: Latest a-priori X
- `X_post`: Latest a-posteriori X
- `P_prior`: Latest a-priori P
- `P_post`: Latest a-posteriori P
- `e`: Forecast error
- `inv_F`: Inverse of the forecast error covariance
- `L`: Convenient shortcut for the filter and smoother
"""
mutable struct OnlineKalmanStatus <: KalmanStatus
    t::Int64
    loglik::Union{Float64, Nothing}
    X_prior::Union{FloatVector, Nothing}
    X_post::Union{FloatVector, Nothing}
    P_prior::Union{SymMatrix, Nothing}
    P_post::Union{SymMatrix, Nothing}
    e::Union{FloatVector, Nothing}
    inv_F::Union{SymMatrix, Nothing}
    L::Union{FloatMatrix, Nothing}
end

# OnlineKalmanStatus constructor
OnlineKalmanStatus() = OnlineKalmanStatus(0, [nothing for i=1:8]...);

"""
    DynamicKalmanStatus(...)

Define a mutable structure to handle any type of Kalman filter output.

The Kalman filter history is stored when `store_history` is set to true in the filter settings.

# Arguments
- `t`: Current point in time
- `loglik`: Loglikelihood
- `X_prior`: Latest a-priori X
- `X_post`: Latest a-posteriori X
- `P_prior`: Latest a-priori P
- `P_post`: Latest a-posteriori P
- `e`: Forecast error
- `inv_F`: Inverse of the forecast error covariance
- `L`: Convenient shortcut for the filter and smoother
- `history_X_prior`: History of a-priori X
- `history_X_post`: History of a-posteriori X
- `history_P_prior`: History of a-priori P
- `history_P_post`: History of a-posteriori P
- `history_e`: History of the forecast error
- `history_inv_F`: History of the inverse of the forecast error covariance
- `history_L`: History of the shortcut L
"""
mutable struct DynamicKalmanStatus <: KalmanStatus
    t::Int64
    loglik::Union{Float64, Nothing}
    X_prior::Union{FloatVector, Nothing}
    X_post::Union{FloatVector, Nothing}
    P_prior::Union{SymMatrix, Nothing}
    P_post::Union{SymMatrix, Nothing}
    e::Union{FloatVector, Nothing}
    inv_F::Union{SymMatrix, Nothing}
    L::Union{FloatMatrix, Nothing}
    history_X_prior::Union{Array{FloatVector,1}, Nothing}
    history_X_post::Union{Array{FloatVector,1}, Nothing}
    history_P_prior::Union{Array{SymMatrix,1}, Nothing}
    history_P_post::Union{Array{SymMatrix,1}, Nothing}
    history_e::Union{Array{FloatVector,1}, Nothing}
    history_inv_F::Union{Array{SymMatrix,1}, Nothing}
    history_L::Union{Array{FloatMatrix,1}, Nothing}
end

# DynamicKalmanStatus constructor
DynamicKalmanStatus() = DynamicKalmanStatus(0, [nothing for i=1:15]...);

"""
    SizedKalmanStatus(...)

Define a mutable structure that always store the filter history up to time T.

# Arguments
- `t`: Current point in time
- `loglik`: Loglikelihood
- `X_prior`: Latest a-priori X
- `X_post`: Latest a-posteriori X
- `P_prior`: Latest a-priori P
- `P_post`: Latest a-posteriori P
- `e`: Forecast error
- `inv_F`: Inverse of the forecast error covariance
- `L`: Convenient shortcut for the filter and smoother
- `history_X_prior`: History of a-priori X
- `history_X_post`: History of a-posteriori X
- `history_P_prior`: History of a-priori P
- `history_P_post`: History of a-posteriori P
- `history_e`: History of the forecast error
- `history_inv_F`: History of the inverse of the forecast error covariance
- `history_L`: History of the shortcut L
"""
mutable struct SizedKalmanStatus <: KalmanStatus
    t::Int64
    history_length::Int64
    loglik::Union{Float64, Nothing}
    X_prior::Union{FloatVector, Nothing}
    X_post::Union{FloatVector, Nothing}
    P_prior::Union{SymMatrix, Nothing}
    P_post::Union{SymMatrix, Nothing}
    e::Union{FloatVector, Nothing}
    inv_F::Union{SymMatrix, Nothing}
    L::Union{FloatMatrix, Nothing}
    history_X_prior::Array{FloatVector,1}
    history_X_post::Array{FloatVector,1}
    history_P_prior::Array{SymMatrix,1}
    history_P_post::Array{SymMatrix,1}
    history_e::Array{FloatVector,1}
    history_inv_F::Array{SymMatrix,1}
    history_L::Array{FloatMatrix,1}
end

# SizedKalmanStatus constructor
function SizedKalmanStatus(T::Int64)
    history_X_prior = Array{FloatVector,1}(undef, T);
    history_X_post = Array{FloatVector,1}(undef, T);
    history_P_prior = Array{SymMatrix,1}(undef, T);
    history_P_post = Array{SymMatrix,1}(undef, T);
    history_e = Array{FloatVector,1}(undef, T);
    history_inv_F = Array{SymMatrix,1}(undef, T);
    history_L = Array{FloatMatrix,1}(undef, T);
    return SizedKalmanStatus(0, T, [nothing for i=1:8]..., history_X_prior, history_X_post, history_P_prior, history_P_post, history_e, history_inv_F, history_L);
end

#=
--------------------------------------------------------------------------------------------------------------------------------
UC models: structures
--------------------------------------------------------------------------------------------------------------------------------
=#

abstract type UCSettings end

"""
    VARIMASettings(...)

Define an immutable structure to manage VARIMA specifications.

# Arguments
- `Y_levels`: Observed measurements (`nxT`) - in levels
- `Y`: Observed measurements (`nxT`) - differenced and demeaned
- `μ`: Sample average (per series)
- `n`: Number of series
- `d`: Degree of differencing
- `p`: Order of the autoregressive model
- `q`: Order of the moving average model
- `nr`: n*max(p, q+1)
- `np`: n*p
- `nq`: n*q
- `nnp`: (n^2)*p
- `nnq`: (n^2)*q
"""
struct VARIMASettings <: UCSettings
    Y_levels::Union{FloatMatrix, JMatrix{Float64}}
    Y::Union{FloatMatrix, JMatrix{Float64}}
    μ::FloatVector
    n::Int64
    d::Int64
    p::Int64
    q::Int64
    nr::Int64
    np::Int64
    nq::Int64
    nnp::Int64
    nnq::Int64
end

# VARIMASettings constructor
function VARIMASettings(Y_levels::Union{FloatMatrix, JMatrix{Float64}}, d::Int64, p::Int64, q::Int64)

    # Initialise
    n = size(Y_levels, 1);
    r = max(p, q+1);

    # Differenciate data
    Y = copy(Y_levels);

    if d > 0
        for i=1:d
            Y = diff(Y, dims=2);
        end
    end

    # Mean
    μ = mean_skipmissing(Y)[:,1];

    # Demean data
    Y = demean(Y);

    # VARIMASettings
    return VARIMASettings(Y_levels, Y, μ, n, d, p, q, n*r, n*p, n*q, (n^2)*p, (n^2)*q);
end

"""
    ARIMASettings(...)

Define an alias of VARIMASettings for arima models.
"""
ARIMASettings(Y_levels::Union{FloatMatrix, JMatrix{Float64}}, d::Int64, p::Int64, q::Int64) = VARIMASettings(Y_levels, d, p, q);
