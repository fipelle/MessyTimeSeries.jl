"""
    kfilter!(settings::KalmanSettings, status::KalmanStatus)

Kalman filter: a-priori prediction and a-posteriori update.

# Model
The state space model used below is,

``Y_{t} = B*X_{t} + e_{t}``

``X_{t} = C*X_{t-1} + v_{t}``

Where ``e_{t} ~ N(0, R)`` and ``v_{t} ~ N(0, V)``.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
"""
function kfilter!(settings::KalmanSettings, status::KalmanStatus)

    # Update status.t
    status.t += 1;

    # A-priori prediction
    apriori!(typeof(status.X_prior), settings, status);

    # Handle missing observations
    ind_not_missings = find_observed_data(settings, status);

    # Ex-post update
    aposteriori!(settings, status, ind_not_missings);

    # Update history of *_prior and *_post
    if settings.store_history == true
        push!(status.history_X_prior, status.X_prior);
        push!(status.history_X_post, status.X_post);
        push!(status.history_P_prior, status.P_prior);
        push!(status.history_P_post, status.P_post);
    end
end

"""
    apriori!(::Type{Nothing}, settings::KalmanSettings, status::KalmanStatus)

Kalman filter a-priori prediction for t==1.

# Arguments
- `::Type{Nothing}`: first prediction
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct

    apriori!(::Type{FloatVector}, settings::KalmanSettings, status::KalmanStatus)

Kalman filter a-priori prediction.

# Arguments
- `::Type{FloatVector}`: standard prediction
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
"""
function apriori!(::Type{Nothing}, settings::KalmanSettings, status::KalmanStatus)

    status.X_prior = apriori(settings.X0, settings);
    status.P_prior = apriori(settings.P0, settings);

    if settings.compute_loglik == true
        status.loglik = 0.0;
    end

    if settings.store_history == true
        status.history_X_prior = Array{FloatVector,1}();
        status.history_X_post = Array{FloatVector,1}();
        status.history_P_prior = Array{SymMatrix,1}();
        status.history_P_post = Array{SymMatrix,1}();
    end
end

function apriori!(::Type{FloatVector}, settings::KalmanSettings, status::KalmanStatus)
    status.X_prior = apriori(status.X_post, settings);
    status.P_prior = apriori(status.P_post, settings);
end

"""
    apriori(X::FloatVector, settings::KalmanSettings)

Kalman filter a-priori prediction for X.

# Arguments
- `X`: Last expected value of the states
- `settings`: KalmanSettings struct

    apriori(P::SymMatrix, settings::KalmanSettings)

Kalman filter a-priori prediction for P.

# Arguments
- `P`: Last conditional covariance the states
- `settings`: KalmanSettings struct
"""
apriori(X::FloatVector, settings::KalmanSettings) = settings.C * X;
apriori(P::SymMatrix, settings::KalmanSettings) = Symmetric(settings.C * P * settings.C' + settings.V)::SymMatrix;

"""
    find_observed_data(settings::KalmanSettings, status::KalmanStatus)

Return position of the observed measurements at time t.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
"""
function find_observed_data(settings::KalmanSettings, status::KalmanStatus)
    if status.t <= settings.T
        Y_t_all = @view settings.Y[:, status.t];
        ind_not_missings = findall(ismissing.(Y_t_all) .== false);
        if length(ind_not_missings) > 0
            return ind_not_missings;
        end
    end
end

"""
    aposteriori!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Array{Int64,1})

Kalman filter a-posteriori update. Measurements are observed (or partially observed) at time t.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
- `ind_not_missings`: Position of the observed measurements

    aposteriori!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing)

Kalman filter a-posteriori update. All measurements are not observed at time t.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
- `ind_not_missings`: Empty array
"""
function aposteriori!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Array{Int64,1})

    Y_t = @view settings.Y[ind_not_missings, status.t];
    B_t = @view settings.B[ind_not_missings, :];
    R_t = @view settings.R[ind_not_missings, ind_not_missings];

    # Forecast error
    ε_t = Y_t - B_t*status.X_prior;
    Σ_t = Symmetric(B_t*status.P_prior*B_t' + R_t)::SymMatrix;

    # Kalman gain
    K_t = status.P_prior*B_t'*inv(Σ_t);

    # A posteriori estimates
    status.X_post = status.X_prior + K_t*ε_t;
    status.P_post = Symmetric(status.P_prior - K_t*B_t*status.P_prior)::SymMatrix;

    # Update log likelihood
    if settings.compute_loglik == true
        update_loglik!(status, ε_t, Σ_t);
    end
end

function aposteriori!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing)
    status.X_post = copy(status.X_prior);
    status.P_post = copy(status.P_prior);
end

"""
    update_loglik!(status::KalmanStatus, ε_t::FloatVector, Σ_t::SymMatrix)

Update status.loglik.

# Arguments
- `status`: KalmanStatus struct
- `ε_t`: Forecast error
- `Σ_t`: Forecast error covariance
"""
function update_loglik!(status::KalmanStatus, ε_t::FloatVector, Σ_t::SymMatrix)
    status.loglik -= 0.5*(logdet(Σ_t) + ε_t'*inv(Σ_t)*ε_t);
end

"""
    kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, h::Int64)

Forecast X up to h-steps ahead.

    kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, P::Union{SymMatrix, Nothing}, h::Int64)

Forecast X and P up to h-steps ahead.

# Model
The state space model used below is,

``Y_{t} = B*X_{t} + e_{t}``

``X_{t} = C*X_{t-1} + v_{t}``

Where ``e_{t} ~ N(0, R)`` and ``v_{t} ~ N(0, V)``.
"""
function kforecast(settings::KalmanSettings, Xt::Union{FloatVector, Nothing}, h::Int64)

    # Initialise forecast history
    history_X = Array{FloatVector,1}();

    X = copy(Xt);

    # Loop over forecast horizons
    for horizon=1:h
        X = apriori(X, settings);
        push!(history_X, X);
    end

    # Return output
    return history_X;
end

function kforecast(settings::KalmanSettings, Xt::Union{FloatVector, Nothing}, Pt::Union{SymMatrix, Nothing}, h::Int64)

    # Initialise forecast history
    history_X = Array{FloatVector,1}();
    history_P = Array{SymMatrix,1}();

    X = copy(Xt);
    P = copy(Pt);

    # Loop over forecast horizons
    for horizon=1:h
        X = apriori(X, settings);
        P = apriori(P, settings);
        push!(history_X, X);
        push!(history_P, P);
    end

    # Return output
    return history_X, history_P;
end

"""
    ksmoother(settings::KalmanSettings, status::KalmanStatus)

Kalman smoother: RTS smoother from the last evaluated time period in status to t==0.

# Model
The state space model used below is,

``Y_{t} = B*X_{t} + e_{t}``

``X_{t} = C*X_{t-1} + v_{t}``

Where ``e_{t} ~ N(0, R)`` and ``v_{t} ~ N(0, V)``.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
"""
function ksmoother(settings::KalmanSettings, status::KalmanStatus)

    # Initialise smoother history
    history_X = Array{FloatVector,1}();
    history_P = Array{SymMatrix,1}();
    push!(history_X, copy(status.X_post));
    push!(history_P, copy(status.P_post));

    # Loop over t
    for t=status.t:-1:2

        # Pointers
        Xp = status.history_X_prior[t];
        Pp = status.history_P_prior[t];
        Xf_lagged = status.history_X_post[t-1];
        Pf_lagged = status.history_P_post[t-1];
        Xs = history_X[1];
        Ps = history_P[1];

        # Smoothed estimates for t-1
        J1 = compute_J1(Pf_lagged, Pp, settings);
        pushfirst!(history_X, backwards_pass(Xf_lagged, J1, Xs, Xp));
        pushfirst!(history_P, backwards_pass(Pf_lagged, J1, Ps, Pp));
    end

    # Pointers
    Xp = status.history_X_prior[1];
    Pp = status.history_P_prior[1];
    Xs = history_X[1];
    Ps = history_P[1];

    # Compute smoothed estimates for t==0
    J1 = compute_J1(settings.P0, Pp, settings);
    X0 = backwards_pass(settings.X0, J1, Xs, Xp);
    P0 = backwards_pass(settings.P0, J1, Ps, Pp);

    # Return output
    return history_X, history_P, X0, P0;
end

"""
    compute_J1(Pf_lagged::SymMatrix, Pp::SymMatrix, settings::KalmanSettings)

Compute J_{t-1} as in Shumway and Stoffer (2011, chapter 6) to operate the RTS smoother.
"""
compute_J1(Pf_lagged::SymMatrix, Pp::SymMatrix, settings::KalmanSettings) = Pf_lagged*settings.C'*inv(Pp);

"""
    backwards_pass(Xf_lagged::FloatVector, J1::FloatArray, Xs::FloatVector, Xp::FloatVector)

Backwards pass for X to get the smoothed states at time t-1.

# Arguments
- `Xf_lagged`: Filtered states for time t-1
- `J1`: J_{t-1} as in Shumway and Stoffer (2011, chapter 6)
- `Xs`: Smoothed states for time t
- `Xp`: A-priori states for time t

    backwards_pass(Pf_lagged::SymMatrix, J1::FloatArray, Ps::SymMatrix, Pp::SymMatrix)

Backwards pass for P to get the smoothed conditional covariance at time t-1.

# Arguments
- `Pf_lagged`: Filtered conditional covariance for time t-1
- `J1`: J_{t-1} as in Shumway and Stoffer (2011, chapter 6)
- `Ps`: Smoothed conditional covariance for time t
- `Pp`: A-priori conditional covariance for time t
"""
backwards_pass(Xf_lagged::FloatVector, J1::FloatArray, Xs::FloatVector, Xp::FloatVector) = Xf_lagged + J1*(Xs-Xp);
backwards_pass(Pf_lagged::SymMatrix, J1::FloatArray, Ps::SymMatrix, Pp::SymMatrix) = Symmetric(Pf_lagged + J1*(Ps-Pp)*J1')::SymMatrix;
