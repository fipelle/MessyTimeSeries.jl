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
apriori(P::SymMatrix, settings::KalmanSettings) = Symmetric(settings.C * P * settings.C' + settings.DQD)::SymMatrix;

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
        status.history_e = Array{FloatVector,1}();
        status.history_inv_F = Array{SymMatrix,1}();
        status.history_L = Array{FloatMatrix,1}();
    end
end

function apriori!(::Type{FloatVector}, settings::KalmanSettings, status::KalmanStatus)
    status.X_prior = apriori(status.X_post, settings);
    status.P_prior = apriori(status.P_post, settings);
end

"""
    find_observed_data(settings::KalmanSettings, status::KalmanStatus)

Return position of the observed measurements at time status.t.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct

    find_observed_data(settings::KalmanSettings, t::Int64)

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

function find_observed_data(settings::KalmanSettings, t::Int64)
    if t <= settings.T
        Y_t_all = @view settings.Y[:, t];
        ind_not_missings = findall(ismissing.(Y_t_all) .== false);
        if length(ind_not_missings) > 0
            return ind_not_missings;
        end
    end
end

"""
    update_loglik!(status::KalmanStatus)

Update status.loglik.

# Arguments
- `status`: KalmanStatus struct
"""
function update_loglik!(status::KalmanStatus)
    status.loglik -= 0.5*(-logdet(status.inv_F) + status.e'*status.inv_F*status.e);
end

"""
    aposteriori!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::IntVector)

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
function aposteriori!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::IntVector)

    Y_t = @view settings.Y[ind_not_missings, status.t];
    B_t = @view settings.B[ind_not_missings, :];
    R_t = @view settings.R[ind_not_missings, ind_not_missings];

    # Forecast error
    status.e = Y_t - B_t*status.X_prior;
    status.inv_F = inv(Symmetric(B_t*status.P_prior*B_t' + R_t))::SymMatrix;

    # Kalman gain
    K_t = status.P_prior*B_t'*status.inv_F;

    # Convenient shortcut
    status.L = I - K_t*B_t;

    # A posteriori estimates
    status.X_post = status.X_prior + K_t*status.e;
    status.P_post = Symmetric(status.L*status.P_prior*status.L' + K_t*R_t*K_t'); # Joseph form

    # Update log likelihood
    if settings.compute_loglik == true
        update_loglik!(status);
    end
end

function aposteriori!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing)
    status.X_post = copy(status.X_prior);
    status.P_post = copy(status.P_prior);
    status.e = zeros(1);
    status.inv_F = Symmetric(zeros(1,1));
    status.L = Matrix(I, settings.m, settings.m) |> FloatMatrix;
end

"""
    kfilter!(settings::KalmanSettings, status::KalmanStatus)

Kalman filter: a-priori prediction and a-posteriori update.

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
        push!(status.history_e, status.e);
        push!(status.history_inv_F, status.inv_F);
        push!(status.history_L, status.L);
    end
end

"""
    kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, h::Int64)

Forecast X up to h-step ahead.

    kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, P::Union{SymMatrix, Nothing}, h::Int64)

Forecast X and P up to h-step ahead.
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
    compute_smoothing_factors(settings::KalmanSettings, ind_not_missings::IntVector, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix)

Compute the factors to perform the backwards pass.

    compute_smoothing_factors(settings::KalmanSettings, ind_not_missings::Nothing, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix)

Compute the factors to perform the backwards pass when all series are missing.
"""
function compute_smoothing_factors(settings::KalmanSettings, ind_not_missings::IntVector, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix)

    # Retrieve coefficients
    B_t = @view settings.B[ind_not_missings, :];
    B_inv_F = B_t'*inv_F;
    L_C = L'*settings.C';

    # Compute J1 and J2
    J1 = B_inv_F*e + L_C*J1;
    J2 = Symmetric(B_inv_F*B_t + L_C*J2*L_C');

    # Return output
    return J1, J2;
end

function compute_smoothing_factors(settings::KalmanSettings, ind_not_missings::Nothing, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix)

    # Compute J1 and J2
    J1 = settings.C'*J1;
    J2 = Symmetric(settings.C'*J2*settings.C);

    # Return output
    return J1, J2;
end

"""
    backwards_pass(Xf::FloatVector, Pf::SymMatrix, J1::FloatVector)

Backward pass for the state vector.

    backwards_pass(Pf::SymMatrix, J2::FloatVector)

Backward pass for the covariance of the states.
"""
backwards_pass(Xf::FloatVector, Pf::SymMatrix, J1::FloatVector) = Xf + Pf*J1;
backwards_pass(Pf::SymMatrix, J2::FloatVector) = Symmetric(Pf - Pf*J2*Pf);

"""
    ksmoother(settings::KalmanSettings, status::KalmanStatus)

Kalman smoother: RTS smoother from the last evaluated time period in status to t==0.

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

    J1 = zeros(settings.m);
    J2 = zeros(settings.m, settings.m);

    # Loop over t (from status.t-1 to 2)
    for t=status.t-1:-1:2

        # Pointers
        Xf = status.history_X_post[t];
        Pf = status.history_P_post[t];
        e = status.history_e[t];
        inv_F = status.history_inv_F[t];
        L = status.history_L[t];

        # Handle missing observations
        ind_not_missings = find_observed_data(settings, t);

        # Smoothed estimates for t
        J1, J2 = compute_smoothing_factors(settings, ind_not_missings, J1, J2, e, inv_F, L);
        pushfirst!(history_X, backwards_pass(Xf, J1, Pf));
        pushfirst!(history_P, backwards_pass(Pf, J2));
    end

    # Pointers
    Xf = status.history_X_post[1];
    Pf = status.history_P_post[1];
    e = status.history_e[1];
    inv_F = status.history_inv_F[1];
    L = status.history_L[1];

    # Handle missing observations
    ind_not_missings = find_observed_data(settings, 1);

    # Compute smoothed estimates for t==0
    J1, J2 = compute_smoothing_factors(settings, ind_not_missings, J1, J2, e, inv_F, L);
    X0 = backwards_pass(Xf, J1, Pf);
    P0 = backwards_pass(Pf, J2);

    # Return output
    return history_X, history_P, X0, P0;
end
