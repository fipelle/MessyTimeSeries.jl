"""
    increase_time!(status::KalmanStatus)
    increase_time!(status::SizedKalmanStatus)

Increase time by one unit.
"""
increase_time!(status::KalmanStatus) = status.t += 1;
increase_time!(status::SizedKalmanStatus) = increase_time!(status.online_status);

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
    initialise_apriori!(settings::KalmanSettings, status::KalmanStatus)

Initialise X_prior, P_prior and loglik.
"""
function initialise_apriori!(settings::KalmanSettings, status::KalmanStatus)

    # First a-priori prediction for X and P
    status.X_prior = apriori(settings.X0, settings);
    status.P_prior = apriori(settings.P0, settings);

    # Initialise buffers
    status.buffer_J1 = similar(status.X_prior);
    status.buffer_J2 = similar(status.P_prior);
    status.buffer_m_m = similar(status.P_prior);

    # Initialise loglikelihood
    if settings.compute_loglik == true
        status.loglik = 0.0;
    end
end

"""
    initialise_status_history!(settings::KalmanSettings, status::OnlineKalmanStatus)
    initialise_status_history!(settings::KalmanSettings, status::DynamicKalmanStatus)
    initialise_status_history!(settings::KalmanSettings, status::SizedKalmanStatus)

Initialise the `status.history_*` arrays when needed.
"""
function initialise_status_history!(settings::KalmanSettings, status::OnlineKalmanStatus)
    if settings.store_history == true
        error("History must not be stored with an `OnlineKalmanStatus` struct. Try with an `DynamicKalmanStatus` or `SizedKalmanStatus` struct.")
    end
end

function initialise_status_history!(settings::KalmanSettings, status::DynamicKalmanStatus)
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

function initialise_status_history!(settings::KalmanSettings, status::SizedKalmanStatus)
    if settings.store_history == false
        error("History must be stored with a `SizedKalmanStatus` struct. Try with an `OnlineKalmanStatus` or `DynamicKalmanStatus` struct.")
    end
end

"""
    apriori!(old_X_prior::Nothing, settings::KalmanSettings, status::KalmanStatus)
    apriori!(old_X_prior::Nothing, settings::KalmanSettings, status::SizedKalmanStatus)

Kalman filter a-priori prediction for t==1.

# Arguments
- `old_X_prior`: latest a-priori prediction (i.e., `nothing` for the first point in time)
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct

    apriori!(old_X_prior::FloatVector, settings::KalmanSettings, status::KalmanStatus)
    apriori!(old_X_prior::FloatVector, settings::KalmanSettings, status::SizedKalmanStatus)

Kalman filter a-priori prediction.

# Arguments
- `old_X_prior`: latest a-priori prediction
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
"""
function apriori!(old_X_prior::Nothing, settings::KalmanSettings, status::KalmanStatus)
    initialise_apriori!(settings, status);
    initialise_status_history!(settings, status);
end

function apriori!(old_X_prior::Nothing, settings::KalmanSettings, status::SizedKalmanStatus)
    initialise_apriori!(settings, status.online_status);
    initialise_status_history!(settings, status);
end

function apriori!(old_X_prior::FloatVector, settings::KalmanSettings, status::KalmanStatus)
    status.X_prior = apriori(status.X_post, settings);
    status.P_prior = apriori(status.P_post, settings);
end

apriori!(old_X_prior::FloatVector, settings::KalmanSettings, status::SizedKalmanStatus) = apriori!(old_X_prior, settings, status.online_status);

"""
    call_apriori!(settings::KalmanSettings, status::KalmanStatus)
    call_apriori!(settings::KalmanSettings, status::SizedKalmanStatus)

API to call `apriori!`.
"""
call_apriori!(settings::KalmanSettings, status::KalmanStatus) = apriori!(status.X_prior, settings, status);
call_apriori!(settings::KalmanSettings, status::SizedKalmanStatus) = apriori!(status.online_status.X_prior, settings, status);

"""
    find_observed_data(settings::KalmanSettings, status::KalmanStatus)
    find_observed_data(settings::KalmanSettings, status::SizedKalmanStatus)

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
    if status.t <= settings.Y.T
        Y_t_all = @view settings.Y.data[:, status.t];
        ind_not_missings = findall(ismissing.(Y_t_all) .== false);
        if length(ind_not_missings) > 0
            return ind_not_missings;
        end
    end
end

find_observed_data(settings::KalmanSettings, status::SizedKalmanStatus) = find_observed_data(settings, status.online_status);

function find_observed_data(settings::KalmanSettings, t::Int64)
    if t <= settings.Y.T
        Y_t_all = @view settings.Y.data[:, t];
        ind_not_missings = findall(ismissing.(Y_t_all) .== false);
        if length(ind_not_missings) > 0
            return ind_not_missings;
        end
    end
end

"""
    update_inv_F!(R::UniformScaling{Float64}, status::KalmanStatus, B_it::SubArray{Float64})

Update status.inv_F when using the sequential processing approach.

# Notes
- `B_it` is a (mx1) column vector.

    update_inv_F!(R::SymMatrix, status::KalmanStatus, B_t::SubArray{Float64}, ind_not_missings::IntVector)

Update status.inv_F.
"""
function update_inv_F!(R::UniformScaling{Float64}, status::KalmanStatus, B_it::SubArray{Float64})
    F_it = status.buffer_m_n_obs'*B_it; # I cannot use mul!(...) here since R is UniformScaling{Float64}
    F_it += R;
    @infiltrate

    push!(status.inv_F, 1/F_it); # 1/F_it is a scalar
end

function update_inv_F!(R::SymMatrix, status::KalmanStatus, B_t::SubArray{Float64}, ind_not_missings::IntVector)
    F_t = R[ind_not_missings, ind_not_missings];
    mul!(F_t, status.buffer_m_n_obs', B_t', 1.0, 1.0);
    status.inv_F = inv(Symmetric(F_t));
end

"""
    update_P_post!(status::KalmanStatus, K_it::FloatMatrix, R_it::UniformScaling{Float64}, ind_not_missings::IntVector)
    update_P_post!(P_post_old::SymMatrix, R::SymMatrix, status::KalmanStatus, K_t::FloatMatrix, ind_not_missings::IntVector)
    update_P_post!(P_post_old::Nothing, R::SymMatrix, status::KalmanStatus, K_t::FloatMatrix, ind_not_missings::IntVector)

Update status.P_post.
"""
function update_P_post!(status::KalmanStatus, K_it::FloatVector, R_it::UniformScaling{Float64})
    mul!(status.buffer_m_m, status.L, status.P_post);
    @infiltrate

    mul!(status.P_post.data, status.buffer_m_m, status.L');
    @infiltrate

    status.buffer_m_n_obs = K_it*R_it; # I cannot use mul!(...) here due to the type of K_it and R_it
    @infiltrate

    mul!(status.P_post.data, status.buffer_m_n_obs, K_it', 1.0, 1.0);
    @infiltrate
end

function update_P_post!(P_post_old::SymMatrix, R::SymMatrix, status::KalmanStatus, K_t::FloatMatrix, ind_not_missings::IntVector)
    R_t = @view R[ind_not_missings, ind_not_missings];
    mul!(status.buffer_m_m, status.L, status.P_prior);
    mul!(status.P_post.data, status.buffer_m_m, status.L');
    mul!(status.buffer_m_n_obs, K_t, R_t);
    mul!(status.P_post.data, status.buffer_m_n_obs, K_t', 1.0, 1.0);
end

function update_P_post!(P_post_old::Nothing, R::SymMatrix, status::KalmanStatus, K_t::FloatMatrix, ind_not_missings::IntVector)
    R_t = @view R[ind_not_missings, ind_not_missings];
    mul!(status.buffer_m_m, status.L, status.P_prior);
    status.P_post = Symmetric(status.buffer_m_m*status.L');
    mul!(status.buffer_m_n_obs, K_t, R_t);
    mul!(status.P_post.data, status.buffer_m_n_obs, K_t', 1.0, 1.0);
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
    aposteriori!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::IntVector)

Kalman filter a-posteriori update. Measurements are observed (or partially observed) at time t.

The update for the covariance matrix is implemented by using the Joseph's stabilised form (Bucy and Joseph, 1968).

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
- `ind_not_missings`: Position of the observed measurements

    aposteriori!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing)
    aposteriori!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::Nothing)

Kalman filter a-priori prediction. All measurements are not observed at time t.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
- `ind_not_missings`: Empty array
"""
function aposteriori!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::IntVector)

    # Convenient view
    B_t = @view settings.B[ind_not_missings, :];

    # Forecast error
    status.e = settings.Y.data[ind_not_missings, status.t];
    mul!(status.e, B_t, status.X_prior, -1.0, 1.0);

    # Convenient shortcut for the forecast error covariance matrix and Kalman gain
    # The line below initialises `status.buffer_m_n_obs` for the current point in time
    status.buffer_m_n_obs = status.P_prior*B_t';

    # Inverse of the forecast error covariance matrix
    update_inv_F!(settings.R, status, B_t, ind_not_missings);

    # Kalman gain
    K_t = status.buffer_m_n_obs*status.inv_F;

    # Convenient shortcut for the Joseph form and needed statistics for the Kalman smoother
    status.L = Matrix(1.0I, settings.m, settings.m);
    mul!(status.L, K_t, B_t, -1.0, 1.0);

    # A posteriori estimates: X_post
    status.X_post = copy(status.X_prior);
    mul!(status.X_post, K_t, status.e, 1.0, 1.0);

    # A posteriori estimates: P_post (P_post is updated using the Joseph form)
    update_P_post!(status.P_post, settings.R, status, K_t, ind_not_missings);
    
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
    status.L = Matrix(1.0I, settings.m, settings.m);
end

aposteriori!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::Union{IntVector, Nothing}) = aposteriori!(settings, status.online_status, ind_not_missings);

"""
    aposteriori_sequential!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::IntVector)
    aposteriori_sequential!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::IntVector)

Sequential Kalman filter a-posteriori update. Measurements are observed (or partially observed) at time t.

The update for the covariance matrix is implemented by using the Joseph's stabilised form (Bucy and Joseph, 1968).

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
- `ind_not_missings`: Position of the observed measurements

# Notes
- The sequential processing (Anderson and Moore, 1979, section 6.4) employed in this function is also described as the univariate form of the Kalman filter (e.g., Durbin and Koopman, 2000).

    aposteriori_sequential!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing)
    aposteriori_sequential!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::Nothing)

Kalman filter a-priori prediction for sequential processing. All measurements are not observed at time t.

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
- `ind_not_missings`: Empty array

# Notes
- The sequential processing (Anderson and Moore, 1979, section 6.4) employed in this function is also described as the univariate form of the Kalman filter (e.g., Durbin and Koopman, 2000).
"""
function aposteriori_sequential!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::IntVector)

    # Initialise sequential processing to the latest a-priori prediction
    # TBD: avoid this copy by using an appropriate if-else statement in the loop
    status.X_post = copy(status.X_prior);
    status.P_post = copy(status.P_prior);

    # Initialise key terms
    status.e = Float64[];
    status.inv_F = Float64[];
    @infiltrate

    # Sequentially process the observables
    for i in ind_not_missings

        # Convenient view
        B_it = @view settings.B[i, :]; # this is (mx1) vector <- the use of an adjoint of a view would increase the number of operations required to finalise the a-posteriori update (too many adjoint calls)
        
        # Forecast error
        push!(status.e, settings.Y.data[i, status.t] - B_it'*status.X_post); # I cannot use mul!(...) here since I am updating status.e element-wise
        @infiltrate

        # Convenient shortcut for the forecast error covariance matrix and Kalman gain
        # The line below initialises `status.buffer_m_n_obs` for the current series and point in time
        status.buffer_m_n_obs = status.P_post*B_it; # this is a (mx1) vector
        @infiltrate

        # Inverse of the forecast error covariance matrix
        update_inv_F!(settings.R, status, B_it);
        @infiltrate

        # Kalman gain
        K_it = status.buffer_m_n_obs*status.inv_F[end];
        @infiltrate

        # Convenient shortcut for the Joseph form and needed statistics for the Kalman smoother
        status.L = Matrix(1.0I, settings.m, settings.m);
        mul!(status.L, K_it, B_it', -1.0, 1.0);
        @infiltrate

        # A posteriori estimates: X_post
        status.X_post += K_it*status.e[end]; # I cannot use mul!(...) here since status.e[end] is a scalar
        @infiltrate

        # A posteriori estimates: P_post (P_post is updated using the Joseph form)
        update_P_post!(status, K_it, settings.R);
        @infiltrate

        # ========================
        # Update log likelihood
        # ========================
        # => TBA
        # ========================
    end
end

aposteriori_sequential!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing) = nothing; # TBC: this may be just a call to the equivalent version of aposteriori!(...)
aposteriori_sequential!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::Union{IntVector, Nothing}) = aposteriori_sequential!(settings, status.online_status, ind_not_missings);

"""
    call_aposteriori!(settings::KalmanSettings, status::KalmanStatus, R::SymMatrix, ind_not_missings::Union{IntVector, Nothing})

Call standard aposteriori!(...) routine.

    call_aposteriori!(settings::KalmanSettings, status::KalmanStatus, R::UniformScaling{Float64}, ind_not_missings::Union{IntVector, Nothing})

Call aposteriori_sequential!(...) for sequential a-posteriori update.
"""
call_aposteriori!(settings::KalmanSettings, status::KalmanStatus, R::SymMatrix, ind_not_missings::Union{IntVector, Nothing}) = aposteriori!(settings, status, ind_not_missings);
call_aposteriori!(settings::KalmanSettings, status::KalmanStatus, R::UniformScaling{Float64}, ind_not_missings::Union{IntVector, Nothing}) = aposteriori_sequential!(settings, status, ind_not_missings);

"""
    update_status_history!(settings::KalmanSettings, status::OnlineKalmanStatus)
    update_status_history!(settings::KalmanSettings, status::DynamicKalmanStatus)
    update_status_history!(settings::KalmanSettings, status::SizedKalmanStatus)

Update the `status.history_*` arrays with a new entry, when needed.
"""
update_status_history!(settings::KalmanSettings, status::OnlineKalmanStatus) = nothing;

function update_status_history!(settings::KalmanSettings, status::DynamicKalmanStatus)
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

function update_status_history!(settings::KalmanSettings, status::SizedKalmanStatus)
    status.history_X_prior[status.online_status.t] = status.online_status.X_prior;
    status.history_X_post[status.online_status.t] = status.online_status.X_post;
    status.history_P_prior[status.online_status.t] = status.online_status.P_prior;
    status.history_P_post[status.online_status.t] = status.online_status.P_post;
    status.history_e[status.online_status.t] = status.online_status.e;
    status.history_inv_F[status.online_status.t] = status.online_status.inv_F;
    status.history_L[status.online_status.t] = status.online_status.L;
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
    increase_time!(status);

    # A-priori prediction
    call_apriori!(settings, status);

    # Handle missing observations
    ind_not_missings = find_observed_data(settings, status);

    # Ex-post update
    call_aposteriori!(settings, status, settings.R, ind_not_missings);

    # Update `status.history_*`
    update_status_history!(settings, status);
end

"""
    reset_kalman_status!(status::KalmanStatus)

Reset essential entries in `status` to their original state (i.e., when t==0)
"""
function reset_kalman_status!(status::KalmanStatus)
    status.t = 0;
    status.X_prior = nothing;
end

"""
    kfilter_full_sample(settings::KalmanSettings, status::KalmanStatus=DynamicKalmanStatus())

Run Kalman filter for ``t=1, \\ldots, T`` and return `status`.
"""
function kfilter_full_sample(settings::KalmanSettings, status::KalmanStatus=DynamicKalmanStatus())
    for t=1:settings.Y.T
        kfilter!(settings, status);
    end

    return status;
end

"""
    kfilter_full_sample!(settings::KalmanSettings, status::SizedKalmanStatus)

Run Kalman filter from `t=1` to `history_length` and update `status` in-place.
"""
function kfilter_full_sample!(settings::KalmanSettings, status::SizedKalmanStatus)
    reset_kalman_status!(status.online_status);
    for t=1:status.history_length
        kfilter!(settings, status);
    end
end

"""
    kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, h::Int64)

Forecast `X` up to `h` steps ahead.

# Arguments
- `settings`: KalmanSettings struct
- `X`: State vector
- `h`: Forecast horizon

    kforecast(settings::KalmanSettings, X::Union{FloatVector, Nothing}, P::Union{SymMatrix, Nothing}, h::Int64)

Forecast `X` and `P` up to `h` steps ahead.

# Arguments
- `settings`: KalmanSettings struct
- `X`: State vector
- `P`: Covariance matrix of the states
- `h`: Forecast horizon
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
    retrieve_status_t(status::KalmanStatus)
    retrieve_status_t(status::SizedKalmanStatus)

Retrieve last filtering point in time.
"""
retrieve_status_t(status::KalmanStatus) = status.t;
retrieve_status_t(status::SizedKalmanStatus) = status.online_status.t;

"""
    update_smoothing_factors!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::IntVector, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix)
    update_smoothing_factors!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::IntVector, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix)

Update J1 and J2 with a-posteriori recursion.

    update_smoothing_factors!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing, J1::FloatVector, J2::SymMatrix)
    update_smoothing_factors!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::Nothing, J1::FloatVector, J2::SymMatrix)
    update_smoothing_factors!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix)

Update J1 and J2 with a-priori recursion when all series are missing.
"""
function update_smoothing_factors!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::IntVector, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix)

    # Retrieve coefficients
    B_t = @view settings.B[ind_not_missings, :];
    B_inv_F = B_t'*inv_F;
    mul!(status.buffer_m_m, L', settings.C');

    # Compute J1
    mul!(status.buffer_J1, status.buffer_m_m, J1);
    mul!(J1, B_inv_F, e);
    J1 .+= status.buffer_J1;

    # Compute J2
    mul!(status.buffer_J2, status.buffer_m_m, J2);
    mul!(J2.data, status.buffer_J2, status.buffer_m_m');
    mul!(J2.data, B_inv_F, B_t, 1.0, 1.0);
end

update_smoothing_factors!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::IntVector, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix) = update_smoothing_factors!(settings, status.online_status, ind_not_missings, J1, J2, e, inv_F, L);

function update_smoothing_factors!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing, J1::FloatVector, J2::SymMatrix)
    copyto!(J1, settings.C'*J1);
    mul!(status.buffer_J2, settings.C', J2);
    mul!(J2.data, status.buffer_J2, settings.C);
end

update_smoothing_factors!(settings::KalmanSettings, status::SizedKalmanStatus, ind_not_missings::Nothing, J1::FloatVector, J2::SymMatrix) = update_smoothing_factors!(settings, status.online_status, ind_not_missings, J1, J2);
update_smoothing_factors!(settings::KalmanSettings, status::KalmanStatus, ind_not_missings::Nothing, J1::FloatVector, J2::SymMatrix, e::FloatVector, inv_F::SymMatrix, L::FloatMatrix) = update_smoothing_factors!(settings, status, ind_not_missings, J1, J2);

"""
    backwards_pass(Xp::FloatVector, Pp::SymMatrix, J1::FloatVector)

Backward pass for the state vector.

    backwards_pass(Pp::SymMatrix, J2::SymMatrix)

Backward pass for the covariance of the states.
"""
backwards_pass(Xp::FloatVector, Pp::SymMatrix, J1::FloatVector) = Xp + Pp*J1;
backwards_pass(Pp::SymMatrix, J2::SymMatrix) = Symmetric(Pp - Pp*J2*Pp);

"""
    ksmoother(settings::KalmanSettings, status::KalmanStatus, t_stop::Int64=1)

Kalman smoother: RTS smoother from the last evaluated time period in `status` up to ``t==0``.

The smoother is implemented following the approach proposed in Durbin and Koopman (2012).

# Arguments
- `settings`: KalmanSettings struct
- `status`: KalmanStatus struct
- `t_stop`: Optional argument that can be used to define the last smoothing period (default: 1)
"""
function ksmoother(settings::KalmanSettings, status::KalmanStatus, t_stop::Int64=1)

    # Initialise smoother history
    history_X = Array{FloatVector,1}();
    history_P = Array{SymMatrix,1}();

    J1 = zeros(settings.m);
    J2 = Symmetric(zeros(settings.m, settings.m));

    # Loop over t
    for t=retrieve_status_t(status):-1:t_stop

        # Pointers
        Xp = status.history_X_prior[t];
        Pp = status.history_P_prior[t];
        e = status.history_e[t];
        inv_F = status.history_inv_F[t];
        L = status.history_L[t];

        # Handle missing observations
        ind_not_missings = find_observed_data(settings, t);

        # Smoothed estimates for t
        update_smoothing_factors!(settings, status, ind_not_missings, J1, J2, e, inv_F, L);
        pushfirst!(history_X, backwards_pass(Xp, Pp, J1));
        pushfirst!(history_P, backwards_pass(Pp, J2));
    end

    # Compute smoothed estimates for t==0
    if t_stop == 1
        update_smoothing_factors!(settings, status, nothing, J1, J2);
        X0 = backwards_pass(settings.X0, settings.P0, J1);
        P0 = backwards_pass(settings.P0, J2);
    
    # Type-stable output for when t_stop != 1 (i.e., if the routine stops before t==1) and X0, P0 cannot be computed
    else
        X0 = zeros(1);
        P0 = Symmetric(ones(1,1));
    end

    # Return output
    return history_X, history_P, X0, P0;
end
