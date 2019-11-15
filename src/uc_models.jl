#=
--------------------------------------------------------------------------------------------------------------------------------
UC models: general interface
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    fmin_uc_models(θ_unbound::FloatVector, lb::FloatVector, ub::FloatVector, transform_id::Array{Int64,1}, model_structure::Function, settings::UCSettings)

Return -1*loglikelihood for the UC model specified by model_structure(settings)

# Arguments
- `θ_unbound`: Model parameters (with unbounded support)
- `lb`: Lower bound for the parameters
- `ub`: Upper bound for the parameters
- `transform_id`: Type of transformation required for the parameters (0 = none, 1 = generalised log, 2 = generalised logit)
- `model_structure`: Function to setup the state-space structure
- `settings`: Settings for model_structure
"""
function fmin_uc_models(θ_unbound::FloatVector, lb::FloatVector, ub::FloatVector, transform_id::Array{Int64,1}, model_structure::Function, uc_settings::UCSettings)

    # Compute parameters with bounded support
    θ = copy(θ_unbound);
    for i=1:length(θ)
        if transform_id[i] == 1
            θ[i] = get_bounded_log(θ_unbound[i], lb[i]);
        elseif transform_id[i] == 2
            θ[i] = get_bounded_logit(θ_unbound[i], lb[i], ub[i]);
        end
    end

    # Kalman status and settings
    status = KalmanStatus();
    settings = ImmutableKalmanSettings(model_structure(θ, uc_settings)...);

    # Compute loglikelihood for t = 1, ..., T
    for t=1:size(settings.Y,2)
        kfilter!(settings, status);
    end

    # Return -loglikelihood
    return -status.loglik;
end

"""
    forecast(settings::KalmanSettings, h::Int64)

Compute the h-step ahead forecast for the data included in settings.

# Arguments
- `settings`: KalmanSettings struct
- `h`: Forecast horizon

    forecast(settings::KalmanSettings, X::FloatVector, h::Int64)

Compute the h-step ahead forecast for the data included in settings, starting from X.

# Arguments
- `settings`: KalmanSettings struct
- `X`: Last known value of the latent states
- `h`: Forecast horizon
"""
function forecast(settings::KalmanSettings, h::Int64)

    # Initialise Kalman status
    status = KalmanStatus();

    # Filter for t=1,...,T
    for t=1:size(settings.Y,2)
        kfilter!(settings, status);
    end

    # Return forecast
    return settings.B*hcat(kforecast(settings, status.X_post, h)...);
end

forecast(settings::KalmanSettings, X::FloatVector, h::Int64) = settings.B*hcat(kforecast(settings, X, h)...);

#=
--------------------------------------------------------------------------------------------------------------------------------
ARIMA model
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    arma_structure(θ::FloatVector, settings::ARIMASettings)

ARMA(p,q) representation as in Hamilton (1994).

# Arguments
- `θ`: Model parameters (eigenvalues + variance of the innovation)
- `settings`: ARIMASettings struct
"""
function arma_structure(θ::FloatVector, settings::ARIMASettings)

    # Initialise adjusted parameters
    ϑ = copy(θ);

    # ARMA eigenvalues -> coefficients (this enforces causality and invertibility in the past)
    ϑ[1:settings.q] = eigvals_to_coeff(θ[1:settings.q]);
    ϑ[settings.q+1:settings.q+settings.p] = eigvals_to_coeff(θ[settings.q+1:settings.q+settings.p]);

    # Observation equation
    B = [1 permutedims(ϑ[1:settings.q]) zeros(1,settings.r-settings.q-1)];
    R = Symmetric(ones(1,1)*1e-8);

    # Transition equation
    C = [permutedims(ϑ[settings.q+1:settings.q+settings.p]) zeros(1,settings.r-settings.p);
         Matrix(I, settings.r-1, settings.r-1) zeros(settings.r-1)];
    V = Symmetric(cat(dims=[1,2], ϑ[settings.q+settings.p+1], zeros(settings.r-1, settings.r-1)));

    # Return state-space structure
    return settings.Y, B, R, C, V;
end

"""
    arima(θ::FloatVector, settings::ARIMASettings)

Return KalmanSettings for an arima(d,p,q) model with parameters θ.

# Arguments
- `θ`: Model parameters (eigenvalues + variance of the innovation)
- `settings`: ARIMASettings struct

    arima(settings::ARIMASettings, args...)

Estimate arima(d,p,q) model.

# Arguments
- `settings`: ARIMASettings struct
- `args`: Arguments for Optim.optimize
"""
function arima(θ::FloatVector, settings::ARIMASettings)

    # Compute state-space parameters
    output = ImmutableKalmanSettings(arma_structure(θ, settings)...);

    # Warning 1: invertibility (in the past)
    eigval_ma = eigvals(companion_form(output.B[2:end]));
    if maximum(abs.(eigval_ma)) >= 1
        @warn("Invertibility (in the past) is not properly enforced! \n Re-estimate the model increasing the degree of differencing.");
    end

    # Warning 2: causality
    eigval_ar = eigvals(output.C);
    if maximum(abs.(eigval_ar)) >= 1
        @warn("Causality is not properly enforced! \n Re-estimate the model increasing the degree of differencing.");
    end

    # Warning 3: parameter redundancy
    intersection_ar_ma = intersect(eigval_ar, eigval_ma);
    if length(intersection_ar_ma) > 0
        @warn("Parameter redundancy! \n Check the AR and MA polynomials.");
    end

    # Return output
    return output
end

function arima(settings::ARIMASettings, args...)

    # Starting point
    θ_starting = 1e-8*ones(settings.p+settings.q+1);

    # Bounds
    lb = [-0.99*ones(settings.p+settings.q); 1e-8];
    ub = [0.99*ones(settings.p+settings.q);  Inf];
    transform_id = [2*ones(settings.p+settings.q); 1] |> Array{Int64,1};

    # Estimate the model
    res = Optim.optimize(θ_unbound->fmin_uc_models(θ_unbound, lb, ub, transform_id, arma_structure, settings), θ_starting, args...);

    # Apply bounds
    θ_minimizer = copy(res.minimizer);
    for i=1:length(θ_minimizer)
        if transform_id[i] == 1
            θ_minimizer[i] = get_bounded_log(θ_minimizer[i], lb[i]);
        elseif transform_id[i] == 2
            θ_minimizer[i] = get_bounded_logit(θ_minimizer[i], lb[i], ub[i]);
        end
    end

    # Return output
    return arima(θ_minimizer, settings);
end

"""
    forecast(settings::KalmanSettings, h::Int64, arima_settings::ARIMASettings)

Compute the h-step ahead forecast for the data included in settings (in the arima_settings.Y_levels scale)

# Arguments
- `settings`: KalmanSettings struct
- `h`: Forecast horizon
- `arima_settings`: ARIMASettings struct
"""
function forecast(settings::KalmanSettings, h::Int64, arima_settings::ARIMASettings)

    # TODO: speed up the function

    # Compute forecast for the de-meaned data in settings.Y
    forecast_Y = forecast(settings, h);

    # Initialise adjustment factor
    adj_factor = arima_settings.μ * ones(1,h);

    # Compute adjustment factor
    if arima_settings.d > 0

        # Initialise Y_all
        Y_all = zeros(arima_settings.d, arima_settings.d);

        # The first row of Y_all is the data in levels
        Y_all[1,:] = arima_settings.Y_levels[1, end-arima_settings.d+1:end];

        # Differenced data, ex. (1-L)^d * Y_levels
        for i=1:arima_settings.d-1
            Y_all[1+i,:] = [NaN * ones(1,i) diff(permutedims(Y_all[i,:]), dims=2)];
        end

        # Cut Y_all
        Y_all = Y_all[:,end];

        for hz=1:h

            # Update adjustment factor
            adj_factor[1,hz] += sum(forecast_Y[1,hz] .+ Y_all);

            # Update Y_all
            Y_all[end] += forecast_Y[1,hz];

            for i=1:arima_settings.d-1
                Y_all[end-i] += Y_all[end-i+1];
            end
        end
    end

    # Return forecast
    return adj_factor .+ forecast_Y;
end
