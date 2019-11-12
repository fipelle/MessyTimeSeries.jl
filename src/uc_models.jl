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
- `KalmanSettings`: KalmanSettings struct
- `h`: Forecast horizon

    forecast(settings::KalmanSettings, X::FloatVector, h::Int64)

Compute the h-step ahead forecast for the data included in settings, starting from X.

# Arguments
- `KalmanSettings`: KalmanSettings struct
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
- `θ`: Model parameters (real eigenvalues + variance of the innovation)
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
    arima(settings::ARIMASettings; f_tol::Float64=1e-4, x_tol::Float64=1e-4, max_iter::Int64=10^5, verb::Bool=true)

Estimate arima(d,p,q) model.

# Arguments
- `settings`: ARIMASettings struct
- `f_tol`: Function tolerance (default: 1e-2)
- `x_tol`: Parameters tolerance (default: 1e-2)
- `max_iter`: Maximum number of iterations (default: 10^5)
- `verb`: Verbose output from Optim (default: true)

    arima(θ::FloatVector, settings::ARIMASettings)

Return KalmanSettings for an arima(d,p,q) model with parameters θ.

# Arguments
- `θ`: Model parameters (real eigenvalues + variance of the innovation)
- `settings`: ARIMASettings struct
"""
function arima(settings::ARIMASettings; f_tol::Float64=1e-4, x_tol::Float64=1e-4, max_iter::Int64=10^5, verb::Bool=true)

    # Optim options
    optim_opts = Optim.Options(iterations=max_iter, f_tol=f_tol, x_tol=x_tol, show_trace=verb, show_every=500);

    # Starting point
    θ_starting = 1e-8*ones(settings.p+settings.q+1);

    # Bounds
    lb = [-0.99*ones(settings.p+settings.q); 1e-8];
    ub = [0.99*ones(settings.p+settings.q);  Inf];
    transform_id = [2*ones(settings.p+settings.q); 1] |> Array{Int64,1};

    # Estimate the model
    # TODO: debug arima and transf functions
    res = Optim.optimize(θ_unbound->fmin_uc_models(θ_unbound, lb, ub, transform_id, arma_structure, settings), θ_starting, NelderMead(), optim_opts);

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

function arima(θ::FloatVector, settings::ARIMASettings)

    # Compute state-space parameters
    output = ImmutableKalmanSettings(arma_structure(θ, settings)...);

    # Warning 1: invertibility (in the past)
    eigval_ma = maximum(abs.(eigvals(companion_form(output.B[2:end]))));
    if eigval_ma >= 1
        @warn("Invertibility (in the past) is not properly enforced! \n Re-estimate the model increasing the degree of differencing.");
    end

    # Warning 2: causality
    eigval_ar = maximum(abs.(eigvals(output.C)));
    if eigval_ar >= 1
        @warn("Causality is not properly enforced! \n Re-estimate the model increasing the degree of differencing.");
    end

    # TODO: check for parameter redundancy

    # Return output
    return output
end

"""
"""
function forecast(settings::KalmanSettings, h::Int64, arima_settings::ARIMASettings)

    # TODO: finish writing this function

    # Compute forecast for the de-meaned data in settings.Y
    forecast_Y = forecast(settings, h);

    # Compute adjustment factor
    adj_factor = zeros(1,h);
    Y_i = copy(arima_settings.Y_levels);
    if arima_settings.d > 0
        for i=1:d
            Y_i = diff(Y_i, dims=2);
            adj_factor[1] += Y_i[end];
        end
    end

    adj_factor[1] += mean_skipmissing(Y_i)[1];

    # Return forecast
    return adj_factor[1] .+ forecast_Y;
end
