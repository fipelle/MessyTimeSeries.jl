"""
    fmin_uc_models(θ::FloatVector, model_structure::Function, settings::UCSettings)

Return -1*loglikelihood for the UC model specified by model_structure(settings)

# Arguments
- `θ`: Model parameters
- `model_structure`: Function to setup the state-space structure
- `settings`: Settings for model_structure
"""
function fmin_uc_models(θ::FloatVector, model_structure::Function, settings::UCSettings)

    # Kalman status and settings
    kstatus = KalmanStatus();
    ksettings = ImmutableKalmanSettings(model_structure(θ, settings)...);

    # Compute loglikelihood for t = 1, ..., T
    for t=1:size(ksettings.Y,2)
        kfilter!(ksettings, kstatus);
    end

    # Return -loglikelihood
    return -kstatus.loglik;
end

#=
--------------------------------------------------------------------------------------------------------------------------------
ARIMA models
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    arma_structure(θ::FloatVector, settings::ARIMASettings)

ARMA(p,q) representation as in Hamilton (1994).

# Arguments
- `θ`: Model parameters
- `settings`: ARIMASettings struct
"""
function arma_structure(θ::FloatVector, settings::ARIMASettings)

    # Observation equation
    B = [1 permutedims(θ[1:settings.r-1])];
    R = Symmetric(ones(1,1)*1e-8);

    # Transition equation
    C = [permutedims(θ[settings.r:2*settings.r-1]); Matrix(I, settings.r-1, settings.r-1) zeros(settings.r-1)];
    V = Symmetric(cat(dims=[1,2], θ[2*settings.r], zeros(settings.r-1, settings.r-1)));

    # Return state-space structure
    return settings.Z, B, R, C, V;
end

"""
    arima(settings::ARIMASettings)

Estimate arima(d,p,q) model.

# Arguments
- `settings`: ARIMASettings struct

    arima(settings::ARIMASettings)

Return KalmanSettings for an arima(d,p,q) model with parameters θ.

# Arguments
- `θ`: Model parameters
- `settings`: ARIMASettings struct
"""
function arima(settings::ARIMASettings; rt::Float64=0.95, f_tol::Float64=1e-3, x_tol::Float64=1e-3, max_iter::Int64=10^5)

    # Optim options
    optim_opts = Optim.Options(iterations=max_iter, show_trace=true, show_every=100);

    # Starting point
    θ_starting = 1e-8*ones(2*settings.r);

    # Bounds
    θ_lower = [-0.9*ones(2*settings.r-1); 1e-8*ones(1)];
    θ_upper = [0.9*ones(2*settings.r-1);  100*ones(1)];

    # Estimate the model
    # TODO: control non-stationary case
    res = Optim.optimize(θ->fmin_uc_models(θ, arma_structure, settings), θ_lower, θ_upper, θ_starting, SAMIN(rt=rt, f_tol=f_tol, x_tol=x_tol), optim_opts);

    # Return output
    # TODO: add ad-hoc ImmutableKalmanSettings
    return arima(res.minimizer, settings);
end

function arima(θ::FloatVector, settings::ARIMASettings)
    return ImmutableKalmanSettings(arma_structure(θ, settings)...);
end
