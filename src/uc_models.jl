#=
--------------------------------------------------------------------------------------------------------------------------------
UC models: general interface
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    penalty_eigen(λ::Float64)

Return penalty value for a single eigenvalue λ.
"""
penalty_eigen(λ::Float64) = abs(λ) < 1 ? abs(λ)/(1-abs(λ)) : 1/eps();

"""
    fmin_uc_models(θ_unbound::FloatVector, lb::FloatVector, ub::FloatVector, transform_id::Array{Int64,1}, model_structure::Function, settings::UCSettings)

Return fmin for the UC model specified by model_structure(settings)

# Arguments
- `θ_unbound`: Model parameters (with unbounded support)
- `lb`: Lower bound for the parameters
- `ub`: Upper bound for the parameters
- `transform_id`: Type of transformation required for the parameters (0 = none, 1 = generalised log, 2 = generalised logit)
- `model_structure`: Function to setup the state-space structure
- `settings`: Settings for model_structure
- `tightness`: Controls the strength of the penalty (if any)
"""
function fmin_uc_models(θ_unbound::FloatVector, lb::FloatVector, ub::FloatVector, transform_id::Array{Int64,1}, model_structure::Function, uc_settings::UCSettings, tightness::Float64)

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
    model_instance, model_penalty = model_structure(θ, uc_settings);
    settings = ImmutableKalmanSettings(model_instance...);

    # Compute loglikelihood for t = 1, ..., T
    for t=1:size(settings.Y,2)
        kfilter!(settings, status);
    end

    # Return fmin
    return -status.loglik + tightness*model_penalty;
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
VARIMA model
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    varma_structure(θ::FloatVector, settings::VARIMASettings)

VARMA(p,q) representation similar to the form reported in Hamilton (1994) for ARIMA(p,q) models.

# Arguments
- `θ`: Model parameters (coefficients + variance of the innovation)
- `settings`: VARIMASettings struct
"""
function varma_structure(θ::FloatVector, settings::VARIMASettings)

    # Initialise
    ϑ = copy(θ);
    I_n = Matrix(I, settings.n, settings.n) |> FloatMatrix;
    UT_n = UpperTriangular(ones(settings.n, settings.n)) |> FloatMatrix;
    UT_n[I_n.==1] .= 0;

    # Observation equation
    B = [I_n reshape(ϑ[1:settings.nnq], settings.n, settings.nq) zeros(settings.n, settings.nr-settings.nq-settings.n)];
    R = Symmetric(I_n*1e-8);

    # Transition equation: coefficients
    C = [reshape(ϑ[settings.nnq+1:settings.nnq+settings.nnp], settings.n, settings.np) zeros(settings.n, settings.nr-settings.np);
         Matrix(I, settings.np-settings.n, settings.np-settings.n) zeros(settings.np-settings.n, settings.n)];

    # Initialise VARMA(p,q) var-cov matrix
    V1 = zeros(settings.n, settings.n);

    # Main diagonal: variances
    V1[I_n.==1] .= ϑ[settings.nnq+settings.nnp+1:settings.nnq+settings.nnp+settings.n];

    # Out-of-diagonal elements: covariances in the upper triangular part of the var-cov matrix
    V1[UT_n.==1] .= ϑ[settings.nnq+settings.nnp+settings.n+1:end];

    # Transition equation: variance
    V = Symmetric(cat(dims=[1,2], Symmetric(V1), zeros(settings.np-settings.n, settings.np-settings.n)));

    # Companion form for the moving average part
    companion_vma = [B[:,settings.n+1:settings.n+settings.nq];
                     Matrix(I, settings.nq-settings.n, settings.nq-settings.n) zeros(settings.nq-settings.n, settings.n)];

    # Compute penalty
    varma_penalty = 0.0;
    for λ = [eigvals(C); eigvals(companion_vma)]
        varma_penalty += penalty_eigen(λ);
    end

    # Return state-space structure
    return (settings.Y, B, R, C, V), varma_penalty;
end

"""
    varima(θ::FloatVector, settings::VARIMASettings)

Return KalmanSettings for a varima(d,p,q) model with parameters θ.

# Arguments
- `θ`: Model parameters (coefficients + variance of the innovation)
- `settings`: VARIMASettings struct

    varima(settings::VARIMASettings, args...)

Estimate varima(d,p,q) model.

# Arguments
- `settings`: VARIMASettings struct
- `tightness`: Controls the strength of the penalty for the non-causal / non-invertible case
- `args`: Arguments for Optim.optimize
"""
function varima(θ::FloatVector, settings::VARIMASettings)

    # Compute state-space parameters
    model_instance, _ = varma_structure(θ, settings);
    output = ImmutableKalmanSettings(model_instance...);

# TBD: update the warnings
#=
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
=#

    # Return output
    return output
end

function varima(settings::VARIMASettings, tightness::Float64, args...)

    # No. covariances
    n_cov = settings.n*(settings.n-1)/2;

    # Starting point
    θ_starting = 1e-8*ones(settings.np+settings.nq+settings.n+n_cov);

    # Bounds
    lb = [-0.99*ones(settings.np+settings.nq); 1e-8*ones(settings.n); -Inf*ones(n_cov)];
    ub = [0.99*ones(settings.np+settings.nq);  Inf*ones(settings.n); Inf*ones(n_cov)];
    transform_id = [2*ones(settings.np+settings.nq); ones(settings.n); zeros(n_cov)] |> Array{Int64,1};

    # Estimate the model
    res = Optim.optimize(θ_unbound->fmin_uc_models(θ_unbound, lb, ub, transform_id, varma_structure, settings, tightness), θ_starting, args...);

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
    return varima(θ_minimizer, settings);
end

"""
    forecast(settings::KalmanSettings, h::Int64, varima_settings::VARIMASettings)

Compute the h-step ahead forecast for the data included in settings (in the varima_settings.Y_levels scale)

# Arguments
- `settings`: KalmanSettings struct
- `h`: Forecast horizon
- `varima_settings`: VARIMASettings struct
"""
function forecast(settings::KalmanSettings, h::Int64, varima_settings::VARIMASettings)

    # VARIMA
    if varima_settings.d > 0

        # Initialise Y_all
        Y_all = zeros(varima_settings.d, varima_settings.d);

        # The first row of Y_all is the data in levels
        Y_all[1,:] = varima_settings.Y_levels[end-varima_settings.d+1:end];

        # Differenced data, ex. (1-L)^d * Y_levels
        for i=1:varima_settings.d-1
            Y_all[1+i,:] = [NaN * ones(1,i) permutedims(diff(Y_all[i,:]))];
        end

        # Cut Y_all
        Y_all = Y_all[:,end];

        # Initial cumulated forecast
        fc = cumsum(forecast(settings, h) .+ varima_settings.μ, dims=2);

        # Loop over d to compute a prediction for the levels
        for i=varima_settings.d:-1:1
            fc .+= Y_all[i];
            if i != 1
                fc = cumsum(fc, dims=2);
            end
        end

    # VARMA
    else
        # Compute forecast for varima_settings.Y (adjusted by its mean)
        fc = forecast(settings, h) .+ varima_settings.μ;
    end

    return fc;
end
