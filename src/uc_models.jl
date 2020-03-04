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
VARIMA model
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    varma_structure(θ::FloatVector, settings::VARIMASettings)

VARMA(p,q) representation similar to the form reported in Hamilton (1994) for ARIMA(p,q) models.

# Arguments
- `θ`: Model parameters (eigenvalues + variance of the innovation)
- `settings`: VARIMASettings struct
"""
function varma_structure(θ::FloatVector, settings::VARIMASettings)

    # Initialise
    ϑ = copy(θ);
    I_n = Matrix(I, settings.n, settings.n) |> Array{Float64};
    UT_n = UpperTriangular(ones(settings.n, settings.n)) |> Array;
    UT_n[I_n.==1] .= 0;

    # VARMA(p,q) eigenvalues -> coefficients (this enforces causality and invertibility in the past)
    ϑ[1:settings.nnq] = eigvals_to_coeff(θ[1:settings.nnq]);
    ϑ[settings.nnq+1:settings.nnq+settings.nnp] = eigvals_to_coeff(θ[settings.nnq+1:settings.nnq+settings.nnp]);

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

    # Return state-space structure
    return settings.Y, B, R, C, V;
end

"""
    varima(θ::FloatVector, settings::VARIMASettings)

Return KalmanSettings for a varima(d,p,q) model with parameters θ.

# Arguments
- `θ`: Model parameters (eigenvalues + variance of the innovation)
- `settings`: VARIMASettings struct

    varima(settings::VARIMASettings, args...)

Estimate varima(d,p,q) model.

# Arguments
- `settings`: VARIMASettings struct
- `args`: Arguments for Optim.optimize
"""
function varima(θ::FloatVector, settings::VARIMASettings)

    # Compute state-space parameters
    output = ImmutableKalmanSettings(varma_structure(θ, settings)...);

# TBD: update
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

function varima(settings::VARIMASettings, args...)

    # No. covariances
    no_cov = settings.n*(settings.n-1)/2;

    # Starting point
    θ_starting = 1e-8*ones(settings.np+settings.nq+settings.n+no_cov);

    # Bounds
    lb = [-0.99*ones(settings.np+settings.nq); 1e-8*ones(settings.n); -Inf*ones(no_cov)];
    ub = [0.99*ones(settings.np+settings.nq);  Inf*ones(settings.n); Inf*ones(no_cov)];
    transform_id = [2*ones(settings.np+settings.nq); ones(settings.n); zeros(no_cov)] |> Array{Int64,1};

    # Estimate the model
    res = Optim.optimize(θ_unbound->fmin_uc_models(θ_unbound, lb, ub, transform_id, varma_structure, settings), θ_starting, args...);

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
