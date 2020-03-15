#=
--------------------------------------------------------------------------------------------------------------------------------
UC models: general interface
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    penalty_eigen(λ::Float64)

Return penalty value for a single eigenvalue λ.
"""
penalty_eigen(λ::Float64) = abs(λ) < 1 ? abs(λ)/(1-abs(λ)) : Inf;
penalty_eigen(λ::Complex{Float64}) = abs(λ) < 1 ? abs(λ)/(1-abs(λ)) : Inf;

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

    if ~isinf(model_penalty)
        settings = ImmutableKalmanSettings(model_instance...);

        # Compute loglikelihood for t = 1, ..., T
        for t=1:size(settings.Y,2)
            kfilter!(settings, status);
        end

        # Return fmin
        return -status.loglik + tightness*model_penalty;
    else
        return 1/eps();
    end
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

    # Compute the period referring to the last observation of each series
    last_observations = zeros(settings.n) |> Array{Int64,1};
    for i=1:settings.n
        last_observations[i] = findall(.~ismissing.(settings.Y[i,:]))[end];
    end

    # Starting point for the forecast
    starting_point = minimum(last_observations);

    # Filter for t=1,...,T
    for t=1:settings.T
        kfilter!(settings, status);
    end

    # Initial forecast: series with a shorter history are forecasted until they match the others
    fc = zeros(settings.m, settings.T-starting_point+h);
    if starting_point < settings.T
        fc[:,1:settings.T-starting_point] = hcat(status.history_X_post[starting_point+1:settings.T]...);
    end

    # h-steps ahead forecast of the states from last observed point
    fc[:,settings.T-starting_point+1:end] = hcat(kforecast(settings, status.X_post, h)...);

    # Compute forecast for Y
    Y_fc = settings.B*fc;
    if settings.T-starting_point > 0
        Y_fc[last_observations.==settings.T, settings.T-starting_point] .= NaN;
    end
    
    # Return forecast for Y
    return Y_fc;
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
- `θ`: Model parameters
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
         Matrix(I, settings.nr-settings.n, settings.nr-settings.n) zeros(settings.nr-settings.n, settings.n)];

    # VARMA(p,q) var-cov matrix
    V1 = Diagonal(ϑ[settings.nnq+settings.nnp+1:settings.nnq+settings.nnp+settings.n]) |> FloatMatrix;

    # Transition equation: variance
    V = Symmetric(cat(dims=[1,2], V1, zeros(settings.nr-settings.n, settings.nr-settings.n)));

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
- `θ`: Model parameters
- `settings`: VARIMASettings struct

    varima(settings::VARIMASettings, args...; tightness::Float64=1.0)

Estimate varima(d,p,q) model.

# Arguments
- `settings`: VARIMASettings struct
- `args`: Arguments for Optim.optimize
- `tightness`: Controls the strength of the penalty for the non-causal / non-invertible case (default = 1)
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

function varima(settings::VARIMASettings, args...; tightness::Float64=1.0)

    # Starting point
    θ_starting = 1e-4*ones(settings.nnp+settings.nnq+settings.n);

    # Bounds
    lb = [-0.99*ones(settings.nnp+settings.nnq); 1e-6*ones(settings.n)];
    ub = [0.99*ones(settings.nnp+settings.nnq);  Inf*ones(settings.n)];
    transform_id = [2*ones(settings.nnp+settings.nnq); ones(settings.n)] |> Array{Int64,1};

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
    arima(settings::VARIMASettings, args...; tightness::Float64=1.0)

Define an alias of the varima function for arima models.
"""
arima(settings::VARIMASettings, args...; tightness::Float64=1.0) = varima(settings, args..., tightness=tightness);

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

        Y = zeros(varima_settings.d, varima_settings.n);

        # Loop over each series
        for i=1:varima_settings.n

            # Initialise Y_all
            Y_all = zeros(varima_settings.d, varima_settings.d);

            # Last observed point
            last_observation = findall(.~ismissing.(varima_settings.Y_levels[i,:]))[end];

            # The first row of Y_all is the data in levels
            Y_all[1,:] = varima_settings.Y_levels[i, last_observation-varima_settings.d+1:last_observation];

            # Differenced data, ex. (1-L)^d * Y_levels
            for j=1:varima_settings.d-1
                Y_all[1+j,:] = [NaN * ones(1,j) permutedims(diff(Y_all[j,:]))];
            end

            # Cut Y_all
            Y[:,i] = permutedims(Y_all[:,end]);
        end

        # Initial cumulated forecast
        fc_differenced = forecast(settings, h) .+ varima_settings.μ;
        fc = zeros(size(fc_differenced));

        # Loop over d to compute a prediction for the levels
        for i=1:varima_settings.n
            starting_point = findall(.~isnan.(fc_differenced[i,:]))[1];
            fc[i,starting_point:end] .= cumsum(fc_differenced[i,starting_point:end], dims=2);

            for j=varima_settings.d:-1:1
                fc[i,starting_point:end] .+= Y[j,i];
                if j != 1
                    fc[i,starting_point:end] = cumsum(fc[i,starting_point:end]);
                end
            end
        end

        # Insert NaNs
        fc[isnan.(fc_differenced)] .= NaN;

    # VARMA
    else
        # Compute forecast for varima_settings.Y (adjusted by its mean)
        fc = forecast(settings, h) .+ varima_settings.μ;
    end

    return fc;
end
