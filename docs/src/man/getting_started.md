# Getting started

This page contains a few examples to get started with ```MessyTimeSeries```. These examples are based on economic data from [FRED](https://fred.stlouisfed.org/), downloaded via ```FredData```. Make sure that your FRED API is accessible to ```FredData``` (see [here](https://github.com/micahjsmith/FredData.jl)).

Before getting into the examples, please run the following snippet:
```@example getting_started
# Load packages
using Dates, DataFrames, LinearAlgebra, Optim, Plots, Measures, StableRNGs;
using FredData: Fred, get_data;
using MessyTimeSeries;

# Initialise FredData
f = Fred();

# Download seasonal adjusted (SA) industrial production (IP)
dates = get_data(f, "INDPRO", observation_start="1984-01-01", units="lin").data[!, :date];
sa_ip = get_data(f, "INDPRO", observation_start="1984-01-01", units="lin").data[!, :value];

# Download not seasonally adjusted (NSA) industrial production (IP)
nsa_ip = get_data(f, "IPB50001N", observation_start="1984-01-01", units="lin").data[:, :value];

# Set random seed
rng = StableRNG(1);
```

!!! note "Advanced estimation and validation algorithms"
    Advanced estimation and validation algorithms are not included in this package, but in [```MessyTimeSeriesOptim```](https://github.com/fipelle/MessyTimeSeriesOptim.jl).

!!! warning
    Please note that this tutorial evaluates the following sections in the order reported below.

## Centred moving averages

Centred moving averages have numerous purposes in time series analysis. ```MessyTimeSeries``` has a simple implementation for it. Suppose, for instance, that you are interested in computing the centred moving average of NSA Industrial Production to reduce the effect of seasonality. This can be done running:
```@example getting_started
nsa_ip_cma = centred_moving_average(permutedims(nsa_ip), 13);
nothing # hide
```

The size of `nsa_ip_cma` is identical to the one of the original data, but includes `6` leading and trailing missing observations, since the window is set to 13. The results can be inspected graphically via
```@example getting_started
fig = plot(dates, nsa_ip[:], label="Industrial production (NSA)", legend=:topleft, ylims=(Inf, 120))
plot!(fig, dates, nsa_ip_cma[:], label="Industrial production (NSA, CMA)", ylims=(Inf, 120));
```

## Two-sided interpolation

`MessyTimeSeries` provides two simplified interfaces for interpolating time-series data. More advanced interpolations can be implemented via the state-space modelling functions described in the next section.

!!! note
    The following examples remove about 50% of the observed data to produce more evident interpolations graphically.

### Mean-reverting data

Compute the month-on-month (%) changes of industrial production and input a number of missing observations via
```@example getting_started
# Compute MoM (%) data
sa_ip_mom = 100*(sa_ip[2:end]./sa_ip[1:end-1] .- 1);

# Populate `missings_coordinates`
missings_coordinates = unique(rand(rng, 1:length(sa_ip_mom), 300));

# Input missing observations
sa_ip_mom_incomplete = permutedims(sa_ip_mom) |> JMatrix{Float64};
sa_ip_mom_incomplete[1, missings_coordinates] .= missing;
nothing # hide
```

The function `interpolate_series` replaces the missing observations with the average of the observed values. Results can be inspected graphically as follows:
```@example getting_started
# Interpolate
sa_ip_mom_interpolated = interpolate_series(sa_ip_mom_incomplete);

# Plots
fig = plot(dates[2:end], sa_ip_mom, label="Industrial production (SA, MoM%)", legend=:topleft, ylims=(-16, 8));
plot!(fig, dates[2:end], permutedims(sa_ip_mom_interpolated), label="Industrial production (SA, MoM%, interpolated)", legend=:topleft, ylims=(-16, 8));
```

### Non-stationary data

Input a number of missing observations to the SA industrial production index via
```@example getting_started
sa_ip_incomplete = permutedims(sa_ip) |> JMatrix{Float64};
sa_ip_incomplete[1, missings_coordinates] .= missing;
nothing # hide
```

The function `forward_backwards_rw_interpolation` interpolates `sa_ip_incomplete` using a random walk logic both forward and backwards in time. Results can be inspected graphically as follows:
```@example getting_started
# Interpolate
sa_ip_interpolated = forward_backwards_rw_interpolation(sa_ip_incomplete);

# Plots
fig = plot(dates, sa_ip, label="Industrial production (SA)", legend=:topleft, ylims=(Inf, 120));
plot!(fig, dates, permutedims(sa_ip_interpolated), label="Industrial production (SA, interpolated)", legend=:topleft, ylims=(Inf, 120));
```

## Optimal filtering and smoothing

A large chunk of this package focusses on optimal filtering and smoothing problems (Anderson and Moore, 2012). This is an elegant way to model time series that show irregularities such as missing observations with applications ranging from real-time forecasting to structural decompositions (see, for instance, Durbin and Koopman, 2012).  

In this short tutorial, I have illustrated a simple way to seasonal adjust non-stationary time series. This should be enough to get the handle of the basic tools included in this package.

!!! note "Advanced state-space models"
    Advanced state-space models can be easily processed through the functions described below. However, estimating them is often hard, especially with incomplete data. Specialised methods for these problems are implemented in [```MessyTimeSeriesOptim```](https://github.com/fipelle/MessyTimeSeriesOptim.jl).

### Seasonal adjustments

#### Model

Suppose that you would like to seasonally adjust the NSA Industrial Production index. One way to do it is writing a state-space model comprising a trend (``\mu_{t}``) and a latent component identifying the seasonal factor (``\gamma_{t}``):

```math
\begin{alignat*}{2}
    Y_{t}      &= \mu_{t} + \gamma_{t}, \\
    \mu_{t}    &= \beta_{t-1} + \mu_{t-1} + u_{t}, \qquad &&u_{t} \sim N(0, \sigma_{u}^2), \\
    \beta_{t}  &= \beta_{t-1} + v_{t}, \qquad &&v_{t} \sim N(0, \sigma_{v}^2), \\
    \gamma_{t} &= \sum_{j=1}^{s/2} \gamma_{j,t},
\end{alignat*}
```

where

```math
\begin{alignat*}{2}
    \left( \begin{array}{c} \gamma_{j,t} \\ \gamma_{j,t}^{*} \end{array} \right) = \left( \begin{array}{cc} \cos \lambda_j & \sin \lambda_j \\ -\sin \lambda_j & \cos \lambda_j \end{array} \right) \left( \begin{array}{c} \gamma_{j,t-1} \\ \gamma_{j,t-1}^{*} \end{array} \right) + \left( \begin{array}{c} w_{j,t} \\ w_{j,t}^{*} \end{array} \right), \qquad \left( \begin{array}{c} w_{j,t} \\ w_{j,t}^{*} \end{array} \right) \sim N(0, \sigma_{w}^2 I),
\end{alignat*}
```

``\lambda_{j} = 2 \pi j/s``, ``s=12``, ``j=1, \ldots, s/2`` and ``t=1, \ldots, T``.

#### Estimation

Note that this model can be written in the compact form

```math
\begin{alignat*}{2}
    Y_{t} &= B*X_{t} + e_{t} \\
    X_{t} &= C*X_{t-1} + U_{t}
\end{alignat*}
```

where ``e_{t} \sim N(0, 10^{-4})`` and ``U_{t} \sim N(0, Q)`` building the coefficients via
```@example getting_started
# Convenient function for B
build_B(s::Int64) = hcat([[1.0 0.0] for i=1:1+fld(s,2)]...);

# Convenient functions for C
build_C_gamma_j(λj::Float64) = [cos(λj) sin(λj); -sin(λj) cos(λj)];
build_C_gamma(s::Int64) = cat(dims=[1,2], [build_C_gamma_j(2*pi*j/s) for j=1:fld(s,2)]...);
build_C(s::Int64) = cat(dims=[1,2], [1 1; 0 1], build_C_gamma(s));

# Convenient functions for Q
build_Q_mu_beta(var_mu::Float64, var_beta::Float64) = Diagonal([var_mu, var_beta]) |> Array;
build_Q_gamma(var_gamma::Float64, s::Int64) = Diagonal(kron(var_gamma*ones(fld(s,2)), [1;0])) |> Array;
build_Q(var_mu::Float64, var_beta::Float64, var_gamma::Float64, s::Int64) = cat(dims=[1,2], build_Q_mu_beta(var_mu, var_beta), build_Q_gamma(var_gamma, s));
nothing # hide
```

The first thing to do for estimating the model using ```Optim``` is to define a relevant objective function to minimise. In this tutorial, I have build on the Kalman filter output to perform MLE:
```@example getting_started
function fmin(Y::Matrix{Float64}, vec_log_sigma::FloatVector)
    B = build_B(12);
    C = build_C(12);
    R = Symmetric(10^-4*ones(1,1));
    D = 1.0*Matrix(I, size(B,2), size(B,2));
    Q = Symmetric(build_Q(exp.(vec_log_sigma)..., 12));

    # Initial conditions
    X0 = zeros(size(B,2));
    P0 = Symmetric(1000.0*Matrix(I, size(B,2), size(B,2))); # diffuse initialisation

    # Run Kalman filter and return log-likelihood
    settings = KalmanSettings(Y, B, R, C, D, Q, X0, P0);
    status = kfilter_full_sample(settings);
    return -status.loglik;
end
nothing # hide
```

!!! note "Free parameters"
    In this simple model, the only free parameters are the variances of the trend, drift and seasonal components in the matrix ```Q```.

Next, you need to run an appropriate ```Optim``` minimisation algorithm, such as:
```@example getting_started
optim_res = optimize(vec_sigma->fmin(permutedims(nsa_ip), vec_sigma), log(0.1)*ones(3), SimulatedAnnealing()); # arbitrary small variances to initialise the model
optim_res.minimizer, optim_res.minimum
```

The optimal model configuration found by ```Optim``` is such that:
```@example getting_started
B = build_B(12);
C = build_C(12);
R = Symmetric(10^-4*ones(1,1));
D = 1.0*Matrix(I, size(B,2), size(B,2));
Q = Symmetric(build_Q(exp.(optim_res.minimizer)..., 12));

# Initial conditions
X0 = zeros(size(B,2));
P0 = Symmetric(1000.0*Matrix(I, size(B,2), size(B,2))); # diffuse initialisation

# Run Kalman filter and smoother
settings = KalmanSettings(permutedims(nsa_ip), B, R, C, D, Q, X0, P0);
status = kfilter_full_sample(settings);
history_X, history_P, X0, P0 = ksmoother(settings, status);
nothing # hide
```

```@example getting_started
p1 = plot(dates, nsa_ip, label="Industrial production (NSA)", legend=:topleft, ylims=(Inf, 120));
plot!(p1, dates, hcat(history_X...)[1,:], label="Industrial production trend", legend=:topleft, ylims=(Inf, 120));
```

```@example getting_started
seasonality = sum(hcat(history_X...)[3:end,:], dims=1);
p2 = plot(dates, permutedims(seasonality), label="Industrial production seasonality", legend=:topleft, ylims=(-6, 6))
```