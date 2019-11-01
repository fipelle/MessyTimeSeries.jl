# TSAnalysis.jl
TSAnalysis.jl includes basic tools for time series analysis and state-space modelling.

```julia
import Pkg;
Pkg.add("TSAnalysis")
```

The implementation for the Kalman filter and smoother uses symmetric matrices (via ```LinearAlgebra```). This is particularly beneficial for the stability and speed of estimation algorithms (e.g., the EM algorithm in Shumway and Stoffer, 1982), and to handle high-dimensional forecasting problems.

## Examples
For these examples, I will use economic data from FRED (https://fred.stlouisfed.org/), which is available in the ```FredData``` package. Instead, the charts are produced using ```Plots```. These packages can be added via:

```julia
import Pkg;
Pkg.add("FredData");
Pkg.add("Plots");
```

To download the data, use the following code:
```julia
# Load FredData
using FredData;
f = Fred();

# Download Industrial Production: Manufacturing (NAICS), Log-levels (monthly, NSA)
fred_df = get_data(f, "IPGMFN", observation_start="1984-01-01", units="log");

# Store data in Array{Float64,1}
Y = fred_df.data.value;
```

Make sure that your FRED API is accessible to ```FredData``` (as in https://github.com/micahjsmith/FredData.jl). Hence, load ```LinearAlgebra```, ```Plots``` and ```TSAnalysis``` via
```julia
using LinearAlgebra;
using Plots;
using TSAnalysis;
```

### Kalman filter
The following examples show how to perform standard univariate state-space decompositions. 

#### Local linear trend + noise decomposition
```julia
# Initialise the Kalman filter and smoother status
kstatus = KalmanStatus();

# Specify the state-space structure
ksettings = ImmutableKalmanSettings(permutedims(Y),
                                    [1.0 0.0], Symmetric(ones(1,1)*0.01),                # Observation equation
                                    [1.0 1.0; 0.0 1.0], Symmetric([1e-4 0.0; 0.0 1e-4]), # Transition equation
                                    zeros(2), Symmetric(1e3*Matrix(I,2,2)));             # Initial conditions

# Filter for t = 1, ..., T (the output is dynamically stored into kstatus)
for t=1:size(Y,1)
    kfilter!(ksettings, kstatus);
end

# Filtered trend
trend_llt = hcat(kstatus.history_X_post...)[1,:];
```

#### Local linear trend + seasonal + noise decomposition
```julia
# Initialise the Kalman filter and smoother status
kstatus = KalmanStatus();

# Specify the state-space structure

# Observation equation
B = hcat([1.0 0.0], [[1.0 0.0] for j=1:6]...);
R = Symmetric(ones(1,1)*0.01);

# Transition equation
C = cat(dims=[1,2], [1.0 1.0; 0.0 1.0], [[cos(2*pi*j/12) sin(2*pi*j/12); -sin(2*pi*j/12) cos(2*pi*j/12)] for j=1:6]...);
V = Symmetric(cat(dims=[1,2], [1e-4 0.0; 0.0 1e-4], 1e-4*Matrix(I,12,12)));

# Initial conditions
X0 = zeros(14);
P0 = Symmetric(cat(dims=[1,2], 1e3*Matrix(I,2,2), zeros(12,12)));

# Settings
ksettings = ImmutableKalmanSettings(permutedims(Y), B, R, C, V, X0, P0);

# Filter for t = 1, ..., T (the output is dynamically stored into kstatus)
for t=1:size(Y,1)
    kfilter!(ksettings, kstatus);
end

# Filtered trend
trend_llts = hcat(kstatus.history_X_post...)[1,:];
```

### Kalman filter (prediction)
```TSAnalysis``` allows to compute *h*-steps ahead predictions (at any point in time) without resetting the Kalman filter. 

#### Local linear trend + seasonal + noise decomposition
In order to compute the 12-steps ahead prediction the block
```julia
# Filter for t = 1, ..., T (the output is dynamically stored into kstatus)
for t=1:size(Y,1)
    kfilter!(ksettings, kstatus);
end
```

must be edited into
```julia
forecast_history = Array{Array{Float64,1},1}();

# Filter for t = 1, ..., T (the output is dynamically stored into kstatus)
for t=1:size(Y,1)
    kfilter!(ksettings, kstatus);
    push!(forecast_history, (B*hcat(kforecast(ksettings, kstatus.X_post, 12)...))[:]);
end
```

### Kalman smoother
At any point in time, the Kalman smoother can be executed via
```julia
ksmoother(ksettings, kstatus);
```

## Bibliography
* R. H. Shumway and D. S. Stoffer. An approach to time series smoothing and forecasting using the EM algorithm. Journal of time series analysis, 3(4):253–264, 1982.
