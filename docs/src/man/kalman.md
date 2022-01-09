# Kalman filter and smoother

## Types

The Kalman filter and smoother are controlled by two custom types: `KalmanSettings` and `KalmanStatus`. `KalmanSettings` contains the data, model parameters and a few auxiliary variables useful to speed-up the filtering and smoothing routines. `KalmanStatus` is an abstract supertype denoting a container for the filter and low-level smoother output. This is specified into the following three structures:
- `OnlineKalmanStatus`: lightweight and specialised for high-dimensional online filtering problems;
- `DynamicKalmanStatus`: specialised for filtering and smoothing problems in which the total number of observed time periods can be dynamically changed;
- `SizedKalmanStatus`: specialised for filtering and smoothing problems in which the total number of observed time periods is fixed to ``T``.

### Kalman filter and smoother input

`KalmanSettings` can be constructed field by field or through a series of constructors that require a significantly smaller amount of variables.

```@docs
KalmanSettings
```

```@docs
KalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, Q::SymMatrix; kwargs...)
KalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix; kwargs...)
KalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatVector, P0::SymMatrix; kwargs...)
```

### Kalman filter and low-level smoother output

Each subtype of `KalmanStatus` can be either initialised manually or through a simplified method. The low-level smoother output are convienent buffer arrays used for low-level matrix operations.

#### OnlineKalmanStatus

```@docs
OnlineKalmanStatus
OnlineKalmanStatus()
```

#### DynamicKalmanStatus

```@docs
DynamicKalmanStatus
DynamicKalmanStatus()
```

#### SizedKalmanStatus

```@docs
SizedKalmanStatus
SizedKalmanStatus(T::Int64)
```

## Functions

### Kalman filter

The a-priori prediction and a-posteriori update can be computed for a single point in time via `kfilter!` or up to ``T`` with `kfilter_full_sample` and `kfilter_full_sample!`.

```@docs
kfilter!
```

```@docs
kfilter_full_sample
```

```@docs
kfilter_full_sample!
```

```@docs
kforecast
```

### Kalman smoother

```@docs
ksmoother
```

## Index

```@index
Pages = ["kalman.md"]
Depth = 2
```