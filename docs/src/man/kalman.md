# Kalman filter and smoother

## Index

```@index
Pages = ["kalman.md"]
Depth = 2
```

## Types

```@docs
KalmanSettings
KalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, Q::SymMatrix; kwargs...)
KalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix; kwargs...)
KalmanSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatVector, P0::SymMatrix; kwargs...)
```

```@docs
OnlineKalmanStatus
```

```@docs
DynamicKalmanStatus
```

```@docs
SizedKalmanStatus
```

## Functions

### Kalman filter

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