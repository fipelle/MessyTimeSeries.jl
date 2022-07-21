# Annex functions

## Functions

This package also includes the following general-purpose functions useful for time series analysis.

## Building blocks for time series algorithms

### Convenient mathematical and statistical operations
```@docs
no_combinations
rand_without_replacement
soft_thresholding
solve_discrete_lyapunov
```

### Convergence check
```@docs
check_bounds
isconverged
```

### Parameter transformations
```@docs
get_bounded_log
get_unbounded_log
get_bounded_logit
get_unbounded_logit
```

## Sample statistics for incomplete data
```@docs
mean_skipmissing
std_skipmissing
sum_skipmissing
```

## Time-series operations

### Foundations
```@docs
companion_form
lag
diff2
diff_or_diff2
```

### Interpolation and moving averages
```@docs
centred_moving_average
forward_backwards_rw_interpolation
interpolate_series
```

### Standardisation
```@docs
demean
standardise
```

## Index

```@index
Pages = ["methods.md"]
Depth = 2
```