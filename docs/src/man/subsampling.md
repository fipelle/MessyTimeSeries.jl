# Subsampling

## Functions

This package provides a series of methods for partitioning time-series data based on jackkniving and bootstrapping.

### Jackknife

The first group of algorithms includes the following generalisation of the jackknife for dependent data:
- the block jackknife (Kunsch, 1989);
- the artificial delete-``d`` jackknife (Pellegrino, 2022).

```@docs
block_jackknife
artificial_jackknife
optimal_d
```

### Bootstrap

The second group includes the following bootstrap versions compatible with time series:
- the moving block bootstrap (Kunsch, 1989; Liu and Singh, 1992);
- the stationary block bootstrap (Politis and Romano, 1994).

```@docs
moving_block_bootstrap
stationary_block_bootstrap
```

## Index

```@index
Pages = ["subsampling.md"]
Depth = 2
```