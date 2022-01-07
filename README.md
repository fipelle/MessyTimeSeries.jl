# MessyTimeSeries.jl
A Julia implementation of basic tools for time series analysis compatible with incomplete data.

| **Documentation**                                                              |
|:-------------------------------------------------------------------------------:
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url]      |

Advanced estimation and validation algorithms are included in [```MessyTimeSeriesOptim```](https://github.com/fipelle/MessyTimeSeriesOptim.jl).

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add MessyTimeSeries
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("MessyTimeSeries")
```


[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://fipelle.github.io/MessyTimeSeries.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://fipelle.github.io/MessyTimeSeries.jl/stable
