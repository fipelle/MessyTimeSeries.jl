__precompile__()

module MessyTimeSeries

    # Libraries
    using Dates, Distributed, Logging;
    using LinearAlgebra, Distributions, StableRNGs, Statistics;
    using Infiltrator;
    
    # Custom dependencies
    local_path = dirname(@__FILE__);
    include("$(local_path)/types.jl");
    include("$(local_path)/methods.jl");
    include("$(local_path)/kalman.jl");
    include("$(local_path)/subsampling.jl");

    # Export types
    export IntVector, IntMatrix, IntArray, FloatVector, FloatMatrix, FloatArray, JVector, JMatrix, JArray, DiagMatrix, SymMatrix,
           KalmanSettings, KalmanStatus, OnlineKalmanStatus, DynamicKalmanStatus, SizedKalmanStatus, ARIMASettings, VARIMASettings;

    # Export methods
    export check_bounds, nan_to_missing!, error_info, verb_message, interpolate_series, forward_backwards_rw_interpolation, centred_moving_average,
            soft_thresholding, solve_discrete_lyapunov, isconverged, trimmed_mean, sum_skipmissing, mean_skipmissing, std_skipmissing, is_vector_in_matrix, 
            demean, standardise, diff2, diff_or_diff2, lag, companion_form, no_combinations, rand_without_replacement, get_bounded_log, get_unbounded_log, get_bounded_logit, get_unbounded_logit;

    # Export functions
    export kfilter!, kfilter_full_sample, kfilter_full_sample!, kforecast, ksmoother, fmin_uc_models, arima, varima, forecast,
           block_jackknife, optimal_d, artificial_jackknife, moving_block_bootstrap, stationary_block_bootstrap;
end
