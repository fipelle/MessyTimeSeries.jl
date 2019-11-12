__precompile__()

module TSAnalysis

    # Libraries
    using Dates, Distributed, Logging;
    using LinearAlgebra, Distributions, Optim, Statistics;

    # Custom dependencies
    const local_path = dirname(@__FILE__);
    include("$local_path/types.jl");
    include("$local_path/methods.jl");
    include("$local_path/kalman.jl");
    include("$local_path/uc_models.jl");

    # Export types
    export JVector, JArray, FloatVector, FloatArray, SymMatrix, DiagMatrix,
           ImmutableKalmanSettings, MutableKalmanSettings, KalmanStatus, ARIMASettings;

    # Export methods
    export check_bounds, isnothing, error_info, verb_message, interpolate, soft_thresholding, isconverged,
            mean_skipmissing, std_skipmissing, is_vector_in_matrix, demean, lag, companion_form;

    # Export functions
    export kfilter!, kforecast, ksmoother, arima, forecast;
end
