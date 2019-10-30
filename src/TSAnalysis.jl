__precompile__()

module TSAnalysis

    # Libraries
    using Distributed;
    using Dates, Logging;
    using LinearAlgebra, Distributions, Statistics;

    # Custom dependencies
    const local_path = dirname(@__FILE__);
    include("$local_path/types.jl");
    include("$local_path/methods.jl");
    include("$local_path/kalman.jl");

    # Export
    export JVector, JArray, KalmanSettings, ImmutableKalmanSettings, MutableKalmanSettings, KalmanStatus;
    export mean_skipmissing, std_skipmissing, is_vector_in_matrix, demean, lag, companion_form, ext_companion_form;
    export kfilter!, kforecast, ksmoother;
end
