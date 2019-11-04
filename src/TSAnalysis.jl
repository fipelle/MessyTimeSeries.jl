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

    # Export
    export JVector, JArray, ImmutableKalmanSettings, MutableKalmanSettings, KalmanStatus, ARIMASettings;
    export mean_skipmissing, std_skipmissing, is_vector_in_matrix, demean, lag, companion_form, ext_companion_form;
    export kfilter!, kforecast, ksmoother;
    export arima;
end
