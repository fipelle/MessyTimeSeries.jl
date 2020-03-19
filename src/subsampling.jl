"""
    block_jackknife(Y::JArray{Float64,2}, subsample::Float64)

Generate block jackknife samples as in Kunsch (1989).

This technique subsamples a time series dataset by removing, in turn, all the blocks of consecutive observations
with a given size.

# Arguments
- `Y`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `subsample`: block size as a percentage of the original sample size. It is bounded between 0 and 1.

# References
Kunsch (1989) and Pellegrino (2020)
"""
function block_jackknife(Y::JArray{Float64,2}, subsample::Float64)

    # Check inputs
    check_bounds(subsample, 0, 1);

    # Dimensions
    n, T = size(Y);

    # Block size
    block_size = Int64(ceil(subsample*T));
    if block_size == 0
        error("subsample is too small!");
    end

    # Number of block jackknifes samples - as in Kunsch (1989)
    samples = T-block_size+1;

    # Initialise jackknife_data
    jackknife_data = JArray{Float64,3}(undef, n, T, samples);

    # Loop over j=1, ..., samples
    for j=1:samples

        # Index of missings
        ind_j = collect(j:j+block_size-1);

        # Block jackknife data
        jackknife_data[:, :, j] = Y;
        jackknife_data[:, ind_j, j] .= missing;
    end

    # Return jackknife_data
    return jackknife_data;
end

"""
    optimal_d(n::Int64, T::Int64)

Select the optimal value for d.

# Arguments
- `n`: Number of series
- `T`: Number of observations
"""
function optimal_d(n::Int64, T::Int64)
    objfun_array = zeros(n*T);

    for d=1:fld(n*T,2)
        objfun_array[d] = objfun_optimal_d(n, T, d);
    end

    return argmax(objfun_array);
end

"""
    objfun_optimal_d(n::Int64, T::Int64, d::Int64)

Objective function to select the optimal value for d.

# Arguments
- `n`: Number of series
- `T`: Number of observations
- `d`: Candidate d
"""
function objfun_optimal_d(n::Int64, T::Int64, d::Int64)
    fun = no_combinations(n*T, d) - (d>=n).*no_combinations(n*T-n, d-n)*T;

    for i=2:fld(d, n)
        fun -= (-1^(i-1))*no_combinations(T, i)*no_combinations(n*T-i*n, d-i*n);
    end

    return fun;
end

"""
    artificial_jackknife(Y::JArray{Float64,2}, subsample::Float64, max_samples::Int64)

Generate artificial jackknife samples as in Pellegrino (2020).

The artificial delete-d jackknife is an extension of the delete-d jackknife for dependent data problems.
This technique replaces the actual data removal step with a fictitious deletion, which consists of
imposing `d`-dimensional (artificial) patterns of missing observations to the data. This approach
does not alter the data order nor destroy the correlation structure.

# Arguments
- `Y`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `subsample`: `d` as a percentage of the original sample size. It is bounded between 0 and 1.
- `max_samples`: if `C(n*T,d)` is large, artificial_jackknife would generate `max_samples` jackknife samples.

# References
Pellegrino (2020)
"""
function artificial_jackknife(Y::JArray{Float64,2}, subsample::Float64, max_samples::Int64)

    # Dimensions
    n, T = size(Y);
    nT = n*T;

    # Check inputs
    check_bounds(subsample, 0, 1);

    # Set d using subsample
    d = Int64(ceil(subsample*nT));

    # Error management
    if d == 0
        error("subsample is too small!");
    end

    # Warning
    if subsample > 0.5
        @warn "this algorithm might be unstable for `subsample` larger than 0.5!";
    end

    # Get vec(Y)
    vec_Y = convert(JArray{Float64}, Y[:]);

    # Initialise loop (controls)
    C_nT_d = no_combinations(nT, d);
    samples = convert(Int64, min(C_nT_d, max_samples));
    zeros_vec = zeros(d);

    # Initialise loop (output)
    ind_missings = Array{Int64}(zeros(d, samples));
    jackknife_data = JArray{Float64,3}(undef, n, T, samples);

    # Loop over j=1, ..., samples
    for j=1:samples

        # First draw
        if j == 1
            if samples == C_nT_d
                ind_missings[:,j] = rand_without_replacement(nT, d);
            else
                ind_missings[:,j] = rand_without_replacement(n, T, d);
            end

        # Iterates until ind_missings[:,j] is neither a vector of zeros, nor already included in ind_missings
        else
            while ind_missings[:,j] == zeros_vec || is_vector_in_matrix(ind_missings[:,j], ind_missings[:, 1:j-1])
                if samples == C_nT_d
                    ind_missings[:,j] = rand_without_replacement(nT, d);
                else
                    ind_missings[:,j] = rand_without_replacement(n, T, d);
                end
            end
        end

        # Add (artificial) missing observations
        vec_Y_j = copy(vec_Y);
        vec_Y_j[ind_missings[:,j]] .= missing;

        # Store data
        jackknife_data[:, :, j] = reshape(vec_Y_j, n, T);
    end

    # Return jackknife_data
    return jackknife_data;
end
