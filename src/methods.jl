#=
--------------------------------------------------------------------------------------------------------------------------------
Base and generic math
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    isnothing(::Any)

True if the argument is ```nothing``` (false otherwise).
"""
isnothing(::Any) = false;
isnothing(::Nothing) = true;

"""
    verb_message(verb::Bool, message::String)

@info `message` if `verb` is true.
"""
verb_message(verb::Bool, message::String) = verb ? @info(message) : nothing;

"""
    check_bounds(X::Number, LB::Number, UB::Number)

Check whether `X` is larger or equal than `LB` and lower or equal than `UB`

    check_bounds(X::Number, LB::Number)

Check whether `X` is larger or equal than `LB`
"""
check_bounds(X::Number, LB::Number, UB::Number) = X < LB || X > UB ? throw(DomainError) : nothing
check_bounds(X::Number, LB::Number) = X < LB ? throw(DomainError) : nothing

"""
    error_info(err::Exception)
    error_info(err::RemoteException)

Return error main information
"""
error_info(err::Exception) = (err, err.msg, stacktrace(catch_backtrace()));
error_info(err::RemoteException) = (err.captured.ex, err.captured.ex.msg, [err.captured.processed_bt[i][1] for i=1:length(err.captured.processed_bt)]);

"""
    mean_skipmissing(X::AbstractArray{Float64,1})
    mean_skipmissing(X::AbstractArray{Union{Missing, Float64},1})

Compute the mean of the observed values in `X`.

    mean_skipmissing(X::AbstractArray{Float64})
    mean_skipmissing(X::AbstractArray{Union{Missing, Float64}})

Compute the mean of the observed values in `X` column wise.

# Examples
```jldoctest
julia> mean_skipmissing([1.0; missing; 3.0])
2.0

julia> mean_skipmissing([1.0 2.0; missing 3.0; 3.0 5.0])
3-element Array{Float64,1}:
 1.5
 3.0
 4.0
```
"""
mean_skipmissing(X::AbstractArray{Float64,1}) = mean(X);
mean_skipmissing(X::AbstractArray{Float64}) = mean(X, dims=2);
mean_skipmissing(X::AbstractArray{Union{Missing, Float64},1}) = mean(skipmissing(X));
mean_skipmissing(X::AbstractArray{Union{Missing, Float64}}) = vcat([mean_skipmissing(X[i,:]) for i=1:size(X,1)]...);

"""
    std_skipmissing(X::AbstractArray{Float64,1})
    std_skipmissing(X::AbstractArray{Union{Missing, Float64},1})

Compute the standard deviation of the observed values in `X`.

    std_skipmissing(X::AbstractArray{Float64})
    std_skipmissing(X::AbstractArray{Union{Missing, Float64}})

Compute the standard deviation of the observed values in `X` column wise.

# Examples
```jldoctest
julia> std_skipmissing([1.0; missing; 3.0])
1.4142135623730951

julia> std_skipmissing([1.0 2.0; missing 3.0; 3.0 5.0])
3-element Array{Float64,1}:
   0.7071067811865476
 NaN
   1.4142135623730951
```
"""
std_skipmissing(X::AbstractArray{Float64,1}) = std(X);
std_skipmissing(X::AbstractArray{Float64}) = std(X, dims=2);
std_skipmissing(X::AbstractArray{Union{Missing, Float64},1}) = std(skipmissing(X));
std_skipmissing(X::AbstractArray{Union{Missing, Float64}}) = vcat([std_skipmissing(X[i,:]) for i=1:size(X,1)]...);

"""
    is_vector_in_matrix(vect::AbstractVector, matr::AbstractMatrix)

Check whether the vector `vect` is included in the matrix `matr`.

# Examples
julia> is_vector_in_matrix([1;2], [1 2; 2 3])
true
"""
is_vector_in_matrix(vect::AbstractVector, matr::AbstractMatrix) = sum(sum(vect .== matr, dims=1) .== length(vect)) > 0;

"""
    isconverged(new::Float64, old::Float64, tol::Float64, ε::Float64, increasing::Bool)

Check whether `new` is close enough to `old`.

# Arguments
- `new`: new objective or loss
- `old`: old objective or loss
- `tol`: tolerance
- `ε`: small Float64
- `increasing`: true if `new` increases, at each iteration, with respect to `old`
"""
isconverged(new::Float64, old::Float64, tol::Float64, ε::Float64, increasing::Bool) = increasing ? (new-old)./(abs(old)+ε) <= tol : -(new-old)./(abs(old)+ε) <= tol;

"""
    soft_thresholding(z::Float64, ζ::Float64)

Soft thresholding operator.
"""
soft_thresholding(z::Float64, ζ::Float64) = sign(z)*max(abs(z)-ζ, 0);

"""
    square_vandermonde_matrix(λ::FloatVector)

Construct square vandermonde matrix on the basis of a vector of eigenvalues λ.
"""
square_vandermonde_matrix(λ::FloatVector) = λ'.^collect(length(λ)-1:-1:0);

#=
--------------------------------------------------------------------------------------------------------------------------------
Parameter transformations
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    get_bounded_log(Θ_unbound::Float64, MIN::Float64)

Compute parameters with bounded support using a generalised log transformation.
"""
get_bounded_log(Θ_unbound::Float64, MIN::Float64) = exp(Θ_unbound) + MIN;

"""
    get_unbounded_log(Θ_bound::Float64, MIN::Float64)

Compute parameters with unbounded support using a generalised log transformation.
"""
get_unbounded_log(Θ_bound::Float64, MIN::Float64) = log(Θ_bound - MIN);

"""
    get_bounded_logit(Θ_unbound::Float64, MIN::Float64, MAX::Float64)

Compute parameters with bounded support using a generalised logit transformation.
"""
get_bounded_logit(Θ_unbound::Float64, MIN::Float64, MAX::Float64) = (MIN + (MAX * exp(Θ_unbound))) / (1 + exp(Θ_unbound));

"""
    get_unbounded_logit(Θ_bound::Float64, MIN::Float64, MAX::Float64)

Compute parameters with unbounded support using a generalised logit transformation.
"""
get_unbounded_logit(Θ_bound::Float64, MIN::Float64, MAX::Float64) = log((Θ_bound - MIN) / (MAX - Θ_bound));

#=
--------------------------------------------------------------------------------------------------------------------------------
Time series
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    demean(X::FloatVector)
    demean(X::JVector)

Demean data.

    demean(X::FloatMatrix)
    demean(X::JArray)

Demean data.

# Examples
```jldoctest
julia> demean([1.0; 1.5; 2.0; 2.5; 3.0])
5-element Array{Float64,1}:
 -1.0
 -0.5
  0.0
  0.5
  1.0

julia> demean([1.0 3.5 1.5 4.0 2.0; 4.5 2.5 5.0 3.0 5.5])
2×5 Array{Float64,2}:
 -1.4   1.1  -0.9   1.6  -0.4
  0.4  -1.6   0.9  -1.1   1.4
```
"""
demean(X::FloatVector) = X .- mean(X);
demean(X::FloatMatrix) = X .- mean(X,dims=2);
demean(X::JVector) = X .- mean_skipmissing(X);
demean(X::JArray) = X .- mean_skipmissing(X);

"""
    standardise(X::FloatVector)
    standardise(X::JVector)

Demean data.

    standardise(X::FloatMatrix)
    standardise(X::JArray)

Demean data.

# Examples
```jldoctest
julia> standardise([1.0; 1.5; 2.0; 2.5; 3.0])
5-element Array{Float64,1}:
 -1.2649110640673518
 -0.6324555320336759
  0.0
  0.6324555320336759
  1.2649110640673518

julia> standardise([1.0 3.5 1.5 4.0 2.0; 4.5 2.5 5.0 3.0 5.5])
2×5 Array{Float64,2}:
 -1.08173    0.849934  -0.695401   1.23627   -0.309067
  0.309067  -1.23627    0.695401  -0.849934   1.08173
```
"""
standardise(X::FloatVector) = (X .- mean(X))./std(X);
standardise(X::FloatMatrix) = (X .- mean(X,dims=2))./std(X,dims=2);
standardise(X::JVector) = (X .- mean_skipmissing(X))./std_skipmissing(X);
standardise(X::JArray) = (X .- mean_skipmissing(X))./std_skipmissing(X);

"""
    interpolate(X::JArray{Float64,2}, n::Int64, T::Int64)

Interpolate each series in `X`, in turn, by replacing missing observations with the sample average of the non-missing values.

# Arguments
- `X`: observed measurements (`nxT`)
- `n` and `T` are the number of series and observations

    interpolate(X::FloatMatrix, n::Int64, T::Int64)

Return `X`.

# Arguments
- `X`: observed measurements (`nxT`)
- `n` and `T` are the number of series and observations
"""
function interpolate(X::JArray{Float64,2}, n::Int64, T::Int64)
    data = copy(X);
    for i=1:n
        data[i, ismissing.(X[i, :])] .= mean_skipmissing(X[i, :]);
    end
    data = data |> FloatArray;
    return data;
end

interpolate(X::FloatMatrix, n::Int64, T::Int64) = X;

"""
    forward_backwards_rw_interpolation(X::JArray{Float64,2}, n::Int64, T::Int64)

Interpolate each non-stationary series in `X`, in turn, using a random walk logic both forward and backwards in time.

# Arguments
- `X`: observed measurements (`nxT`)
- `n` and `T` are the number of series and observations

    forward_backwards_rw_interpolation(X::FloatMatrix, n::Int64, T::Int64)

Return `X`.
"""
function forward_backwards_rw_interpolation(X::JArray{Float64,2}, n::Int64, T::Int64)

    data = copy(X);

    # Loop over n and T
    for i=1:n

        # Forward interpolation
        drift_forward = mean_skipmissing(diff(data, dims=2))[:];
        for t=2:T
            if ismissing(data[i,t]) && ~ismissing(data[i,t-1])
                data[i,t] = data[i,t-1] + drift_forward[i];
            end
        end

        # Backwards interpolation
        drift_backwards = mean_skipmissing(diff(data, dims=2))[:];
        for t=T:-1:1
            if ismissing(data[i,t]) && ~ismissing(data[i,t+1])
                data[i,t] = data[i,t+1] - drift_backwards[i];
            end
        end
    end

    # Return interpolated data
    data = data |> FloatMatrix;
    return data;
end

forward_backwards_rw_interpolation(X::FloatMatrix, n::Int64, T::Int64) = X;

"""
    centred_moving_average(X::Union{JArray{Float64}, FloatArray}, n::Int64, T::Int64, window::Int64)

Compute the centred moving average of `X`.

# Arguments
- `X`: observed measurements (`nxT`)
- `n` and `T` are the number of series and observations
- `window` is the total number of observations (lagging, current and leading) included in the average
"""
function centred_moving_average(X::Union{JArray{Float64}, FloatArray}, n::Int64, T::Int64, window::Int64)

    # Checks on the moving average window
    check_bounds(T, 3);
    check_bounds(window, 3, T);

    # window_adj must be odd
    window_adj = copy(window);
    if mod(window-1,2) != 0
        window_adj = window-1;
    end
    one_sided_window = (window_adj-1)/2 |> Int64;

    # Compute centred moving average
    data = missing .* zeros(n,T) |> JArray{Float64};
    for t=one_sided_window+1:T-one_sided_window
        for i=1:n
            # Convenient view
            X_window = @view X[i, t-one_sided_window:t+one_sided_window];
            if sum(ismissing.(X_window)) < window_adj
                data[i,t] = mean_skipmissing(X[i, t-one_sided_window:t+one_sided_window]);
            end
        end
    end

    return data;
end

"""
    lag(X::FloatArray, p::Int64)

Construct the data required to run a standard vector autoregression.

# Arguments
- `X`: observed measurements (`nxT`), where `n` and `T` are the number of series and observations.
- `p`: number of lags in the vector autoregression

# Output
- `X_{t}`
- `X_{t-1}`
"""
function lag(X::FloatArray, p::Int64)

    # VAR(p) data
    X_t = X[:, 1+p:end];
    X_lagged = vcat([X[:, p-j+1:end-j] for j=1:p]...);

    # Return output
    return X_t, X_lagged;
end

"""
    companion_form(Ψ::FloatMatrix; extended::Bool=false)

Construct the companion form matrix from the generic coefficients Ψ. 

If extended is true, it increases the typical dimension of the companion matrix by n rows.
"""
function companion_form(Ψ::FloatMatrix; extended::Bool=false)

    # Dimensions
    n = size(Ψ,1);
    p = Int64(size(Ψ,2)/n);

    # Number of rows of the block of identities in the companion form matrix
    standard_rows = n*(p-1);
    extra_rows = n*(extended == true);
    total_rows = standard_rows + extra_rows;

    # Return companion matrix
    companion = [Ψ zeros(n, extra_rows); Matrix(I, total_rows, total_rows) zeros(total_rows, n)];
    return companion;
end

#=
-------------------------------------------------------------------------------------------------------------------------------
Combinatorics and probability
-------------------------------------------------------------------------------------------------------------------------------
=#

"""
    no_combinations(n::Int64, k::Int64)

Compute the binomial coefficient of `n` observations and `k` groups, for big integers.

# Examples
```jldoctest
julia> no_combinations(1000000,100000)
7.333191945934207610471288280331309569215030711272858517142085449641265002716664e+141178
```
"""
no_combinations(n::Int64, k::Int64) = factorial(big(n))/(factorial(big(k))*factorial(big(n-k)));

"""
    rand_without_replacement(nT::Int64, d::Int64)

Draw `length(P)-d` elements from the positional vector `P` without replacement.
`P` is permanently changed in the process.

rand_without_replacement(n::Int64, T::Int64, d::Int64)

Draw `length(P)-d` elements from the positional vector `P` without replacement.
In the sampling process, no more than n-1 elements are removed for each point in time.
`P` is permanently changed in the process.

# Examples
```jldoctest
julia> rand_without_replacement(20, 5)
15-element Array{Int64,1}:
  1
  2
  3
  5
  7
  8
 10
 11
 13
 14
 16
 17
 18
 19
 20
```
"""
function rand_without_replacement(nT::Int64, d::Int64)

    # Positional vector
    P = collect(1:nT);

    # Draw without replacement d times
    for i=1:d
        deleteat!(P, findall(P.==rand(P)));
    end

    # Return output
    return setdiff(1:nT, P);
end

function rand_without_replacement(n::Int64, T::Int64, d::Int64)

    # Positional vector
    P = collect(1:n*T);

    # Full set of coordinates
    coord = [repeat(1:n, T) kron(1:T, convert(Array{Int64}, ones(n)))];

    # Counter
    coord_counter = convert(Array{Int64}, zeros(T));

    # Loop over d
    for i=1:d

        while true

            # New candidate draw
            draw = rand(P);
            coord_draw = @view coord[draw, :];

            # Accept the draw if all observations are not missing for time t = coord[draw, :][2]
            if coord_counter[coord_draw[2]] < n-1
                coord_counter[coord_draw[2]] += 1;

                # Draw without replacement
                deleteat!(P, findall(P.==draw));
                break;
            end
        end
    end

    # Return output
    return setdiff(1:n*T, P);
end
