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

Note: the eigenvalues are ordered in a convenient way for `eigvals_to_coeff`.
"""
square_vandermonde_matrix(λ::FloatVector) = λ'.^collect(length(λ)-1:-1:0);

#=
--------------------------------------------------------------------------------------------------------------------------------
Parameter transformations
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    eigvals_to_coeff(λ::FloatVector)

Compute the companion form corresponding to a vector of eigenvalues and returns the first row.

Note: this approach returns a non-zero vector of coefficients if the elements of λ are unique (distinct eigenvalues).

# Arguments
- `λ`: Vector of eigenvalues
- `k`: Number of coefficients to retrieve from the companion form matrix (default: 1)
"""
function eigvals_to_coeff(λ::FloatVector; k::Int64=1)

    try
        # Vandermonde matrix
        V = square_vandermonde_matrix(λ);

        # Return coefficients
        if k == 1
            return permutedims(V[1,:])*Diagonal(λ)*inv(V);
        else
            return V[1:k,:]*Diagonal(λ)*inv(V); # TODO: this line has not been tested properly, since there is not yet support for multivariate models.
        end

    catch err
        if isa(err, SingularException)
            return zeros(length(λ));
        else
            rethrow(err);
        end
    end
end

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
    interpolate(X::JArray{Float64}, n::Int64, T::Int64)

Interpolate each series in `X`, in turn, by replacing missing observations with the sample average of the non-missing values.

# Arguments
- `X`: observed measurements (`nxT`)
- `n` and `T` are the number of series and observations

    interpolate(X::Array{Float64}, n::Int64, T::Int64)

# Arguments
- `X`: observed measurements (`nxT`)
- `n` and `T` are the number of series and observations
"""
function interpolate(X::JArray{Float64}, n::Int64, T::Int64)
    data = copy(X);
    for i=1:n
        data[i, ismissing.(X[i, :])] .= mean_skipmissing(X[i, :]);
    end
    data = convert(Array{Float64}, data);
    return data;
end

interpolate(X::FloatArray, n::Int64, T::Int64) = X;

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
    companion_form(θ::FloatVector)

Construct the companion form parameters of θ.
"""
function companion_form(θ::FloatVector)

    # Number of parameters in θ
    k = length(θ);

    # Return companion form
    return [permutedims(θ); Matrix(I, k-1, k-1) zeros(k-1)];
end
