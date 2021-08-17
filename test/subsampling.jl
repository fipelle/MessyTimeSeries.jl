"""
    subsampling_test(folder_name::String, subsampling_method::Function, args...)

Run a series of tests to check whether the subsampling functions in subsampling.jl work.
"""
function subsampling_test(folder_name::String, subsampling_method::Function, args...)

    # Load data
    Y = read_test_input("./input/subsampling/data");

    # Copy Y and subsample
    Y_copy = deepcopy(Y);
    args_copy = deepcopy(args);

    # Run `subsampling_method` with default random seed
    output = subsampling_method(Y, args...);

    # Load benchmark output
    benchmark = read_test_input("./input/subsampling/$(folder_name)/output_chunk1");
    benchmark_size = length(readdir("./input/subsampling/$(folder_name)"));

    for i=2:size(output,3)
        benchmark = cat(dims=3, benchmark, read_test_input("./input/subsampling/$(folder_name)/output_chunk$(i)"));
    end

    # Run tests
    @test Y_copy == Y;
    @test args_copy == args;
    @test size(output,3) == benchmark_size;
    @test sum(output .=== benchmark) == prod(size(output));
end

@testset "block jackknife" begin
    subsampling_test("block_jackknife", block_jackknife, 0.2)
end

@testset "artificial jackknife" begin
    subsampling_test("artificial_jackknife", artificial_jackknife, 0.2, 100)
end

@testset "moving block bootstrap" begin
    subsampling_test("moving_block_bootstrap", moving_block_bootstrap, 0.2, 100)
end

@testset "stationary block bootstrap" begin
    subsampling_test("stationary_block_bootstrap", stationary_block_bootstrap, 0.2, 100)
end
