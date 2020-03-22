@testset "block jackknife" begin

    # Load data
    Y = read_test_input("./input/subsampling/data");

    # Settings
    subsample = 0.2;

    # Copy Y and subsample
    Y_copy = deepcopy(Y);
    subsample_copy = deepcopy(subsample);

    # Run `block jackknife` function
    output = block_jackknife(Y, subsample);

    # Load benchmark output
    benchmark = read_test_input("./input/subsampling/block_jackknife/output_chunk1");
    benchmark_size = length(readdir("./input/subsampling/block_jackknife"));

    for i=2:size(output,3)
        benchmark = cat(dims=3, benchmark, read_test_input("./input/subsampling/block_jackknife/output_chunk$(i)"));
    end

    # Run tests
    @test Y_copy == Y;
    @test subsample_copy == subsample;
    @test size(output,3) == benchmark_size;
    @test sum(output .=== benchmark) == prod(size(output));
end

@testset "artificial jackknife" begin

    # Load data
    Y = read_test_input("./input/subsampling/data");

    # Settings
    subsample = 0.2;
    max_samples = 100;

    # Copy Y and subsample
    Y_copy = deepcopy(Y);
    subsample_copy = deepcopy(subsample);
    max_samples_copy = deepcopy(max_samples);

    # Run `artificial jackknife` function with fixed random seed
    Random.seed!(1);
    output = artificial_jackknife(Y, subsample, max_samples);

    # Load benchmark output
    benchmark = read_test_input("./input/subsampling/artificial_jackknife/output_chunk1");
    benchmark_size = length(readdir("./input/subsampling/artificial_jackknife"));

    for i=2:size(output,3)
        benchmark = cat(dims=3, benchmark, read_test_input("./input/subsampling/artificial_jackknife/output_chunk$(i)"));
    end

    # Run tests
    @test Y_copy == Y;
    @test subsample_copy == subsample;
    @test max_samples_copy == max_samples;
    @test size(output,3) == benchmark_size;
    @test sum(output .=== benchmark) == prod(size(output));
end
