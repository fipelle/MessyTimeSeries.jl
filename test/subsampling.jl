@testset "block jackknife" begin

    # Load data
    Y = read_test_input("./input/subsampling/data");

    # Settings
    subsample = 0.2;

    # Copy Y and subsample
    Y_copy = deepcopy(Y);
    subsample_copy = deepcopy(subsample);

    # Run `block jackknife` function
    block_jackknife(Y, subsample);

    # Load benchmark output
    Y = read_test_input("./input/subsampling/data");

    # Run tests

end
