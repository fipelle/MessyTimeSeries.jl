using LinearAlgebra, Test;#, TSAnalysis;
include("../src/TSAnalysis.jl")
using Main.TSAnalysis;

"""
    read_test_input(filepath::String)

Read input data necessary to run the test for the Kalman routines. It does not use external dependencies to read input files.
"""
function read_test_input(filepath::String)

    # Load CSV into Array{SubString{String},1}
    data_str = split(read(open("$filepath.txt"), String), "\n");
    deleteat!(data_str, findall(x->x=="", data_str));

    # Return output
    data = eval(Meta.parse(data_str[1]));
    return data;
end
