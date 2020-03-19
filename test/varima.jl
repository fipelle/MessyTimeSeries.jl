"""
    varima_test(Y::Array{Float64,2}, d::Int64, p::Int64, q::Int64, benchmark_data::Tuple)

Run a series of tests to check whether the varima functions in uc_models.jl work.
"""
function varima_test(Y::Array{Float64,2}, d::Int64, p::Int64, q::Int64, benchmark_data::Tuple)

    # Benchmark data
    benchmark_X0, benchmark_P0, benchmark_B, benchmark_R, benchmark_C, benchmark_V, benchmark_fc = benchmark_data;

    # Tests on VARIMASettings
    varima_settings = VARIMASettings(Y, d, p, q);
    @test varima_settings.d == d;
    @test varima_settings.p == p;
    @test varima_settings.q == q;
    @test varima_settings.n == size(Y,1);
    @test varima_settings.nr == size(Y,1)*max(p, q+1);
    @test varima_settings.np == size(Y,1)*p;
    @test varima_settings.nq == size(Y,1)*q;
    @test varima_settings.nnp == size(Y,1)^2*p;
    @test varima_settings.nnq == size(Y,1)^2*q;

    # Estimate parameters
    varima_out = varima(varima_settings, NelderMead(), Optim.Options(iterations=10000, f_tol=1e-2, x_tol=1e-2, g_tol=1e-2, show_trace=true, show_every=500));

    # Test on the parameters
    @test round.(varima_out.B, digits=10) == benchmark_B;
    @test round.(varima_out.R, digits=10) == benchmark_R;
    @test round.(varima_out.C, digits=10) == benchmark_C;
    @test round.(varima_out.V, digits=10) == benchmark_V;
    @test round.(varima_out.X0, digits=10) == benchmark_X0;
    @test round.(varima_out.P0, digits=10) == benchmark_P0;

    # 12-step ahead forecast
    fc = forecast(varima_out, 12, varima_settings);
    @test round.(fc, digits=10) == benchmark_fc;
end

@testset "arma" begin

    # Load data
    Y = permutedims(read_test_input("./input/arma/data"));

    # Settings for ARMA(1,1)
    d = 0;
    p = 1;
    q = 1;

    # Correct estimates: initial conditions
    benchmark_X0 = read_test_input("./input/arma/benchmark_X0");
    benchmark_P0 = read_test_input("./input/arma/benchmark_P0");

    # Correct estimates: observation equation
    benchmark_B = read_test_input("./input/arma/benchmark_B");
    benchmark_R = read_test_input("./input/arma/benchmark_R");

    # Correct estimates: transition equation
    benchmark_C = read_test_input("./input/arma/benchmark_C");
    benchmark_V = read_test_input("./input/arma/benchmark_V");

    # Correct estimates: forecast
    benchmark_fc = read_test_input("./input/arma/benchmark_fc");

    # Benchmark data
    benchmark_data = (benchmark_X0, benchmark_P0, benchmark_B, benchmark_R, benchmark_C, benchmark_V, benchmark_fc);

    # Run tests
    varima_test(Y, d, p, q, benchmark_data);
end

@testset "arima" begin

    # Load data
    Y = permutedims(read_test_input("./input/arima/data"));

    # Settings for arima(1,1)
    d = 1;
    p = 1;
    q = 1;

    # Correct estimates: initial conditions
    benchmark_X0 = read_test_input("./input/arima/benchmark_X0");
    benchmark_P0 = read_test_input("./input/arima/benchmark_P0");

    # Correct estimates: observation equation
    benchmark_B = read_test_input("./input/arima/benchmark_B");
    benchmark_R = read_test_input("./input/arima/benchmark_R");

    # Correct estimates: transition equation
    benchmark_C = read_test_input("./input/arima/benchmark_C");
    benchmark_V = read_test_input("./input/arima/benchmark_V");

    # Correct estimates: forecast
    benchmark_fc = read_test_input("./input/arima/benchmark_fc");

    # Benchmark data
    benchmark_data = (benchmark_X0, benchmark_P0, benchmark_B, benchmark_R, benchmark_C, benchmark_V, benchmark_fc);

    # Run tests
    varima_test(Y, d, p, q, benchmark_data);
end

@testset "varma" begin

    # Load data
    Y = read_test_input("./input/varma/data");

    # Settings for ARMA(1,1)
    d = 0;
    p = 1;
    q = 1;

    # Correct estimates: initial conditions
    benchmark_X0 = read_test_input("./input/varma/benchmark_X0");
    benchmark_P0 = read_test_input("./input/varma/benchmark_P0");

    # Correct estimates: observation equation
    benchmark_B = read_test_input("./input/varma/benchmark_B");
    benchmark_R = read_test_input("./input/varma/benchmark_R");

    # Correct estimates: transition equation
    benchmark_C = read_test_input("./input/varma/benchmark_C");
    benchmark_V = read_test_input("./input/varma/benchmark_V");

    # Correct estimates: forecast
    benchmark_fc = read_test_input("./input/varma/benchmark_fc");

    # Benchmark data
    benchmark_data = (benchmark_X0, benchmark_P0, benchmark_B, benchmark_R, benchmark_C, benchmark_V, benchmark_fc);

    # Run tests
    varima_test(Y, d, p, q, benchmark_data);
end
