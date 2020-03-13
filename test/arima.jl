"""
    arima_test(arima_settings::ARIMASettings, benchmark_data::Tuple)

Run a series of tests to check whether the arima functions in uc_models.jl work.
"""
function arima_test(Y::Array{Float64,2}, d::Int64, p::Int64, q::Int64, benchmark_data::Tuple)

    # Benchmark data
    benchmark_X0, benchmark_P0, benchmark_B, benchmark_R, benchmark_C, benchmark_V, benchmark_fc = benchmark_data;

    # Tests on ARIMASettings
    arima_settings = ARIMASettings(Y, d, p, q);
    @test arima_settings.d == d;
    @test arima_settings.p == p;
    @test arima_settings.q == q;
    @test arima_settings.r == max(p, q+1);

    # Estimate parameters
    arima_out = varima(arima_settings, 1/(arima_settings.np+arima_settings.nq), NelderMead(),
                       Optim.Options(iterations=10000, f_tol=1e-4, x_tol=1e-4, show_trace=true, show_every=500));

    # Test on the parameters
    @test round.(arima_out.B, digits=10) == benchmark_B;
    @test round.(arima_out.R, digits=10) == benchmark_R;
    @test round.(arima_out.C, digits=10) == benchmark_C;
    @test round.(arima_out.V, digits=10) == benchmark_V;
    @test round.(arima_out.X0, digits=10) == benchmark_X0;
    @test round.(arima_out.P0, digits=10) == benchmark_P0;

    # 12-step ahead forecast
    fc = forecast(arima_out, 12, arima_settings);
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
    arima_test(Y, d, p, q, benchmark_data);
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
    arima_test(Y, d, p, q, benchmark_data);
end
