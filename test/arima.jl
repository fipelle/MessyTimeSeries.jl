"""
    arima_test(arima_settings::ARIMASettings, benchmark_data::Tuple)

Run a series of tests to check whether the arima functions in uc_models.jl work.
"""
function arima_test(arima_settings::ARIMASettings, benchmark_data::Tuple)

    # TODO: add check for ARIMASettings constructor

    # Estimate parameters
    arima_out = arima(arima_settings, NelderMead(), Optim.Options(iterations=10000, f_tol=1e-4, x_tol=1e-4, show_trace=true, show_every=500));

    # 12-step ahead forecast
    fc = forecast(arima_out, 12, arima_settings);

end

@testset "arma" begin

    # Load data
    Y = permutedims(read_test_input("./input/arma/data"));

    # Settings for ARMA(1,1)
    arima_settings = ARIMASettings(Y, 0, 1, 1);

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

end
