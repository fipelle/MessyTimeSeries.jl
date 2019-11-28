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
    Y = permutedims(read_test_input("./input/arima/arma"));

    # Settings for ARMA(1,1)
    arima_settings = ARIMASettings(Y, 0, 1, 1);
end
