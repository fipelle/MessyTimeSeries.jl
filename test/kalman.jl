using LinearAlgebra, Test, TSAnalysis;

"""
    ksettings_input_test(ksettings::KalmanSettings, Y::JArray, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, V::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

Return true if the entries of ksettings are correct (false otherwise).
"""
function ksettings_input_test(ksettings::KalmanSettings, Y::JArray, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, V::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)
    return ~false in [ksettings.Y == Y;
                      ksettings.B == B;
                      ksettings.R == R;
                      ksettings.C == C;
                      ksettings.V == V;
                      ksettings.compute_loglik == compute_loglik;
                      ksettings.store_history == store_history];
end

@testset "univariate model" begin

    # Initialise data and state-space parameters
    Y = [0.35 0.62 missing 1.00 1.11 1.95 2.76 2.73 3.45 3.66];
    B = ones(1,1);
    R = Symmetric(1e-8*ones(1,1));
    C = 0.9*ones(1,1);
    V = Symmetric(ones(1,1));

    # Loop over ImmutableKalmanSettings and MutableKalmanSettings
    for ksettings_type = [ImmutableKalmanSettings; MutableKalmanSettings]

        # Tests on KalmanSettings
        ksettings1 = ksettings_type(Y, B, R, C, V, compute_loglik=true, store_history=true);
        @test ksettings_input_test(ksettings1, Y, B, R, C, V, compute_loglik=true, store_history=true);

        ksettings2 = ksettings_type(Y, B, R, C, V, compute_loglik=false, store_history=true);
        @test ksettings_input_test(ksettings2, Y, B, R, C, V, compute_loglik=false, store_history=true);

        ksettings3 = ksettings_type(Y, B, R, C, V, compute_loglik=true, store_history=false);
        @test ksettings_input_test(ksettings3, Y, B, R, C, V, compute_loglik=true, store_history=false);

        ksettings4 = ksettings_type(Y, B, R, C, V, compute_loglik=false, store_history=false);
        @test ksettings_input_test(ksettings4, Y, B, R, C, V, compute_loglik=false, store_history=false);

        ksettings5 = ksettings_type(Y, B, R, C, V);
        @test ksettings_input_test(ksettings5, Y, B, R, C, V);

        # Initial conditions
        @test ksettings1.X0 == [0.0];
        @test floor.(ksettings1.P0, digits=8)[1] == 5.26315789;
        @test ksettings1.X0 == ksettings2.X0;
        @test ksettings1.X0 == ksettings3.X0;
        @test ksettings1.X0 == ksettings4.X0;
        @test ksettings1.X0 == ksettings5.X0;
        @test ksettings1.P0 == ksettings2.P0;
        @test ksettings1.P0 == ksettings3.P0;
        @test ksettings1.P0 == ksettings4.P0;
        @test ksettings1.P0 == ksettings5.P0;

        # Set default ksettings
        ksettings = copy(ksettings5);

        #=
        # First prediction
        @test

        # First update
        @test

        # First forecast
        @test

        # Prediction
        @test

        # Update
        @test

        # Forecast
        @test

        #=
        # Prediction with some missing observations
        @test

        # Update with some missing observations
        @test

        # Forecast with some missing observations
        @test
        =#

        # Prediction with missing observations (only)
        @test

        # Update with missing observations (only)
        @test

        # Forecast with missing observations (only)
        @test

        # Last prediction
        @test

        # Last update
        @test

        # Last forecast
        @test

        # Kalman smoother (last period)
        @test

        # Kalman smoother (first period)
        @test
        =#
    end
end
