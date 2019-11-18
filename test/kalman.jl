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

"""
    kalman_test(Y::JArray, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, V::SymMatrix, benchmark_data::Tuple)

Run a series of tests to check whether the kalman.jl functions work.
"""
function kalman_test(Y::JArray, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, V::SymMatrix, benchmark_data::Tuple)

    # Benchmark data
    benchmark_X0, benchmark_P0, benchmark_X_prior, benchmark_P_prior, benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik,
        benchmark_X0_sm, benchmark_P0_sm, benchmark_X_sm, benchmark_P_sm = benchmark_data;

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
        @test ksettings1.X0[1] == benchmark_X0;
        @test floor.(ksettings1.P0, digits=6)[1] == benchmark_P0;
        @test ksettings1.X0 == ksettings2.X0;
        @test ksettings1.X0 == ksettings3.X0;
        @test ksettings1.X0 == ksettings4.X0;
        @test ksettings1.X0 == ksettings5.X0;
        @test ksettings1.P0 == ksettings2.P0;
        @test ksettings1.P0 == ksettings3.P0;
        @test ksettings1.P0 == ksettings4.P0;
        @test ksettings1.P0 == ksettings5.P0;

        # Set default ksettings
        ksettings = ksettings5;

        # Initialise kstatus
        kstatus = KalmanStatus();

        for t=1:length(Y)

            # Run filter
            kfilter!(ksettings, kstatus);

            # A-priori
            @test floor.(kstatus.X_prior[1], digits=6) == benchmark_X_prior[t];
            @test floor.(kstatus.P_prior[1], digits=6) == benchmark_P_prior[t];

            # A-posteriori
            @test floor.(kstatus.X_post[1], digits=6) == benchmark_X_post[t];
            @test floor.(kstatus.P_post[1], digits=6) == benchmark_P_post[t];

            # 12-step ahead forecast
            @test floor.(kforecast(ksettings, kstatus.X_post, 12)[end], digits=6)[1] == benchmark_X_fc[t];
            @test floor.(kforecast(ksettings, kstatus.X_post, kstatus.P_post, 12)[1][end], digits=6)[1] == benchmark_X_fc[t];
            @test floor.(kforecast(ksettings, kstatus.X_post, kstatus.P_post, 12)[2][end], digits=6)[1] == benchmark_P_fc[t];
        end

        # Final value of the loglikelihood
        @test floor.(kstatus.loglik, digits=6) == benchmark_loglik

        # Kalman smoother
        X_sm, P_sm, X0_sm, P0_sm = ksmoother(ksettings, kstatus);

        for t=1:length(Y)
            @test floor.(X_sm[t], digits=6)[1] == benchmark_X_sm[t];
            @test floor.(P_sm[t], digits=6)[1] == benchmark_P_sm[t];
        end

        @test floor.(X0_sm, digits=6)[1] == benchmark_X0_sm;
        @test floor.(P0_sm, digits=6)[1] == benchmark_P0_sm;
    end
end

@testset "univariate model" begin

    # Initialise data and state-space parameters
    Y = [0.35 0.62 missing missing 1.11 missing 2.76 2.73 3.45 3.66];
    B = ones(1,1);
    R = Symmetric(1e-8*ones(1,1));
    C = 0.9*ones(1,1);
    V = Symmetric(ones(1,1));

    # Correct estimates (Kalman filter)
    benchmark_X0 = 0.0;
    benchmark_P0 = 5.263157;
    benchmark_X_prior = [0.00000; 0.314999; 0.557999; 0.502199; 0.451979; 0.998999; 0.899099; 2.483999; 2.456999; 3.104999];
    benchmark_P_prior = [5.263157; 1.0000; 1.0000; 1.8100; 2.4661; 1.0000; 1.8100; 1.0000; 1.0000; 1.0000];
    benchmark_X_post = [0.349999; 0.619999; 0.557999; 0.502199; 1.109999; 0.998999; 2.759999; 2.729999; 3.449999; 3.659999];
    benchmark_P_post = [0; 0; 1; 1.81; 0; 1; 0; 0; 0; 0];
    benchmark_X_fc = [0.098850; 0.175106; 0.157595; 0.141836; 0.313496; 0.282147; 0.779505; 0.771032; 0.974381; 1.033692];
    benchmark_P_fc = [4.843334; 4.843334; 4.923100; 4.987711; 4.843334; 4.923100; 4.843334; 4.843334; 4.843334; 4.843334];
    benchmark_loglik = -3.358198;

    # Correct estimates (Kalman smoother)
    benchmark_X0_sm = 0.315;
    benchmark_P0_sm = 1.0;
    benchmark_X_sm = [0.350000; 0.619999; 0.774129; 0.936859; 1.110000; 1.924309; 2.759999; 2.730000; 3.449999; 3.659999];
    benchmark_P_sm = [0.0; 0.0; 0.733952; 0.733952; 0.0; 0.552486; 0.0; 0.0; 0.0; 0.0];

    # Benchmark data
    benchmark_data = (benchmark_X0, benchmark_P0, benchmark_X_prior, benchmark_P_prior, benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik,
                      benchmark_X0_sm, benchmark_P0_sm, benchmark_X_sm, benchmark_P_sm);

    # Run tests
    kalman_test(Y, B, R, C, V, benchmark_data);
end
