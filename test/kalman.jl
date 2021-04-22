"""
    ksettings_input_test(ksettings::KalmanSettings, Y::JArray, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatArray, P0::SymMatrix, DQD::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)

Return true if the entries of ksettings are correct (false otherwise).
"""
function ksettings_input_test(ksettings::KalmanSettings, Y::JArray, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatArray, P0::SymMatrix, DQD::SymMatrix; compute_loglik::Bool=true, store_history::Bool=true)
    return ~false in [ksettings.Y == Y;
                      ksettings.B == B;
                      ksettings.R == R;
                      ksettings.C == C;
                      ksettings.D == D;
                      ksettings.Q == Q;
                      ksettings.X0 == X0;
                      round.(ksettings.P0, digits=10) == benchmark_P0;
                      ksettings.DQD == DQD;
                      ksettings.n == n;
                      ksettings.T == T;
                      ksettings.m == m;
                      ksettings.compute_loglik == compute_loglik;
                      ksettings.store_history == store_history];
end

"""
    kalman_test(Y::JArray, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatArray, P0::SymMatrix, DQD::SymMatrix, n::Int64, T::Int64, m::Int64, benchmark_data::Tuple)

Run a series of tests to check whether the kalman.jl functions work.
"""
function kalman_test(Y::JArray, B::FloatMatrix, R::SymMatrix, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatArray, P0::SymMatrix, DQD::SymMatrix, n::Int64, T::Int64, m::Int64, benchmark_data::Tuple)

    # Benchmark data
    benchmark_X0, benchmark_P0, benchmark_X_prior, benchmark_P_prior, benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik,
        benchmark_X0_sm, benchmark_P0_sm, benchmark_X_sm, benchmark_P_sm = benchmark_data;

    # Loop over ImmutableKalmanSettings and MutableKalmanSettings
    for ksettings_type = [ImmutableKalmanSettings; MutableKalmanSettings]

        for test_family_id=1:3

            # Tests on KalmanSettings (min. number of arguments for ksettings_type)
            if test_family_id == 1
                D_I = Matrix(I, m, m) |> FloatMatrix;
                test_family_input = (Y, B, R, C, DQD);
                test_family_benchmark = (Y, B, R, C, D_I, DQD, X0, P0, DQD);

            # Tests on KalmanSettings (std. number of arguments for ksettings_type)
            elseif test_family_id == 2
                test_family_input = (Y, B, R, C, D, Q);
                test_family_benchmark = (Y, B, R, C, D, Q, X0, P0, DQD);
            
            # Tests on KalmanSettings (full number of arguments for ksettings_type)
            elseif test_family_id == 3
                test_family_input = (Y, B, R, C, D, Q, X0, P0);
                test_family_benchmark = (Y, B, R, C, D, Q, X0, P0, DQD);
            end

            # Tests on KalmanSettings (full number of arguments for ksettings_type)
            ksettings1 = ksettings_type(test_family_input..., compute_loglik=true, store_history=true);
            @test ksettings_input_test(ksettings1, test_family_benchmark..., compute_loglik=true, store_history=true);

            ksettings2 = ksettings_type(test_family_input..., compute_loglik=false, store_history=true);
            @test ksettings_input_test(ksettings2, test_family_benchmark..., compute_loglik=false, store_history=true);

            ksettings3 = ksettings_type(test_family_input..., compute_loglik=true, store_history=false);
            @test ksettings_input_test(ksettings3, test_family_benchmark..., compute_loglik=true, store_history=false);

            ksettings4 = ksettings_type(test_family_input..., compute_loglik=false, store_history=false);
            @test ksettings_input_test(ksettings4, test_family_benchmark..., compute_loglik=false, store_history=false);

            ksettings5 = ksettings_type(test_family_input...);
            @test ksettings_input_test(ksettings5, test_family_benchmark...);

            # Select default ksettings
            ksettings = ksettings5;

            # Initialise kstatus
            kstatus = KalmanStatus();

            for t=1:size(Y,2)

                # Run filter
                kfilter!(ksettings, kstatus);

                # A-priori
                @test round.(kstatus.X_prior, digits=10) == benchmark_X_prior[t];
                @test round.(kstatus.P_prior, digits=10) == benchmark_P_prior[t];

                # A-posteriori
                @test round.(kstatus.X_post, digits=10) == benchmark_X_post[t];
                @test round.(kstatus.P_post, digits=10) == benchmark_P_post[t];

                # 12-step ahead forecast
                @test round.(kforecast(ksettings, kstatus.X_post, 12)[end], digits=10) == benchmark_X_fc[t];
                @test round.(kforecast(ksettings, kstatus.X_post, kstatus.P_post, 12)[1][end], digits=10) == benchmark_X_fc[t];
                @test round.(kforecast(ksettings, kstatus.X_post, kstatus.P_post, 12)[2][end], digits=10) == benchmark_P_fc[t];
            end

            # Final value of the loglikelihood
            @test round.(kstatus.loglik, digits=10) == benchmark_loglik;

            # Kalman smoother
            X_sm, P_sm, X0_sm, P0_sm = ksmoother(ksettings, kstatus);

            for t=1:size(Y,2)
                @test round.(X_sm[t], digits=10) == benchmark_X_sm[t];
                @test round.(P_sm[t], digits=10) == benchmark_P_sm[t];
            end

            @test round.(X0_sm, digits=10) == benchmark_X0_sm;
            @test round.(P0_sm, digits=10) == benchmark_P0_sm;
        end
    end
end

@testset "univariate model" begin

    # Initialise data and state-space parameters
    Y = [0.35 0.62 missing missing 1.11 missing 2.76 2.73 3.45 3.66];
    B = ones(1,1);
    R = Symmetric(1e-4*ones(1,1));
    C = 0.9*ones(1,1);
    V = Symmetric(ones(1,1));

    # Correct estimates: initial conditions
    benchmark_X0 = read_test_input("./input/univariate/benchmark_X0");
    benchmark_P0 = read_test_input("./input/univariate/benchmark_P0");

    # Correct estimates: a priori
    benchmark_X_prior = read_test_input("./input/univariate/benchmark_X_prior");
    benchmark_P_prior = read_test_input("./input/univariate/benchmark_P_prior");

    # Correct estimates: a posteriori
    benchmark_X_post = read_test_input("./input/univariate/benchmark_X_post");
    benchmark_P_post = read_test_input("./input/univariate/benchmark_P_post");

    # Correct estimates: 12-step ahead forecast
    benchmark_X_fc = read_test_input("./input/univariate/benchmark_X_fc");
    benchmark_P_fc = read_test_input("./input/univariate/benchmark_P_fc");

    # Correct estimates: loglikelihood
    benchmark_loglik = read_test_input("./input/univariate/benchmark_loglik")[1];

    # Correct estimates: kalman smoother (smoothed initial conditions)
    benchmark_X0_sm = read_test_input("./input/univariate/benchmark_X0_sm");
    benchmark_P0_sm = read_test_input("./input/univariate/benchmark_P0_sm");

    # Correct estimates: kalman smoother
    benchmark_X_sm = read_test_input("./input/univariate/benchmark_X_sm");
    benchmark_P_sm = read_test_input("./input/univariate/benchmark_P_sm");

    # Benchmark data
    benchmark_data = (benchmark_X0, benchmark_P0, benchmark_X_prior, benchmark_P_prior, benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik,
                      benchmark_X0_sm, benchmark_P0_sm, benchmark_X_sm, benchmark_P_sm);

    # Run tests
    kalman_test(Y, B, R, C, V, benchmark_data);
end

@testset "multivariate model" begin

    # Initialise data and state-space parameters
    Y = [0.72 missing 1.86 missing missing 2.52 2.98 3.81 missing 4.36;
         0.95 0.70 missing missing missing missing 2.84 3.88 3.84 4.63];

    B = [1.0 0.0; 1.0 1.0];
    R = Symmetric(1e-4*Matrix(I,2,2));
    C = [0.9 0.0; 0.0 0.1];
    V = Symmetric(1.0*Matrix(I,2,2));

    # Correct estimates: initial conditions
    benchmark_X0 = read_test_input("./input/multivariate/benchmark_X0");
    benchmark_P0 = read_test_input("./input/multivariate/benchmark_P0");

    # Correct estimates: a priori
    benchmark_X_prior = read_test_input("./input/multivariate/benchmark_X_prior");
    benchmark_P_prior = read_test_input("./input/multivariate/benchmark_P_prior");

    # Correct estimates: a posteriori
    benchmark_X_post = read_test_input("./input/multivariate/benchmark_X_post");
    benchmark_P_post = read_test_input("./input/multivariate/benchmark_P_post");

    # Correct estimates: 12-step ahead forecast
    benchmark_X_fc = read_test_input("./input/multivariate/benchmark_X_fc");
    benchmark_P_fc = read_test_input("./input/multivariate/benchmark_P_fc");

    # Correct estimates: loglikelihood
    benchmark_loglik = read_test_input("./input/multivariate/benchmark_loglik")[1];

    # Correct estimates: kalman smoother (smoothed initial conditions)
    benchmark_X0_sm = read_test_input("./input/multivariate/benchmark_X0_sm");
    benchmark_P0_sm = read_test_input("./input/multivariate/benchmark_P0_sm");

    # Correct estimates: kalman smoother
    benchmark_X_sm = read_test_input("./input/multivariate/benchmark_X_sm");
    benchmark_P_sm = read_test_input("./input/multivariate/benchmark_P_sm");

    # Benchmark data
    benchmark_data = (benchmark_X0, benchmark_P0, benchmark_X_prior, benchmark_P_prior, benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik,
                      benchmark_X0_sm, benchmark_P0_sm, benchmark_X_sm, benchmark_P_sm);

    # Run tests
    kalman_test(Y, B, R, C, V, benchmark_data);
end
