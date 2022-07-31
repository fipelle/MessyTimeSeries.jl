"""
    ksettings_input_test(ksettings::KalmanSettings, Y::JArray, B::FloatMatrix, R::Union{UniformScaling{Float64}, SymMatrix}, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatArray, P0::SymMatrix, DQD::SymMatrix, n::Int64, T::Int64, m::Int64; compute_loglik::Bool=true, store_history::Bool=true)

Return true if the entries of ksettings are correct (false otherwise).
"""
function ksettings_input_test(ksettings::KalmanSettings, Y::JArray, B::FloatMatrix, R::Union{UniformScaling{Float64}, SymMatrix}, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, X0::FloatArray, P0::SymMatrix, DQD::SymMatrix, n::Int64, T::Int64, m::Int64; compute_loglik::Bool=true, store_history::Bool=true)
    
    return ~(false in [ksettings.Y.data === Y;
                      ksettings.B == B;
                      ksettings.R == R;
                      ksettings.C == C;
                      ksettings.D == D;
                      ksettings.Q == Q;
                      ksettings.X0 == X0;
                      round.(ksettings.P0, digits=10) == P0;
                      ksettings.DQD == DQD;
                      ksettings.Y.n == n;
                      ksettings.Y.T == T;
                      ksettings.m == m;
                      ksettings.compute_loglik == compute_loglik;
                      ksettings.store_history == store_history]);
end

"""
    test_kalman_output(ksettings::KalmanSettings, kstatus::DynamicKalmanStatus, benchmark_data::Tuple)
    test_kalman_output(ksettings::KalmanSettings, kstatus::SizedKalmanStatus, benchmark_data::Tuple)

kalman_test internals.
"""
function test_kalman_output(ksettings::KalmanSettings, kstatus::DynamicKalmanStatus, benchmark_data::Tuple)

    benchmark_n, benchmark_T, benchmark_m, benchmark_X0, benchmark_P0, benchmark_DQD, benchmark_X_prior, benchmark_P_prior, 
        benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik, benchmark_X0_sm, benchmark_P0_sm, 
            benchmark_X_sm, benchmark_P_sm = benchmark_data;

    for t=1:ksettings.Y.T

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

    # A-priori
    @test round.(hcat(kstatus.history_X_prior...), digits=10) == hcat(benchmark_X_prior...);
    @test round.(cat(dims=[1,2], kstatus.history_P_prior...), digits=10) == cat(dims=[1,2], benchmark_P_prior...);

    # A-posteriori
    @test round.(hcat(kstatus.history_X_post...), digits=10) == hcat(benchmark_X_post...);
    @test round.(cat(dims=[1,2], kstatus.history_P_post...), digits=10) == cat(dims=[1,2], benchmark_P_post...);

    # Final value of the loglikelihood
    @test round.(kstatus.loglik, digits=10) == benchmark_loglik;
end

function test_kalman_output(ksettings::KalmanSettings, kstatus::SizedKalmanStatus, benchmark_data::Tuple)

    benchmark_n, benchmark_T, benchmark_m, benchmark_X0, benchmark_P0, benchmark_DQD, benchmark_X_prior, benchmark_P_prior, 
        benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik, benchmark_X0_sm, benchmark_P0_sm, 
            benchmark_X_sm, benchmark_P_sm = benchmark_data;

    for t=1:ksettings.Y.T

        # Run filter
        kfilter!(ksettings, kstatus);

        # A-priori
        @test round.(kstatus.online_status.X_prior, digits=10) == benchmark_X_prior[t];
        @test round.(kstatus.online_status.P_prior, digits=10) == benchmark_P_prior[t];

        # A-posteriori
        @test round.(kstatus.online_status.X_post, digits=10) == benchmark_X_post[t];
        @test round.(kstatus.online_status.P_post, digits=10) == benchmark_P_post[t];

        # 12-step ahead forecast
        @test round.(kforecast(ksettings, kstatus.online_status.X_post, 12)[end], digits=10) == benchmark_X_fc[t];
        @test round.(kforecast(ksettings, kstatus.online_status.X_post, kstatus.online_status.P_post, 12)[1][end], digits=10) == benchmark_X_fc[t];
        @test round.(kforecast(ksettings, kstatus.online_status.X_post, kstatus.online_status.P_post, 12)[2][end], digits=10) == benchmark_P_fc[t];
    end

    # A-priori
    @test round.(hcat(kstatus.history_X_prior...), digits=10) == hcat(benchmark_X_prior...);
    @test round.(cat(dims=[1,2], kstatus.history_P_prior...), digits=10) == cat(dims=[1,2], benchmark_P_prior...);

    # A-posteriori
    @test round.(hcat(kstatus.history_X_post...), digits=10) == hcat(benchmark_X_post...);
    @test round.(cat(dims=[1,2], kstatus.history_P_post...), digits=10) == cat(dims=[1,2], benchmark_P_post...);
    
    # Final value of the loglikelihood
    @test round.(kstatus.online_status.loglik, digits=10) == benchmark_loglik;
end

"""
    kalman_test(Y::JArray, B::FloatMatrix, R::Union{UniformScaling{Float64}, SymMatrix}, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, benchmark_data::Tuple)

Run a series of tests to check whether the kalman.jl functions work.
"""
function kalman_test(Y::JArray, B::FloatMatrix, R::Union{UniformScaling{Float64}, SymMatrix}, C::FloatMatrix, D::FloatMatrix, Q::SymMatrix, benchmark_data::Tuple)

    # Benchmark data
    benchmark_n, benchmark_T, benchmark_m, benchmark_X0, benchmark_P0, benchmark_DQD, benchmark_X_prior, benchmark_P_prior, 
        benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik, benchmark_X0_sm, benchmark_P0_sm, 
            benchmark_X_sm, benchmark_P_sm = benchmark_data;

    for test_family_id=1:3

        # Tests on KalmanSettings (min. number of arguments)
        if test_family_id == 1
            D_I = Matrix(1.0I, benchmark_m, benchmark_m);
            test_family_input = (Y, B, R, C, benchmark_DQD);
            test_family_benchmark = (Y, B, R, C, D_I, benchmark_DQD, benchmark_X0, benchmark_P0, benchmark_DQD, benchmark_n, benchmark_T, benchmark_m);

        # Tests on KalmanSettings (std. number of arguments)
        elseif test_family_id == 2
            test_family_input = (Y, B, R, C, D, Q);
            test_family_benchmark = (Y, B, R, C, D, Q, benchmark_X0, benchmark_P0, benchmark_DQD, benchmark_n, benchmark_T, benchmark_m);
            
        # Tests on KalmanSettings (full number of arguments)
        elseif test_family_id == 3
            test_family_input = (Y, B, R, C, D, Q, benchmark_X0, benchmark_P0);
            test_family_benchmark = (Y, B, R, C, D, Q, benchmark_X0, benchmark_P0, benchmark_DQD, benchmark_n, benchmark_T, benchmark_m);
        end

        # Tests on KalmanSettings (full number of arguments)
        ksettings1 = KalmanSettings(test_family_input..., compute_loglik=true, store_history=true);
        @test ksettings_input_test(ksettings1, test_family_benchmark..., compute_loglik=true, store_history=true);

        ksettings2 = KalmanSettings(test_family_input..., compute_loglik=false, store_history=true);
        @test ksettings_input_test(ksettings2, test_family_benchmark..., compute_loglik=false, store_history=true);

        ksettings3 = KalmanSettings(test_family_input..., compute_loglik=true, store_history=false);
        @test ksettings_input_test(ksettings3, test_family_benchmark..., compute_loglik=true, store_history=false);

        ksettings4 = KalmanSettings(test_family_input..., compute_loglik=false, store_history=false);
        @test ksettings_input_test(ksettings4, test_family_benchmark..., compute_loglik=false, store_history=false);

        ksettings5 = KalmanSettings(test_family_input...);
        @test ksettings_input_test(ksettings5, test_family_benchmark...);

        # Select default ksettings
        ksettings = ksettings5;

        # Initialise kstatus
        for kstatus in [DynamicKalmanStatus(), SizedKalmanStatus(ksettings)]

            test_kalman_output(ksettings, kstatus, benchmark_data);

            # Kalman smoother
            X_sm, P_sm, X0_sm, P0_sm = ksmoother(ksettings, kstatus);

            for t=1:ksettings.Y.T
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
    D = ones(1,1);
    Q = Symmetric(ones(1,1));
    
    # Correct estimates: DQD
    benchmark_DQD = copy(Q);
    
    # Correct estimates: initial conditions
    benchmark_X0 = read_test_input("./input/univariate/benchmark_X0");
    benchmark_P0 = Symmetric(read_test_input("./input/univariate/benchmark_P0"));

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
    benchmark_data = (1, size(Y,2), 1, benchmark_X0, benchmark_P0, benchmark_DQD, benchmark_X_prior, benchmark_P_prior, benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik,
                      benchmark_X0_sm, benchmark_P0_sm, benchmark_X_sm, benchmark_P_sm);

    # Run tests
    kalman_test(Y, B, R, C, D, Q, benchmark_data);
    kalman_test(Y, B, 1e-4I, C, D, Q, benchmark_data); # same test with UniformScaling R
end

@testset "multivariate model" begin

    # Initialise data and state-space parameters
    Y = [0.72 missing 1.86 missing missing 2.52 2.98 3.81 missing 4.36;
         0.95 0.70 missing missing missing missing 2.84 3.88 3.84 4.63];

    B = [1.0 0.0; 1.0 1.0];
    R = Symmetric(1e-4*Matrix(I,2,2));
    C = [0.9 0.0; 0.0 0.1];
    D = Matrix(I,2,2) |> FloatMatrix;
    Q = Symmetric(D);
    
    # Correct estimates: DQD
    benchmark_DQD = copy(Q);

    # Correct estimates: initial conditions
    benchmark_X0 = read_test_input("./input/multivariate/benchmark_X0");
    benchmark_P0 = Symmetric(read_test_input("./input/multivariate/benchmark_P0"));

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
    benchmark_data = (2, size(Y,2), 2, benchmark_X0, benchmark_P0, benchmark_DQD, benchmark_X_prior, benchmark_P_prior, benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik,
                      benchmark_X0_sm, benchmark_P0_sm, benchmark_X_sm, benchmark_P_sm);

    # Run tests
    kalman_test(Y, B, R, C, D, Q, benchmark_data);
    kalman_test(Y, B, 1e-4I, C, D, Q, benchmark_data); # same test with UniformScaling R
end

@testset "multivariate model (non-diagonal)" begin
    
    # Initialise data and state-space parameters
    Y = [0.49 missing 1.01	-0.41 1.33 1.5 -0.36 -1.14 0.11 missing;
         0.02 missing 0.20 0.14 missing 0.85 missing -0.57 missing 0.11;
         missing missing 1.15 -0.65 1.32 1.28 -0.57 -1.03 0.14 1.53];

    B = [1 0; 0.5 -0.25; 0.9 0.2];
    R = Symmetric(1e-4*Matrix(I,3,3));
    C = [0.6 0.1; 0.25 -0.2];
    D = [1.0 0.0; 0.5 0.86602540];
    Q = Symmetric(1.0*Matrix(I,2,2));
    
    # Correct estimates: DQD
    benchmark_DQD = Symmetric(D*Q*D');

    # Correct estimates: initial conditions
    benchmark_X0 = read_test_input("./input/multivariate_non_diagonal/benchmark_X0");
    benchmark_P0 = Symmetric(read_test_input("./input/multivariate_non_diagonal/benchmark_P0"));

    # Correct estimates: a priori
    benchmark_X_prior = read_test_input("./input/multivariate_non_diagonal/benchmark_X_prior");
    benchmark_P_prior = read_test_input("./input/multivariate_non_diagonal/benchmark_P_prior");

    # Correct estimates: a posteriori
    benchmark_X_post = read_test_input("./input/multivariate_non_diagonal/benchmark_X_post");
    benchmark_P_post = read_test_input("./input/multivariate_non_diagonal/benchmark_P_post");

    # Correct estimates: 12-step ahead forecast
    benchmark_X_fc = read_test_input("./input/multivariate_non_diagonal/benchmark_X_fc");
    benchmark_P_fc = read_test_input("./input/multivariate_non_diagonal/benchmark_P_fc");

    # Correct estimates: loglikelihood
    benchmark_loglik = read_test_input("./input/multivariate_non_diagonal/benchmark_loglik")[1];

    # Correct estimates: kalman smoother (smoothed initial conditions)
    benchmark_X0_sm = read_test_input("./input/multivariate_non_diagonal/benchmark_X0_sm");
    benchmark_P0_sm = read_test_input("./input/multivariate_non_diagonal/benchmark_P0_sm");

    # Correct estimates: kalman smoother
    benchmark_X_sm = read_test_input("./input/multivariate_non_diagonal/benchmark_X_sm");
    benchmark_P_sm = read_test_input("./input/multivariate_non_diagonal/benchmark_P_sm");

    # Benchmark data
    benchmark_data = (3, size(Y,2), 2, benchmark_X0, benchmark_P0, benchmark_DQD, benchmark_X_prior, benchmark_P_prior, benchmark_X_post, benchmark_P_post, benchmark_X_fc, benchmark_P_fc, benchmark_loglik,
                      benchmark_X0_sm, benchmark_P0_sm, benchmark_X_sm, benchmark_P_sm);

    # Run tests
    kalman_test(Y, B, R, C, D, Q, benchmark_data);
    kalman_test(Y, B, 1e-4I, C, D, Q, benchmark_data); # same test with UniformScaling R
end