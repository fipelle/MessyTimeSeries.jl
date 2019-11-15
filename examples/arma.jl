# Load packages
using TSAnalysis, Optim;

# Load FredData
using FredData;
f = Fred();

# Download Industrial Production: Manufacturing (NAICS), Log-levels (monthly, NSA)
fred_df = get_data(f, "INDPRO", observation_start="1984-01-01", units="log");

# Store data in Array{Float64,1}
Y = fred_df.data.value;

# Take YoY% transformation of the data
Y=permutedims(Y[13:end]-Y[1:end-12]);

# Settings for ARMA(1,1)
arima_settings = ARIMASettings(Y, 0, 1, 1);

# Estimate parameters
arima_out = arima(arima_settings, NelderMead(), Optim.Options(iterations=10000, f_tol=1e-4, x_tol=1e-4, show_trace=true, show_every=500));

# 12-step ahead forecast
fc = forecast(arima_out, 12, arima_settings);
