# TSAnalysis.jl
TSAnalysis.jl includes basic tools for time series analysis and state-space modelling. 

The implementation for the Kalman filter and smoother uses symmetric matrices (via LinearAlgebra.jl). This is particularly beneficial for the stability and speed of estimation algorithms (e.g., the EM algorithm in Shumway and Stoffer, 1982), and to handle high-dimensional forecasting problems. 


##### Bibliography
R. H. Shumway and D. S. Stoffer. An approach to time series smoothing and forecasting using the EM algorithm. Journal of time series analysis, 3(4):253–264, 1982.

 
