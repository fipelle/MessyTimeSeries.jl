using Documenter, MessyTimeSeries;
makedocs(
    sitename="MessyTimeSeries.jl",     
    pages = [
        "index.md",
        "man/getting_started.md",
        "Manual" => ["man/kalman.md", "man/subsampling.md"],
        "Appendix" => ["man/methods.md"],
    ]
);

deploydocs(
    repo = "github.com/fipelle/MessyTimeSeries.jl.git",
)