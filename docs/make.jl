using Documenter, MessyTimeSeries;
makedocs(
    sitename="MessyTimeSeries.jl",     
    pages = [
        "index.md",
        "Main content" => ["man/kalman.md", "man/subsampling.md"],
        "Subsection" => ["man/methods.md"],
    ]
);