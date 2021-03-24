using LinearInterpolations
using Documenter

DocMeta.setdocmeta!(LinearInterpolations, :DocTestSetup, :(using LinearInterpolations); recursive=true)

makedocs(;
    modules=[LinearInterpolations],
    authors="Jan Weidner <jw3126@gmail.com> and contributors",
    repo="https://github.com/jw3126/LinearInterpolations.jl/blob/{commit}{path}#{line}",
    sitename="LinearInterpolations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jw3126.github.io/LinearInterpolations.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jw3126/LinearInterpolations.jl",
)
