push!(LOAD_PATH,"../src/")
using Documenter, ValueFunctionIterations

makedocs(
    sitename="ValueFunctionIterations.jl",
    modules  = [ValueFunctionIterations],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["index.md","RandomVariables.md","API.md"]
)

deploydocs(
    repo = "github.com/Jack-H-Buckner/ValueFunctionIterations.jl.git",
)