using ValueFunctionIterations, Random, StatsBase, Roots, Distributions, ComponentArrays
using Test

@testset "ValueFunctionIterations.jl" begin 
    include("test_markov_chain.jl")
    include("test_MC_integration.jl")
    include("test_quad_integration.jl")
    include("test_VFI.jl")
end