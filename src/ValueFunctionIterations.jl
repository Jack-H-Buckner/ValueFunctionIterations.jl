###############################################################
###############################################################
## Value Function Iterations
## performse value funciton iteration for dynamic optimization 
## problems with continuous states andand discrete action 
## spaces. 
## Author: Jack H. Buckner
## Date: 9/2025
## Oregon State University 
###############################################################
###############################################################

module ValueFunctionIterations

using Interpolations, FastGaussQuadrature, LinearAlgebra, ComponentArrays, StatsBase, ProgressMeter, JLD2 

include("AbstractRandomVariables.jl")
include("ValueFunctions.jl")
include("bellman.jl")
include("DynamicPrograms.jl")
include("analysis.jl")
include("action_spaces.jl")

export DynamicProgram, RandomVariable, ValueFunction, MarkovChain, MCRandomVariable, GaussHermiteRandomVariable, product, sample_discrete, sample_markov_chain, action_spaces, estimate_time, solve!, Constant, BSpline, Cubic, Line, simulate, save_solution, load_solution 

end # module