# ValueFunctionIterations.jl

[![Build Status](https://github.com/jack-h-buckner/ValueFunctionIterations.jl/actions/workflows/CI-V1-11.yml/badge.svg)]

[![Docs](https://img.shields.io/badge/docs-dev-blue)](https://jack-h-buckner.github.io/ValueFunctionIterations.jl)

ValueFunctionIterations.jl provides a framework for solving stochastic dynamic programs with continuous states and discrete action spaces. Problems are defined by specifying a reward function `R` and state update function `F` which depend on the current state `s`, action `u`, randon varible `X` and model paramters `p`. The user also specifies the set of all possible actions `U`, a grid of the state space for the value and policy function approxiamtions and a discount factor $\delta \in [0,1)$. 

ValueFunctionIterations.jl approximate the value function for the dynamic program using the Bsplines from Interpolations.jl and solve the bellman equaitons using value funciton iteration. Random varibles can be included in the model using the AbstractRandomVariable interace provided by ValueFunctionIterations.jl. This interface allows the expectation in the bellman euqaiton to be evaluated using either Montecarlo methods, quadrature, or a mixture of the two. 

The solution to the dynamic program is stored in a DynamicProgram object. This stores the data used to define the problem along with the value and policy functions. The  

