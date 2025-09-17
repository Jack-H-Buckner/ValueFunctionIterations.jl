# ValueFunctionIterations.jl

![Build Status](https://github.com/jack-h-buckner/ValueFunctionIterations.jl/actions/workflows/CI-V1-11.yml/badge.svg)

[![Docs](https://img.shields.io/badge/docs-dev-blue)](https://jack-h-buckner.github.io/ValueFunctionIterations.jl/dev/)

ValueFunctionIterations.jl provides a framework for solving stochastic dynamic programs with continuous states and discrete action spaces. Problems are defined by specifying a reward function `R` and state update function `F` which depend on the current state `s`, action `u`, randon varible `X` and model paramters `p`. The user also specifies the set of all possible actions `U`, a grid of the state space for the value and policy function approxiamtions and a discount factor $\delta \in [0,1)$. 

ValueFunctionIterations.jl approximate the value function for the dynamic program using the Bsplines from Interpolations.jl and solve the bellman equaitons using value funciton iteration. Random varibles can be included in the model using the AbstractRandomVariable interace provided by ValueFunctionIterations.jl. This interface allows the expectation in the bellman euqaiton to be evaluated using either Montecarlo methods, quadrature, or a mixture of the two. 

The solution to the dynamic program is stored in a DynamicProgram object. This stores the data used to define the problem along with the value and policy functions. 

To get started you can load the library from github using

```
]add https://github.com/Jack-H-Buckner/ValueFunctionIterations.jl.git
using ValueFunctionIterations
```

## Example

```julia
using Plots, ValueFunctionIterations, ComponentArrays, Distribtions

# Income from harvesting trees
function R(s,u,X,p)
    if X[1] == 0.0 && u[1] == 0 # if neither damage or harvest recieve nothing
        return 0
    elseif u[1] == 1 # If harvesting occurs recive net revenue 
        return s[1]-p.c
    else X[1] == 1 # If damage occurs, harvest and revice the salvage value (X[2])
        return X[2]*s[1]-p.c
    end
end

# State update function 
function F(s,u,X,p)
    if X[1] == 0.0 && u[1] == 0 # if neither damage or harvest allow growth
        return s[1]*exp(p.r*(1-s[1]/p.k))
    else  # If harvesting or damage occurs go to replanted biomass 
        return 0.02
    end
end 

# Parameters 
# r: growth rate, k: maximum growth, c: cost of harvest, p_s: price for damaged timber 
p = ComponentArray(r = 0.15, k = 1.0, c = 0.25)

# Harvest levels (0 or 97.5%)
u = action_space([0.0,1.0])

# Damage levels (0 or 97.5%) and probabilities (0.99 and 0.01)
X1 = RandomVariable([0.0 1.0;], [0.99, 0.01])

# Define quadrature scheme for normally distributed salvage values using Gauss-Hermite quadrature 
X2 = GaussHermiteRandomVariable(10,[0.5],[0.1^2;;])

# Combine the two random variables 
X = product(X1,X2)


# Discount factor 
δ = 0.99

# Grid of stand sizes
grid = 0.01:0.01:1.00

# Define and solve the dynamic program
sol = DynamicProgram(R, F, p, u,  X, δ, grid; tolerance = 1e-5)

# Plot the policy function 
Plots.plot(grid,broadcast(s -> s - s*sol.P(s)[1],grid), xlabel = "Standing timber",
             ylabel = "Board feet", label = "Standing timber",linewidth = 2)
Plots.plot!(grid,broadcast(s -> s*sol.P(s)[1],grid), label = "Harvest", linewidth = 2)
plot!(size = (400,250))
```
![](docs/src/figures/trees_policy.png)
