###################################################################
###################################################################
## MC Random varaibles:
## Defines a class of random variables where the nodes can be 
## updated in place using a function called sampler!.
## This file defines the class and two constructors
## the firm constructs an instance of the calss given a function
## for drawing individual samples from the distribution and 
## a desiered number of sampls 
###################################################################
###################################################################

"""
    MCRandomVariable

A struct that represents a mulitvariate random variable with samples that can be updated in place.
...
Elements:
    - N: the number of samples
    - dims: the dimension of the random variable
    - nodes: a matrix of samples
    - weights: a vector of weights (1/N)
    - sample: a function for drawing samples from the distribution
"""
mutable struct MCRandomVariable <: AbstractRandomVariable
    N::Int
    dims::Int
    nodes::AbstractMatrix{} 
    weights::AbstractVector{Float64}
    sample::Function 
end 

function (MC::MCRandomVariable)()
    for i in 1:MC.N
        MC.nodes[1:MC.dims,i] = MC.sample()
    end
end 

"""
    MCRandomVariable(sample::Function, N::Int)

Initializes an instance of a MCRandomVariable using a sampler function and 
a desired number of samples. 

The MCRandomVariable stores the sampels in a matrix of size dims x N where dims is the 
dimension of the random variable and N is the number of samples. Calling the MCRandomVariable
object as a function will update the samples in place allowing for memory efficent resampling. 

The MCRandomVariable object also stores a vector of weights which are initialized to 1/N.
This allows the MCRandomVariable to be substituted for quadrature schemes represetned by 
the RandomVaraibles.jl interface. 
...
# Arguments:
    - sample: a function for drawing samples from the distribution
    - N: the number of samples
# Values:
    - a MCRandomVariable object
"""
function MCRandomVariable(sample::Function, N::Int)

    dims = length(sample())
    nodes = zeros(dims,N)
    weights = zeros(N).+1/N
    MC = MCRandomVariable(N, dims, nodes, weights,sample)
    MC()
    return MC
end 


"""
    MCRandomVariable(sample::Function, N::Int)

Initializes an instance of a  MCRandomVariable using a sampler function and 
a desired number of samples and a RandomVariable object.

This function will initialize a MCRandomVariable object that represents the cartesian
product of the variable represented by the sample funciton and X. Thsi is useful if 
you wants to represent the product of a continuous random variable with montecarlo methods 
and a discrete random variable by taking weighted sums. 

The MCRandomVariable object stores the samples in a matrix of size d x (M x N)  where d is the 
dimension of the random variable, and m is the number of nodes in X. Thw first elements of each node 
represent the sample from the sample function. The remaining elements represent the nodes from X.

The weights for each node are equal to 1/N times the corresponding weight from X.
...
# Arguments:
    - sample: a function for drawing samples from the distribution
    - X: a RandomVariable object 
    - N: the number of samples
# Values:
    - a MCRandomVariable object
"""
function MCRandomVariable(sample::Function, X::RandomVariable, N::Int)
    
    new_dims = length(sample())
    Nsample = N*size(X.nodes)[2]
  
    nodes = zeros(new_dims+size(X.nodes)[1],Nsample)
    weights = zeros(Nsample)
    k = 0
    for (i,j) in Iterators.product(1:N,1:size(X.nodes)[2])
        k +=1
        nodes[:,k] = vcat(sample(),X.nodes[:,j])
        weights[k] = X.weights[j]*1/N
    end
    MC = MCRandomVariable(Nsample, new_dims, nodes, weights, sample)
    MC()
    return MC
end 

