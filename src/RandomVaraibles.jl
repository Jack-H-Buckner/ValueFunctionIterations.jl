###################################################################
###################################################################
##  RandomVariable:
## defines a class called   random variables which 
## is defined by a matrix of nodes corresponding to possible values
## and a vector of weights corresponding to the probabi sereis of
## the nodes. 
## The values could correspond to the weights and nodes of a quadrature
## scheme designed to integrate over a distribution, samples drawn
## from a probability distribution using monte carlo methods,
## or exacte values and probabiltiies of a discrete random variable.
## the calss can also allow combinations of these types of random 
## variables. 
## The class is intended to be used to solve integrates of function of the 
## random variables, but evaluating the function at the nodes and
## then taking the weighted average usign the weights vector.
###################################################################
###################################################################


"""
    RandomVariable

This class defines a quadrature scheme for multi variate
random varaibles. The nodes are a matrix of values with each colum corresponding 
to a point in the set of possible samples and the weights give the
probabiltiy of that point. 
...
Elements:
    - nodes: a matrix of values
    - weights: a vector of weights
"""
mutable struct RandomVariable <: AbstractRandomVariable
    nodes
    weights
end 

"""
    product(X:: RandomVariable, Y:: RandomVariable)

Returns a RandomVariable that is the cartesian product of two independent RandomVariables. 
...
# Arguments
    - X: a RandomVariable
    - Y: a RandomVariable
# Values
    - a RandomVariable
"""
function product(X::RandomVariable, Y::RandomVariable)

    nodes = zeros(size(X.nodes)[1]+size(Y.nodes)[1],size(X.nodes)[2]*size(Y.nodes)[2])
    weights = zeros(size(X.nodes)[2]*size(Y.nodes)[2])
    k = 0
    for (i,j) in Iterators.product(1:size(X.nodes)[2],1:size(Y.nodes)[2])
        k +=1
        nodes[:,k] = vcat(X.nodes[:,i],Y.nodes[:,j])
        weights[k] = X.weights[i]*Y.weights[j]
    end
    

    return RandomVariable(nodes,weights)

end


mutable struct RandomVariableFunction <: AbstractRandomVariable
    F::Function
end 

function (X::RandomVariableFunction)(s::Vector{Float64},p::ComponentArray)
    X.F(s,p)
end 





