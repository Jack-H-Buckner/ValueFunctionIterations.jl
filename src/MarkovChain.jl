###################################################################
###################################################################
## MarkovChains
## Defines methods for building markov chagnes that are comptable 
## with the valueFunctionIterations.jl and RandomVariables.jl interface.
###################################################################
###################################################################

"""
    sample_discrete(p,rng)

Samples an index 1:n with probability mass for each index given by p, 
given a draw from a uniform random variable rng on the set (0,1).
...
# Arguments:
    - p: a vector of probabilities
    - rng: a number in [0,1]

# Values:
    - an integer in 1:length(p)
"""
function sample_discrete(p,rng)
    minimum(collect(1:length(p))[cumsum(p) .>= rng])
end 

"""
    sample_markov_chain(x,p,rng)

Samples from a markov chain given the current state x and the transition matrix p
using a uniform random variable rng on the unit interval.
...
# Arguments:
    - x: the current state (integer in 1:m)
    - p: a transition probability matrix of size m
    - rng: a number in [0,1]

# Values:
    - an integer in 1:m
"""
function sample_markov_chain(x,p,rng)
    sample_discrete(p[:,x],rng)
end 

"""
    MarkovChain(p)

Builds RandomVariable object that represents a markov chain and is compatable with ValueFunctionIterations.jl.
The goal of this object is to  take expectations over the outcome of a markov chain in the 
most efficnet possible way using the RandomVariables.jl interface. 

The RandomVariable object is definged to work with the `sample_markov_chain` function. 
The nodes are intended to be passed to the `sample_markov_chain` function as the random number argumet.
If the nodes are sampled using the weights stored in the random varible object and passed to 
`sample_markov_chain` the results wil be the same as if a uniform random number was sampled. 

This allows the weights and nodes to be used to calcualte expectations over the outcome of the markov chain.
with the minimum number of computations possible without changing the weights as a function of the current state. 
...
# Arguments:
    - p: a transition probability matrix of size m

# Values:
    - a RandomVariable object 
"""
function MarkovChain(p)
    pcopy = deepcopy(p)
    probs = unique(sort!(reshape(cumsum(pcopy,dims = 1),prod(size(pcopy)))))
    nodes = reshape(probs,1,length(probs))
    weights = vcat(nodes[1:1],nodes[2:end] .- nodes[1:(end-1)])
    return RandomVariable(nodes, weights)
end 
