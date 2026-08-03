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
    cumulative = zero(eltype(p))
    @inbounds for i in eachindex(p)
        cumulative += p[i]
        if cumulative >= rng
            return i
        end
    end
    # rng can exceed sum(p) by a few units in the last place when the
    # probabilities are the product of floating point arithmetic. The last
    # index is the correct answer in that case.
    return lastindex(p)
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

The columns of p are normalized to sum to one before the nodes are built, so columns that are off
by a small amount of rounding are handled exactly. A column that is off by more than `tol` is
treated as an error rather than as rounding, as are negative or non finite entries.

Note that the nodes are points inside the intervals that `sample_discrete` maps to each state,
they are not the cumulative probabilities themselves.
...
# Arguments:
    - p: a transition probability matrix of size m
    - tol: how far a column of p may sum from one before it is treated as an error

# Values:
    - a RandomVariable object
"""
function MarkovChain(p; tol = 1e-6)
    if any(x -> !isfinite(x) || x < 0, p)
        throw(ArgumentError("the transition matrix contains negative or non finite entries"))
    end

    totals = [sum(p[:,j]) for j in axes(p,2)]
    bad = findall(t -> abs(t - 1) > tol, totals)
    if !isempty(bad)
        throw(ArgumentError("columns $(bad) of the transition matrix sum to $(totals[bad]), "*
                            "which differs from one by more than tol = $(tol)"))
    end

    # normalizing means the nodes describe the transition matrix the user intended, p./sum(p),
    # rather than p itself. This is only safe because the nodes are placed in the interior of
    # their intervals below, sample_markov_chain is called with the original p and so compares
    # the nodes against breakpoints that differ from these by the rounding in each column.
    cumulatives = [cumsum(p[:,j]) ./ totals[j] for j in axes(p,2)]

    # states with no probability leave an empty interval behind, unique drops the ones in the
    # middle of a column and the filter drops the ones at the start of a column
    breakpoints = sort!(unique(reduce(vcat, cumulatives)))
    filter!(x -> x > 0, breakpoints)
    weights = vcat(breakpoints[1:1], breakpoints[2:end] .- breakpoints[1:(end-1)])

    # a node may be any point in the interval it represents, sample_discrete returns the same
    # index for all of them. The midpoint is used rather than the endpoint so each node keeps a
    # wide margin against rounding in the transition matrix, instead of sitting exactly on a
    # boundary where one unit in the last place would move a whole state's probability mass.
    lower = vcat(0.0, breakpoints[1:(end-1)])
    nodes = reshape((lower .+ breakpoints) ./ 2, 1, length(breakpoints))

    return RandomVariable(nodes, weights ./ sum(weights))
end
