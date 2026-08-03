###################################################################
###################################################################
## Gausian quadrature for normal distributions
## This file uses teh FastGaussQuadrature.jl package to generate
## weights and nodes to integrate multivariate normal distributions
## with any mean covariance matrix.
## The main function GaussHermiteRandomVariable returns the nodes of the
## quadrature scheme in a matrix with each column representign the
## coordinates of a points in the multivariate normal distribution.
## the weights are returned in a vector with each entry corresponding
## to a column of the nodes matrix.
## The probabilists' normalization is used (nodes -> sqrt(2)*x and
## weights -> w/sqrt(pi)) so the weights sum to one and the scheme is
## exact for polynomials up to degree 2m-1.
## Author: Jack H. Buckner
## Date: 9/2025
## Oregon State University
###################################################################
###################################################################

function nodes_grid(nodes, weights)

    # initialize nodes and weights matrices
    Nodes = zeros(length(nodes),prod(length.(nodes)))
    Weights = zeros(prod(length.(nodes)))

    # convert nodes into a matrix 
    i = 0
    for node in Iterators.product(nodes...)
        i+=1
        Nodes[:,i] .= node
    end

    # the weight of each grid point is the product of the weights
    # of the one dimensional nodes it is built from
    i = 0
    for w in Iterators.product(weights...)
        i+=1
        Weights[i] = prod(w)
    end

    return Nodes, Weights
end 

 

"""
    GaussHermiteRandomVariable(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64})

Returns an RandomVariable with weights and nodes for a multivariate normal distribution
with covariance matrix Cov and mean vector mu. The weights and nodes are chosen using a guass hermite
quadrature scheme with m nodes per dimension. The weights sum to one and the scheme integrates
polynomials up to degree 2m-1 exactly.
"""
function GaussHermiteRandomVariable(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64})
    dims = size(Cov)[1]

    # probabilists' normalization: nodes -> sqrt(2)*x and weights -> w/sqrt(pi)
    # so the nodes and weights integrate a standard normal distribution
    nodes, weights = FastGaussQuadrature.gausshermite(m, normalize = true)
    standardNodes, weights = nodes_grid((nodes for i in 1:dims),  (weights for i in 1:dims))

    # spectral decomposition, S*rV is a square root of Cov
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,dims,dims).*estuff.values)
    S = real.(estuff.vectors)

    # transform the standard normal nodes to have mean mu and covariance Cov
    nodes = mapslices(x -> S*rV*x.+mu, standardNodes, dims = 1)

    return RandomVariable(nodes, weights)
end 


function RandomVariable(m::Int,mu::Vector{Float64},Cov::Matrix{Float64})
    return GaussHermiteRandomVariable(m,mu,Cov)
end

