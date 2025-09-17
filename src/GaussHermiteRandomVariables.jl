###################################################################
###################################################################
## Gausian quadrature for normal distributions 
## This file uses teh FastGaussQuadrature.jl package to generate
## weights and nodes to integrate multivariate normal distributions 
## with any mean covariance matrix. 
## The main function integate_mv_normal return the nodes of the 
## quadrature scheme in a matrix with each column representign the 
## coordinates of a points in the multivariate normal distribution.
## the weights are returned in a vector with each entry corresponding 
## to a column of the nodes matrix. 
## Author: Jack H. Buckner
## Date: 9/2025
## Oregon State University
###################################################################
###################################################################

function planar_rotation(d,theta)
    R = 1.0*Matrix(I,d,d)
    for i in 1:(d-1)
        R_ = 1.0*Matrix(I,d,d)
        R_[i,i] = cos(theta)
        R_[i,i+1] = -sin(theta)
        R_[i+1,i] = sin(theta)
        R_[i+1,i+1] = cos(theta)
        R .= R*R_
    end 
    return R
end 

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

    # if only one dimension, return original weights vector
    if size(Nodes)[2] == 1
        return Nodes, weights
    end

    # if more than one dimension calcualte product of weights 
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
quadrature scheme. 
"""
function GaussHermiteRandomVariable(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64})
    dims = size(Cov)[1]
    nodes, weights = FastGaussQuadrature.gausshermite(m)
    weights = weights .* (2*pi)^(-1/2).*exp.((nodes.^2)./2)
    standardNodes, weights = nodes_grid((nodes for i in 1:dims),  (weights for i in 1:dims))

    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,dims,dims).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    R = planar_rotation(dims,pi/4)
    
    # transform and plot 
    nodes = mapslices(x -> S*rV*R*x.+mu, standardNodes, dims = 1)

    return RandomVariable(nodes, weights)
end 
