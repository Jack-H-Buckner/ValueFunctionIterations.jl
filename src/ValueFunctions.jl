###################################################################
###################################################################
## Value function aproximators 
## The value function approximators defined here are meant to play 
## nicely with the operators defined in bellman.jl
## they are defined as functors which can be evaluated like functions
## but also store data to use in the funciton evaluation
## in this case the function returns the estiamte value of the value 
### function using an interpolation and the data structure
## stores the grid points and values used in the interpolation
###################################################################
###################################################################
abstract type AbstractValueFunction end
mutable struct RegularGridBspline{N} <: AbstractValueFunction
    values::Vector{Float64}
    states::Matrix{Float64} 
    grid::NTuple{N,StepRangeLen}
    dims#::AbstractVector{Int}
    interpolation
    order
    extrap
end

"""
    RegularGridBspline(N::Int,grids; kwargs...)

Defines a value function that combines a discrete and a continuous state variables
...
# Arguments
- dims...: a regular grid for each state variable. 

# Key words:
 - v0: Initial value, defaults to 0.
 - order: order of BSpline approximation, defaults to BSpline(Cubic(Line(OnGrid())))
 - extrap: extrapolation method, defaults to Flat(). 
"""
function RegularGridBspline(dims...;v0 = 0.0, order = BSpline(Cubic(Line(OnGrid()))), extrap = Flat() )
    dim_size = [length(d) for d in dims]
    states = zeros(length(dim_size),prod(dim_size))
    values = zeros(prod(dim_size))
    i=0
    for s in Iterators.product(dims...)
        i+=1
        states[:,i] .= s
        values[i] = v0
    end
    N = length(dim_size)
    dim_size_tuple=NTuple{N, Int}(dim_size)
    values_interp = reshape(values,dim_size_tuple)
    interp=interpolate(values_interp,order)
    interp=Interpolations.scale(interp,dims...)
    interp= extrapolate(interp, extrap)
    return RegularGridBspline{N}(values,states,dims,dim_size,interp,order,extrap)
end


function RegularGridBspline(to_copy::RegularGridBspline,values; order = to_copy.order, extrap = to_copy.extrap)
    dim_size = to_copy.dims
    N = length(dim_size)
    dim_size=NTuple{N, Int}(dim_size)
    values_interp = reshape(values,dim_size)
    interp=interpolate(values_interp,order)
    interp=Interpolations.scale(interp,to_copy.grid...)
    interp= extrapolate(interp, extrap)
    return RegularGridBspline{N}(values,to_copy.states,to_copy.grid,to_copy.dims,interp,order,extrap)
end


function (x::RegularGridBspline)(state)
    return x.interpolation(state...)
end 

function update!(x::RegularGridBspline, values)
    interp=interpolate(reshape(values,x.dims...),x.order)
    interp=Interpolations.scale(interp,x.grid...)
    interp= extrapolate(interp,x.extrap)
    x.values = reshape(values,prod(x.dims))
    x.interpolation = interp
end 

mutable struct ValueFunctionList{T<:AbstractValueFunction}
    V::AbstractVector{T}
end

function (x::ValueFunctionList)(state)
    return [v(state) for v in x.V]  
end 


mutable struct DiscreteAndContinuous <: AbstractValueFunction
    N::Int
    Bslines::AbstractVector{RegularGridBspline}
    states
    values
end

"""
    DiscreteAndContinuous(N::Int,grids; kwargs...)

Defines a value function that combines a discrete and a continuous state variables
...
# Arguments
- `N::Int`: number of discrete states
- `grids: a vector with an element for level fo the discrete state that is its self a vector with the grids for the continuous state

# Key words:
 - v0: Initial value, defaults to 0.
 - order: order of BSpline approximation, defaults to BSpline(Cubic(Line(OnGrid())))
 - extrap: extrapolation method, defaults to Flat(). 
"""
function DiscreteAndContinuous(N::Int,grids;v0 = 0.0, order = BSpline(Cubic(Line(OnGrid()))), extrap = Flat() )
    Vfunctions = []
    states = zeros(1+length(grids[1]),1)
    for i in 1:N
        Vi = RegularGridBspline(grids[i]...;v0=v0,order=order,extrap=extrap)
        Nstates = size(Vi.states)[2]
        discrete = zeros(1,Nstates) .+ i
        new_states = vcat(discrete, Vi.states)
        states = hcat(states,new_states )
        push!(Vfunctions,Vi)
    end
    states = states[:,2:end]
    values = zeros(size(states)[2])
    return DiscreteAndContinuous(N,Vfunctions,states,values)
end


function DiscreteAndContinuous(to_copy::DiscreteAndContinuous,values::AbstractArray{Float64};v0 = 0.0, order = Bspline(Cubic(Line(OnGrid()))), extrap = Flat() )
    Vfunctions = []
    for i in 1:to_copy.N
        push!(Vfunctions,RegularGridBspline(to_copy.Bslines[i],values[to_copy.states[1,:].==i];order=order,extrap=extrap))
    end
    return DiscreteAndContinuous(to_copy.N,Vfunctions,to_copy.states,to_copy.values)
end


function (x::DiscreteAndContinuous)(state)
    return x.Bslines[round(Int,state[1])](state[2:end]...)
end 


function update!(x::DiscreteAndContinuous, values)
    for i in 1:x.N
        update!(x.Bslines[i],values[x.states[1,:].==i])
    end
    return nothing
end 

