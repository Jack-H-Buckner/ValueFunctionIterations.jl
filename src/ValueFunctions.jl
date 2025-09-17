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

function RegularGridBspline(dims...;v0 = 0.0, order = Bspline(Cubic(Line(OnGrid()))), extrap = Flat() )
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

mutable struct RegularGridBsplineList <: AbstractValueFunction
    Bslines::AbstractVector{RegularGridBspline}
end

function (x::RegularGridBsplineList)(state)
    return [spline.interpolation(state...) for spline in x.Bslines]  
end 