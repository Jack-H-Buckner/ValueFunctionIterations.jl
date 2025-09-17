
"""
    action_space_product(U...)

Returns the cartesian product of the action spaces U defined at abstract rance objects.
...
# Arguments
    - U: two or more AbstractRange objects
# Values
    - a matrix with the cartesian product of the action spaces U.
"""
function action_space(U...)
    dim_size = [length(d) for d in U]
    actions = zeros(length(dim_size),prod(dim_size))
    i = 0
    for u in Iterators.product(U...)
        i+=1
        actions[:,i] .= u
    end
    return actions
end