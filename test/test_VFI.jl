
function BMSY(p)
    target = x -> -((p.r*x-p.k)*exp(p.r*(1-x/p.k))+p.k)/p.k
    find_zero(target,(0,p.k))
end 

function theoretical(x,BMSY)
    if x < BMSY
        return x
    else
        return BMSY
    end 
end 

function main()
    # update functions for a fishery model with known solution 
    R(s,u,X,p) = s[1]*u[1]
    F(s,u,X,p) = s[1]*(1 - u[1])*exp(p.r*(1-s[1]*(1 - u[1])/p.k) + p.σ*X[1]-0.5*p.σ^2)
    p = ComponentArray(r = 0.2, k = 1.0, σ = 0.001)
    BMSY_ = BMSY(p)

    R(s,u,X,p) = s[1]*u[1]
    F(s,u,X,p) = s[1]*(1 - u[1])*exp(p.r*(1-s[1]*(1 - u[1])/p.k) + p.σ*X[1]-0.5*p.σ^2)
    p = ComponentArray(r = 0.2, k = 1.0, σ = 0.001)
    BMSY_ = BMSY(p)

    fvals = 0:0.01:0.99
    u = reshape(collect(fvals),1,length(fvals))
    δ = 0.999
    grid = 0.01:0.025:1.25
    

    # solve with quadrature and test solution
    X = ValueFunctionIterations.GaussHermiteRandomVariable(10,[0.0],[1.0;;])
    sol_quad = ValueFunctionIterations.DynamicProgram(R, F, p, u,  X, δ, grid; tolerance = 1e-5, maxiter = 400)
    test_quad = sum((theoretical.(grid,BMSY_) .- broadcast(s -> s - s*sol_quad.P(s)[1],grid)).^2) < 1e-3

    # solve with MC integration and test solution
    Random.seed!(123)
    sample() = rand(Distributions.Normal(0,1.0),1,1)
    X = ValueFunctionIterations.MCRandomVariable(sample, 100)
    sol_MC = ValueFunctionIterations.DynamicProgram(R, F, p, u,  X, δ, grid; solve = true, tolerance = 1e-5, maxiter = 400)
    test_MC = sum((theoretical.(grid,BMSY_) .- broadcast(s -> s - s*sol_MC.P(s)[1],grid)).^2) < 2*1e-2

    if !(test_quad && test_MC)
        throw("failed to solve VFI")
    end
    
end 
main()