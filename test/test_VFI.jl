
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

    fvals = 0:0.02:0.98
    u = reshape(collect(fvals),1,length(fvals))
    δ = 0.999
    grid = 0.01:0.025:1.25 
    V = ValueFunctionIterations.RegularGridBspline(grid)

    # solve with quadrature and test solution
    X = ValueFunctionIterations.GaussHermiteRandomVariable(10,[0.0],[1.0;;])
    
    sol_quad = ValueFunctionIterations.DynamicProgram(V, R, F, p, u,  X, δ; tolerance = 1e-5, maxiter = 400)
    test_quad = sum((theoretical.(grid,BMSY_) .- broadcast(s -> s - s*sol_quad.P(s)[1],grid)).^2) < 1e-2
    println(test_quad)
    # solve with MC integration and test solution
    Random.seed!(123)
    sample() = rand(Distributions.Normal(0,1.0),1,1)
    X = ValueFunctionIterations.MCRandomVariable(sample, 100)
    V = ValueFunctionIterations.RegularGridBspline(grid)
    sol_MC = ValueFunctionIterations.DynamicProgram(V,R, F, p, u,  X, δ; solve = true, tolerance = 1e-5, maxiter = 400)
    test_MC = sum((theoretical.(grid,BMSY_) .- broadcast(s -> s - s*sol_MC.P(s)[1],grid)).^2) < 5*1e-2
    
    if !(test_quad && test_MC)
        throw("failed to solve VFI")
    end

    # update functions for a fishery model with known solution 
    R2(s,u,X,p) = u[1]
    F2(s,u,X,p) = (s[1] - u[1])*exp(p.r*(1-(s[1]-u[1])/p.k) + p.σ*X[1]-0.5*p.σ^2)
    p = ComponentArray(r = 0.2, k = 1.0, σ = 0.001)
    BMSY_ = BMSY(p)

    u_(s,p) = reshape(collect(0.0:0.02:s[1]),1,length(0.0:0.02:s[1])) 
    δ = 0.999
    grid = 0.01:0.025:1.25
    V2 = ValueFunctionIterations.RegularGridBspline(grid)
    

    # solve with quadrature and test solution
    X = ValueFunctionIterations.GaussHermiteRandomVariable(10,[0.0],[1.0;;])
    sol_quad = ValueFunctionIterations.DynamicProgram(V2,R2,F2, p, u_,  X, δ; tolerance = 1e-5, maxiter = 400)
    test_quad = sum((theoretical.(grid,BMSY_) .- broadcast(s -> s - sol_quad.P(s)[1],grid)).^2) < 1e-2

    if !(test_quad)
        throw("failed to solve VFI with u as a function")
    end
end 
main()