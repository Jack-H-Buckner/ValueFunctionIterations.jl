"""
    DynamicProgram

Stores the data required to define a dynamic programming problem
along with the value V and policy functions P.
...
# Elements:
    - R: the reward function, takes the form R(s,u,X,p) where s is the state and u is the decision variable, X is random varible inputs and p are paramters
    - F: the state update function, takes the form F(s,u,X,p).
    - p: the state update parameters, ComponentArray it must be compatable with both R and F. 
    - u: the decision variables, a matrix where each column give a possibel valueof the decision variable u. 
    - X: the random variables, an AbstractRandomVariable object.
    - δ: the discount factor, Float64.
    - V: the value function, a AbstractValueFunction object.
    - P: the policy function, a AbstractValueFunction object.
"""
mutable struct DynamicProgram{action_type}
    R::Function # rewards 
    F::Function # state update
    p::ComponentArray # state update paramters  
    u::action_type # decision variables
    X::AbstractRandomVariable
    δ::Float64 # discount factor
    V::AbstractValueFunction # value function
    P::ValueFunctionList # policy function
end


function value_function_iteration!(
    V::AbstractValueFunction,
    R::Function, # rewards 
    F::Function, # state update
    p::ComponentArray, # state update paramters
    u,# decision variables matrix or function 
    X::AbstractRandomVariable,
    δ::Float64; # discount factor
    tolerance = 1e-5,
    maxiter = round(Int, 3/(1-δ)))

    converged = false
    prog = Progress(maxiter, 1)
    while !converged && maxiter > 0
        maxiter -= 1
        vals = bellman(V.states,u,X,R,F,p,δ,V)
        if mean(abs.(vals.-V.values)) .<tolerance 
            converged = true 
        end
        update!(V,vals)
        ProgressMeter.next!(prog)
    end

    return V, converged
end 


function interpolate_policy(
    R::Function, # rewards 
    F::Function, # state update
    p::ComponentArray, # state pdate paramters
    u, # decision variables matrix or function 
    X::AbstractRandomVariable,
    δ::Float64, # discount factor
    V::RegularGridBspline;
    order = BSpline(Constant()))

    u_opt = policy(V.states,u,X,R,F,p,δ,V)
    splines = [RegularGridBspline(V,u_opt[i,:], order = order) for i in 1:size(u_opt)[1]]

    return ValueFunctionList{RegularGridBspline}(reshape(splines,size(u_opt)[1]))
end 


function interpolate_policy(
    R::Function, # rewards 
    F::Function, # state update
    p::ComponentArray, # state pdate paramters
    u, # decision variables matrix or function 
    X::AbstractRandomVariable,
    δ::Float64, # discount factor
    V::DiscreteAndContinuous;
    order = BSpline(Constant()))

    u_opt = policy(V.states,u,X,R,F,p,δ,V)
    splines = [DiscreteAndContinuous(V,u_opt[i,:],order = order) for i in 1:size(u_opt)[1]]

    return ValueFunctionList(reshape(splines,size(u_opt)[1]))
end 


function number_of_actions(u,s,p)
    if typeof(u) <: Matrix{Float64}
        return size(u)[1]
    else 
        return size(u(s,p))[1]
    end
end 


function check_random_variable(X,s,p)
    if typeof(X) <: RandomVariableFunction
        return abs.(sum(X(s,p).weights) - 1) < 1e-3
    else 
        return abs.(sum(X.weights) - 1) < 1e-3
    end
end 
"""
    DynamicProgram(R::Function, F::Function,  p::ComponentArray, u::Matrix{Float64}, X::AbstractRandomVariable, δ::Float64, grid...; kwrds...  )

Solves a continuous state, discrete action dynamic optimization problem using value function iteration and returns 
the solution as a DynamicProgram object. 
...
# Arguments:
    - V: the value function, a AbstractValueFunction object
    - R: the reward function, takes the form R(s,u,X,p) where s is the state and u is the decision variable, X is random varible inputs and p are paramters
    - F: the state update function, takes the form F(s,u,X,p). 
    - p: the state update parameters, ComponentArray it must be compatable with both R and F. 
    - u: the decision variables, a matrix where each column give a possibel valueof the decision variable u. 
    - X: the random variables, an AbstractRandomVariable object.
    - δ: the discount factor, Float64. 

# Keyword arguments:
    - solve: whether to solve the problem, defaults to "conditional". The problem will not solve if the estiamted time is larger than ten minutes, this can be over ridden by setting solve = true to always solve or solvefalse to never solve. `.
    - order_policy: the order of the interpolation for the policyfunction, defaults to Constant()
    - extrap: the value to use for extrapolation, defaults to Float()
    - tolerance: the tolerance for VFI convergence, defaults to 1e-5
    - maxiter: the maximum number of iterations, defaults to round(Int, 3/(1-δ))

# Values:
    - a DynamicProgram object
"""
function DynamicProgram(
    V::AbstractValueFunction,
    R::Function, 
    F::Function, 
    p::ComponentArray, 
    u, # matrix or function
    X::AbstractRandomVariable,
    δ::Float64;
    solve = "conditional",
    order_policy = BSpline(Constant()),
    extrap = Flat(),
    tolerance = 1e-5,
    maxiter = round(Int, 3/(1-δ)))

    # get and instance of the state variables to use in checks
    s = V.states[:,1] 

    # check discount factor and random variable
    if !(δ<1)
        error("The discount factor delta must be less than one.")
    end
    if !(check_random_variable(X,s,p))
        print("The wieght in the random variable do not add to one, there might be an issue with your integration method.")
    end

    P = ValueFunctionList([V for i in 1:number_of_actions(u,s,p)])
    DP = 0
    if typeof(u)<:Function
        DP = DynamicProgram{Function}(R,F,p,u,X,δ,V,P)
    else
        DP = DynamicProgram{Matrix{Float64}}(R,F,p,u,X,δ,V,P)
    end

    if solve == true
        solve!(DP;order_policy=order_policy,tolerance=tolerance,maxiter=maxiter)
    elseif solve == "conditional"
        if estimate_time(DP)[1][2] > 600
            print("Estimated time to solve is greater than 10 minutes. \nUse the estimate_time function to check for perfornace bottelnecks or solve with the solve! function.")
        else
            solve!(DP;order_policy=order_policy,tolerance=tolerance,maxiter=maxiter)
        end
    end
    
    return DP
end


"""
    estimate_time(DP::DynamicProgram)

Estimate how long a dynamic program will take to solve.
...
# Arguments:
    - DP: a DynamicProgram object
# Value 
    - a dictionary with keys "Estimate", "One call", "Number of computations", "Estimated iterations"
"""
function estimate_time(DP::DynamicProgram{Matrix{Float64}})
    threads= Threads.nthreads()
    iter = round(Int, 3/(1-DP.δ)) # estimated number of iterations

    # get an instance of the random variable 
    Xt = DP.X
    if typeof(DP.X) <: RandomVariableFunction
        Xt = DP.X(DP.V.states[:,1],DP.p)
    end 

    ncalculations = length(Xt.weights)*size(DP.u)[2]*size(DP.V.states)[2] # number of calcualteion per iteration
    # define one update evluation
    f() = DP.V(DP.F(DP.V.states[:,1],DP.u[:,1],Xt.nodes[:,1],DP.p)...)+DP.R(DP.V.states[:,1],DP.u[:,1],Xt.nodes[:,1],DP.p)
    f() # run once to deal with compilation
    t1 = time() 
    for _ in 1:1000 f() end
    elapsed_time = (time() - t1)/1000
    estimate = elapsed_time*ncalculations*iter/threads
    return ("Estimate (sec)" => round(estimate, digits = 2), 
            "One call" => elapsed_time,  
            "Number of computations" => ncalculations, 
            "actions" => size(DP.u)[2],
            "states" => size(DP.V.states)[2],
            "random variable samples" => length(Xt.weights),
            "Estimated iterations" => iter,
            "Number of threads" => threads)
end 


function estimate_time(DP::DynamicProgram{Function})
    threads= Threads.nthreads()
    iter = round(Int, 3/(1-DP.δ)) # estimated number of iterations

    # get an instance of the random variable 
    Xt = DP.X
    if typeof(DP.X) <: RandomVariableFunction
        Xt = DP.X(DP.V.states[:,1],DP.p)
    end 

    ncalculations = 0
    for i in 1:size(DP.V.states)[2]
        si = DP.V.states[:,i]
        ncalculations += size(DP.u(si,DP.p))[2]
    end 
    ncalculations = length(Xt.weights)*ncalculations # number of calcualteion per iteration
    # define one update evluation
    u1 = DP.u(DP.V.states[:,1],DP.p)
    f() = DP.V(DP.F(DP.V.states[:,1],u1,Xt.nodes[:,1],DP.p))+DP.R(DP.V.states[:,1],u1,Xt.nodes[:,1],DP.p)
    f() # run once to deal with compilation
    t1 = time() 
    for _ in 1:1000 f() end
    elapsed_time = (time() - t1)/1000
    estimate = elapsed_time*ncalculations*iter/threads
    return ("Estimate (sec)" => round(estimate, digits = 2), 
            "One call" => elapsed_time,  
            "Number of computations" => ncalculations, 
            "actions" => "varaible",
            "states" => size(DP.V.states)[2],
            "random variable samples" => length(Xt.weights),
            "Estimated iterations" => iter,
            "Number of threads" => threads)
end 


"""
    solve!(DP::DynamicProgram; kwrds...) 

Rus the VFI algorithm to solve the dynamic program `DP` and updates the value and policy fuctions in place
...
# Arguments:
    - DP: a DynamicProgram object

# Key words:
    - order_policy: the order of the interpolation for the policyfunction, defaults to Constant()
    - tolerance: the tolerance for VFI convergence, defaults to 1e-5
    - maxiter: the maximum number of iterations, defaults to round(Int, 3/(1-δ))
"""
function solve!(DP::DynamicProgram; order_policy = BSpline(Constant()),tolerance = 1e-5,maxiter = round(Int, 3/(1-DP.δ))) 
    value_function_iteration!(DP.V, DP.R,DP.F,DP.p,DP.u,DP.X,DP.δ;tolerance = tolerance,maxiter = maxiter)
    P = interpolate_policy(DP.R,DP.F,DP.p,DP.u,DP.X,DP.δ,DP.V,order=order_policy)
    DP.P = P
    return nothing
end

"""
    save_solution(DP,filename)

Saves the value and policy functions of the dynamic program `DP` to the file `filename`.
"""
function save_solution(DP,filename)
    Vfun = DP.V; Pfun = DP.P
    @save  filename Vfun Pfun
end 

"""
    load_solution(DP,filename)

Load the value and policy functions at `filename`, adn over write the policy and 
value functions in the dynamic program `DP`.
"""
function load_solution(DP,filename)
    @load filename Vfun Pfun
    DP.V = Vfun; DP.P = Pfun
end 