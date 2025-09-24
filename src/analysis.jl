"""
    simulate(DP::DynamicProgram,T::Int)

Simulates the dynamic program `DP` under the optimal policy for `T` timesteps.
The function returns the states in a matrix of size dims(s) by T+1, 
action in a matrix of size dim(u) by T,reward in a vector of size T, 
and value in a vector of size T+1.
...
# Arguments:
    - DP: a DynamicProgram object
    - T: the number of timesteps to simulate

# Values:
    - the states in a matrix of size dims(s) by T+1
    - the actions in a matrix of size dim(u) by T  
    - the rewards in a vector of size T
    - the values in a vector of size T+1
"""
function simulate(DP::DynamicProgram,T::Int;with_policy = false)
    if with_policy
        return simulate_policy(DP,T)
    end
    return simulate_solve(DP,T)
end 

function simulate_policy(DP::DynamicProgram{Matrix{Float64}},T::Int)
    states = zeros(size(DP.V.states)[1],T+1)
    states[:,1] .= DP.V.states[:,rand(1:size(DP.V.states)[2])]
    values = zeros(T+1)
    values[1] = DP.V(states[:,1])
    actions = zeros(size(DP.u)[1],T)
    rewards = zeros(T)
    Xt = DP.X
    if typeof(DP.X) <: RandomVariableFunction
        Xt = DP.X(states[:,1],DP.p)
    end 
    randos = zeros(size(Xt.nodes)[1],T)
    
    for t in 1:T
        if typeof(DP.X) <: RandomVariableFunction
            Xt = DP.X(states[:,t],DP.p)
        end 
        ind = sample(eachindex(Xt.weights),Weights(Xt.weights))
        randos[:,t] .= Xt.nodes[:,ind]
        actions[:,t] .= DP.P(states[:,t])
        rewards[t] = DP.R(states[:,t],actions[:,t],randos[:,t],DP.p)
        states[:,t+1] .= DP.F(states[:,t],actions[:,t],randos[:,t],DP.p)
        values[t+1] = DP.V(states[:,t+1])
    end
    return states,actions,rewards,values,randos
end 


function simulate_solve(DP::DynamicProgram{Matrix{Float64}},T::Int)
    states = zeros(size(DP.V.states)[1],T+1)
    states[:,1] .= DP.V.states[:,rand(1:size(DP.V.states)[2])]
    values = zeros(T)
    actions = zeros(size(DP.u)[1],T)
    rewards = zeros(T)
    Xt = DP.X
    if typeof(DP.X) <: RandomVariableFunction
        Xt = DP.X(states[:,1],DP.p)
    end 
    randos = zeros(size(Xt.nodes)[1],T)
    
    for t in 1:T
        if typeof(DP.X) <: RandomVariableFunction
            Xt = DP.X(states[:,t],DP.p)
        end 
        V,u = ENPV(states[:,t], DP.u, Xt, DP.R, DP.F, DP.p, DP.δ, DP.V)
        values[t] = V; actions[:,t] .= u
        ind = sample(eachindex(Xt.weights),Weights(Xt.weights))
        randos[:,t] .= Xt.nodes[:,ind]
        rewards[t] = DP.R(states[:,t],actions[:,t],randos[:,t],DP.p)
        states[:,t+1] .= DP.F(states[:,t],actions[:,t],randos[:,t],DP.p)
    end
    return states,actions,rewards,values,randos
end 

function simulate_policy(DP::DynamicProgram{Function},T::Int)
    states = zeros(size(DP.V.states)[1],T+1)
    states[:,1] .= DP.V.states[:,rand(1:size(DP.V.states)[2])]
    values = zeros(T+1)
    values[1] = DP.V(states[:,1])
    actions = zeros(size(DP.u(states[:,1],DP.p))[1],T)
    rewards = zeros(T)
    Xt = DP.X
    if typeof(DP.X) <: RandomVariableFunction
        Xt = DP.X(states[:,1],DP.p)
    end 
    randos = zeros(size(Xt.nodes)[1],T)
    
    for t in 1:T
        if typeof(DP.X) <: RandomVariableFunction
            Xt = DP.X(states[:,t],DP.p)
        end  
        ind = sample(eachindex(Xt.weights),Weights(Xt.weights))
        randos[:,t] .= Xt.nodes[:,ind]
        actions[:,t] .= DP.P(states[:,t])
        rewards[t] = DP.R(states[:,t],actions[:,t],randos[:,t],DP.p)
        states[:,t+1] .= DP.F(states[:,t],actions[:,t],randos[:,t],DP.p)
        values[t+1] = DP.V(states[:,t+1])
    end
    return states,actions,rewards,values,randos
end 


function simulate_solve(DP::DynamicProgram{Function},T::Int)
    states = zeros(size(DP.V.states)[1],T+1)
    states[:,1] .= DP.V.states[:,rand(1:size(DP.V.states)[2])]
    values = zeros(T)
    actions = zeros(size(DP.u(states[:,1],DP.p))[1],T)
    rewards = zeros(T)
    Xt = DP.X
    if typeof(DP.X) <: RandomVariableFunction
        Xt = DP.X(states[:,1],DP.p)
    end 
    randos = zeros(size(Xt.nodes)[1],T)
    
    for t in 1:T
        if typeof(DP.X) <: RandomVariableFunction
            Xt = DP.X(states[:,t],DP.p)
        end 
        V,u = ENPV(states[:,t], DP.u(states[:,t],DP.p), Xt, DP.R, DP.F, DP.p, DP.δ, DP.V)
        values[t] = V; actions[:,t] .= u
        ind = sample(eachindex(Xt.weights),Weights(Xt.weights))
        randos[:,t] .= Xt.nodes[:,ind]
        rewards[t] = DP.R(states[:,t],actions[:,t],randos[:,t],DP.p)
        states[:,t+1] .= DP.F(states[:,t],actions[:,t],randos[:,t],DP.p)

    end
    return states,actions,rewards,values,randos
end 
"""
    simulate(DP::DynamicProgram,X::MCRandomVariable,T::Int)

Simulates the dynamic program `DP` under the optimal policy for `T` timesteps. 
Sampling the random variables at each timestep using the sampler in X. 
The function returns the states in a matrix of size dims(s) by T+1, 
action in a matrix of size dim(u) by T,reward in a vector of size T, 
and value in a vector of size T+1.
...
# Arguments:
    - DP: a DynamicProgram object
    - X: a MCRandomVariable object
    - T: the number of timesteps to simulate

# Values:
    - the states in a matrix of size dims(s) by T+1
    - the actions in a matrix of size dim(u) by T  
    - the rewards in a vector of size T
    - the values in a vector of size T+1
"""
function simulate(DP::DynamicProgram,X::MCRandomVariable,T::Int)
    states = zeros(length(DP.V.dims),T+1)
    states[:,1] .= DP.V.states[:,rand(1:size(DP.V.states)[2])]
    values = zeros(T+1)
    values[1] = DP.V(states[:,1]...)
    actions = zeros(size(DP.u)[1],T)
    rewards = zeros(T)
    randos = zeros(size(X.nodes)[1],T)
    
    for t in 1:T
        X() # update the random variables
        randos[:,t] .= X.nodes[:,1]
        actions[:,t] .= DP.P(states[:,t]...)
        rewards[t] = DP.R(states[:,t],actions[:,t],randos[:,t],DP.p)
        states[:,t+1] .= DP.F(states[:,t],actions[:,t],randos[:,t],DP.p)
        values[t+1] = DP.V(states[:,t+1]...)
    end
    return states,actions,rewards,values
end 

"""
    get_value_function(DP::DynamicProgram)

Returns the value function of the dynamic program `DP`.
...
# Arguments:
    - DP: a DynamicProgram object

# Values:
    - an AbstractValueFunction object
"""
function get_value_function(DP::DynamicProgram)
    return DP.V
end

"""
    get_value_function(DP::DynamicProgram)

Returns the policy function of the dynamic program `DP`.
...
# Arguments:
    - DP: a DynamicProgram object

# Values:
    - an AbstractValueFunction object
"""
function get_policy_function(DP::DynamicProgram)
    return DP.P
end


"""
    get_value_function(DP::DynamicProgram)

Returns the reward function of the dynamic program `DP`.

Arguments:
    - DP: a DynamicProgram object

Values:
    - a function with arguments (s,u,X,p)
"""
function get_reward_function(DP::DynamicProgram)
    return DP.R
end

"""
    get_value_function(DP::DynamicProgram)

Returns the state update function of the dynamic program `DP`.
...
# Arguments:
    - DP: a DynamicProgram object

# Values:
    - a function with arguments (s,u,X,p)
"""
function get_update_function(DP::DynamicProgram)
    return DP.F
end



"""
    get_value_function(DP::DynamicProgram)

Returns the action space of a dynamic program `DP`.
...
# Arguments:
    - DP: a DynamicProgram object

# Values:
    - a matrix of size dims(u) by N actions
"""
function get_actions_function(DP::DynamicProgram)
    return DP.u
end

"""
    get_value_function(DP::DynamicProgram)

Returns the paramters of the dynamic program `DP`.
...
# Arguments:
    - DP: a DynamicProgram object

# Values:
    - a ComponentVector of model parameters
"""
function get_parameters_function(DP::DynamicProgram)
    return DP.p
end

"""
    get_value_function(DP::DynamicProgram)

Returns the random variable for a the dynamic program `DP`.
...
# Arguments:
    - DP: a DynamicProgram object

# Values:
    - an AbstractRandomVariable object
"""
function get_random_variables_function(DP::DynamicProgram)
    return DP.X
end

"""
    get_value_function(DP::DynamicProgram)

Returns the discount factor for a the dynamic program `DP`.
...
# Arguments:
    - DP: a DynamicProgram object

# Values:
    - a Float object
"""
function get_discount_factor_function(DP::DynamicProgram)
    return DP.δ
end