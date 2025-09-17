###################################################################
###################################################################
## Compute the bellman equaitons for the fisheirs insurance model.
## The operators will expect four state variables s:
## 1. savings
## 2. log stock
## 3. change in log stock
## 4. the previous mortaltity rate
## The operators will require a function that updates the state variables
## given the current states, decision variables and random varibles to be 
## passed. 
##
## The operators expect two decision varaibles u:
## 1. insurance premium
## 2. change in savings
## The operators defined here will not specify the values of these
## two variables to test. Instead they will compare the perforamce of 
## options passed in a matrix whre each column represents a decision. 
##
## The code allows as many factors X as desiered to be random as desiered
## The random varaibles are integrated over usign quadrature 
## define else where, these functions require, a matrix of values
## for each random variable and weights for column of the matrix.
###################################################################
###################################################################

function Q(s::Vector{Float64}, # state variables 
            u::Vector{Float64},# decision variables
            X::RandomVariable,
            R::Function, # rewards 
            F::Function, # state update
            p::ComponentArray, # state update paramters
            δ::Float64, # discount factor
            V::AbstractValueFunction) # value funciton
    w = X.weights
    Xt = X.nodes
    rt = broadcast(i->R(s,u,Xt[:,i],p), 1:size(X.nodes)[2]) # rewards depend on the state and decision variables
    Vt = broadcast(i->V(F(s,u,Xt[:,i],p)), 1:size(X.nodes)[2])
    Vt = sum(w.*Vt)
    return sum(w.*rt) + δ*Vt
end

function Q(s::Vector{Float64}, # state variables 
            u::Vector{Float64},# decision variables
            X::MCRandomVariable,
            R::Function, # rewards 
            F::Function, # state update
            p::ComponentArray, # state update paramters
            δ::Float64, # discount factor
            V::AbstractValueFunction) # value funciton
    X() # resample the random variables
    w = X.weights
    Xt = X.nodes
    rt = broadcast(i->R(s,u,Xt[:,i],p), 1:size(X.nodes)[2]) # rewards depend on the state and decision variables
    Vt = broadcast(i->V(F(s,u,Xt[:,i],p)), 1:size(X.nodes)[2])
    Vt = sum(w.*Vt)
    return sum(w.*rt) + δ*Vt
end

function ENPV(s::Vector{Float64}, # state variables 
            u::Matrix{Float64},# decision variables
            X::AbstractRandomVariable,
            R::Function, # rewards 
            F::Function, # state update
            p::ComponentArray, # state pdate paramters
            δ::Float64, # discount factor
            V::AbstractValueFunction) # value funciton

    Qs = broadcast(i->Q(s,u[:,i],X,R,F,p,δ,V),1:size(u)[2])
    ind = argmax(Qs)
    ENPV = Qs[ind]
    u_opt = u[:,ind]
    return ENPV, u_opt
end

function bellman(s::Matrix{Float64}, # state variables 
            u::Matrix{Float64},# decision variables
            X::AbstractRandomVariable,
            R::Function, # rewards 
            F::Function, # state update
            p::ComponentArray, # state pdate paramters
            δ::Float64, # discount factor
            V::AbstractValueFunction) # value funciton

    Vals = zeros(size(s)[2])
    Threads.@threads for i in 1:size(s)[2]
        Vals[i]=ENPV(s[:,i],u,X,R,F,p,δ,V)[1]
    end
    return Vals
end 


function policy(s::Matrix{Float64}, # state variables 
            u::Matrix{Float64},# decision variables
            X::AbstractRandomVariable,
            R::Function, # rewards 
            F::Function, # state update
            p::ComponentArray, # state pdate paramters
            δ::Float64, # discount factor
            V::AbstractValueFunction) # value funciton

    u_opt = zeros(size(u)[1],size(s)[2])
    Threads.@threads for i in 1:size(s)[2]
        u_opt[:,i]=ENPV(s[:,i],u,X,R,F,p,δ,V)[2]
    end
    return u_opt
end 

