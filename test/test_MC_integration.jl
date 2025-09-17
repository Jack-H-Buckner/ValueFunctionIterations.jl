Random.seed!(123)

f(x) = sum(x)-sin(0.5*prod(x))

function main()
   Random.seed!(123)
    Nmc = 10^6
    # define discrete and normal random variables
    Mt = RandomVariable([1 2], [0.75 0.25])
    m = 10; mu = [0.0,0]; Cov = [1 0.5; 0.5 1]
    sample() = rand(Distributions.MvNormal(mu,Cov)) 
    Xt = MCRandomVariable(sample, Mt, Nmc); Xt()
    EV_quad = sum(broadcast(i -> f(Xt.nodes[:,i]), 1:size(Xt.nodes)[2]) .* Xt.weights)

    
    # solve integral using monte carlo
    m = rand(Distributions.MvNormal(mu,Cov),Nmc)
    fmc0 = broadcast(i -> f(vcat(m[:,i],1)), 1:size(m)[2])
    fmc1 = broadcast(i -> f(vcat(m[:,i],2)), 1:size(m)[2])
    EV_mc = 0.75*sum(fmc0)/Nmc + 0.25*sum(fmc1)/Nmc

    # compare results
    if !(abs(EV_mc-EV_quad) < 3*sqrt(var(fmc0)+var(fmc1))/sqrt(2*Nmc))
        throw("fails to calcualte integral with montecarlo")
    end 
end

main()