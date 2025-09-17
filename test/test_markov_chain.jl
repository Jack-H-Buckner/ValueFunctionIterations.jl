include("../src/MarkovChain.jl")

using StatsBase, Random 
function main()
    Random.seed!(123)
    p = [0.59 0.4 0.2; 0.39 0.55 0.5; 0.02 0.05 0.3]
    MCSampler = MarkovChain(p)
    x = 1; xvals = []
    pmc = zeros(3,3)
    nmc = 10^6
    for i in 1:nmc
        rng = StatsBase.sample(MCSampler.nodes, Weights(MCSampler.weights))
        xnew = sample_markov_chain(round(Int,x),p,rng)
        if i > 100
            pmc[x,xnew] += 1
        end
        x = xnew
    end

    sum(abs.(pmc./sum(pmc,dims = 1) .- p)) < 0.02
end
main()