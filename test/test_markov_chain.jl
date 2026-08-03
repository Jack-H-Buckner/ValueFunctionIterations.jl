

# reconstructs the transition matrix that the (nodes, weights) rule actually
# reproduces, so the quadrature scheme can be compared to p directly
function implied_transition_matrix(X,p)
    q = zeros(size(p))
    for x in axes(p,2), k in eachindex(X.weights)
        q[sample_markov_chain(x,p,X.nodes[1,k]),x] += X.weights[k]
    end
    return q
end

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

    if !(sum(abs.(pmc./sum(pmc,dims = 1) .- p)) < 0.02)
        throw("fails to reproduce markov chain")
    end
end
main()


# the nodes and weights reproduce the transition matrix exactly, and keep doing
# so when the columns of the matrix are only approximately normalized
function test_transition_matrix()

    p = [0.59 0.4 0.2; 0.39 0.55 0.5; 0.02 0.05 0.3]
    X = MarkovChain(p)
    @test sum(X.weights) ≈ 1.0
    @test maximum(abs.(implied_transition_matrix(X,p) .- p)) < 1e-14

    # columns that are off by a small amount of rounding describe the
    # normalized matrix, and do so to full precision
    perturbed = copy(p)
    perturbed[:,1] .*= 1 + 1e-9
    perturbed[:,3] .*= 1 - 3e-9
    X = MarkovChain(perturbed)
    @test sum(X.weights) ≈ 1.0
    @test maximum(abs.(implied_transition_matrix(X,perturbed) .-
                       perturbed ./ sum(perturbed,dims = 1))) < 1e-14

    # columns that cannot sum to one in binary
    thirds = [1/3 0.1 0.7; 1/3 0.3 0.2; 1/3 0.6 0.1]
    X = MarkovChain(thirds)
    @test sum(X.weights) ≈ 1.0
    @test maximum(abs.(implied_transition_matrix(X,thirds) .- thirds)) < 1e-14
end
test_transition_matrix()


# each node sits inside the interval it represents rather than on its boundary,
# so rounding in the transition matrix cannot move a node onto the wrong state
function test_node_placement()
    for p in ([0.59 0.4 0.2; 0.39 0.55 0.5; 0.02 0.05 0.3],
              [1/3 0.1 0.7; 1/3 0.3 0.2; 1/3 0.6 0.1],
              [0.5 0.0; 0.5 1.0])
        X = MarkovChain(p)
        for k in eachindex(X.weights), x in axes(p,2)
            node = X.nodes[1,k]
            state = sample_markov_chain(x,p,node)
            @test sample_markov_chain(x,p,nextfloat(node)) == state
            @test sample_markov_chain(x,p,prevfloat(node)) == state
        end
    end
end
test_node_placement()


# input that cannot be explained by rounding is an error rather than a silently
# incorrect set of weights
function test_invalid_transition_matrix()
    p = [0.59 0.4 0.2; 0.39 0.55 0.5; 0.02 0.05 0.3]

    # a column that sums to one half
    halved = copy(p); halved[:,2] .*= 0.5
    @test_throws ArgumentError MarkovChain(halved)

    # a negative entry, the column still sums to one
    negative = [0.6 0.4 0.2; -0.1 0.55 0.5; 0.5 0.05 0.3]
    @test_throws ArgumentError MarkovChain(negative)

    # a non finite entry
    infinite = copy(p); infinite[1,1] = Inf
    @test_throws ArgumentError MarkovChain(infinite)

    # rounding sized errors are accepted
    rounded = copy(p); rounded[:,1] .*= 1 + 1e-9
    @test sum(MarkovChain(rounded).weights) ≈ 1.0
end
test_invalid_transition_matrix()
