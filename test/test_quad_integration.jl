Random.seed!(123)

f(x) = sum(x)-sin(0.5*prod(x))

function main()
    Random.seed!(123)

    # define discrete and normal random variables
    Mt = RandomVariable([1 2], [0.75 0.25])
    m = 10; mu = [0.0,0]; Cov = [1 0.5; 0.5 1]
    Rt = GaussHermiteRandomVariable(20,mu,Cov)

    # product of a normal and a discrete random variable
    MRt=product(Rt,Mt)
    EV_quad = sum(broadcast(i -> f(MRt.nodes[:,i]), 1:size(MRt.nodes)[2]) .* MRt.weights)

    Nmc = 10^6
    # solve integral using monte carlo
    m = rand(Distributions.MvNormal(mu,Cov),Nmc)
    fmc0 = broadcast(i -> f(vcat(m[:,i],1)), 1:size(m)[2])
    fmc1 = broadcast(i -> f(vcat(m[:,i],2)), 1:size(m)[2])
    EV_mc = 0.75*sum(fmc0)/Nmc + 0.25*sum(fmc1)/Nmc

    # compare results
    if !(abs(EV_mc-EV_quad) < 3*sqrt(var(fmc0)+var(fmc1))/sqrt(2*Nmc))
        throw("failed to calcualte integral with quadrature")
    end
end

# the gauss hermite scheme with m nodes integrates polynomials of degree
# 2m-1 exactly, these tests check the moments of the quadrature scheme
# against the moments of the normal distribution it represents
function test_moments()

    # univariate standard normal, moments are 0, 1, 0 and 3
    for m in 3:8
        X = GaussHermiteRandomVariable(m,[0.0],[1.0;;])
        x = vec(X.nodes)
        @test sum(X.weights) ≈ 1.0
        @test abs(sum(X.weights .* x)) < 1e-10
        @test sum(X.weights .* x.^2) ≈ 1.0
        @test abs(sum(X.weights .* x.^3)) < 1e-10
        @test sum(X.weights .* x.^4) ≈ 3.0
    end

    # univariate with a non-zero mean and variance
    mu = 2.0; sigma2 = 0.5
    X = GaussHermiteRandomVariable(5,[mu],[sigma2;;])
    x = vec(X.nodes)
    @test sum(X.weights) ≈ 1.0
    @test sum(X.weights .* x) ≈ mu
    @test sum(X.weights .* (x .- mu).^2) ≈ sigma2

    # multivariate with correlated dimensions
    mu = [0.5,-1.0]; Cov = [1 0.5; 0.5 2]
    X = GaussHermiteRandomVariable(5,mu,Cov)
    @test sum(X.weights) ≈ 1.0
    @test X.nodes * X.weights ≈ mu
    dx = X.nodes .- mu
    @test (dx .* reshape(X.weights,1,:)) * transpose(dx) ≈ Cov
end

main()
test_moments()
