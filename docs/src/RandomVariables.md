# AbstractRandomVariables interface

The abstract random variables interface provides a method for calcaulting the expected value of random varaibles in the Bellman equations. The idea behind this interface is that the values the random variable take on is given in a matrix with d by N entries called `nodes` where d is the dimension of the random variable and N is the number of points to evaluate. The probability mass at each point is given in a vector called `weights`. The expected value of a function `f` of a random variable `X` can then be calcualted by evaluating the function at each of the columsn in nodes and multiplying these values by the weights.
```julia 
EfX = sum(X.weights.*mapslices(f,X,dims=1))
```
If `X` is a discrete random variable then `X.weights` is the probability mass function. This framework can also be used for continuous random variables by replacing the nodes and weights with the nodes and weights of a quadrature scheme like guass-hermite weights and nodes for a normal distribution. The frameowrk can also be used for montecarlo integration by sampling the nodes at random and setting the weights equal to `1/N`.

The AbstractRandomVariables interface provides methods to construce normally distributed random variables using gauss-hermite qudrature (`GaussHermiteRandomVariable`), set up monte carlo integration using `MCRandomVariable`, build markov cains with `MarkovChain` and `sample_markov_chain`, and buidl custome discrete random variables and quadratures using `RandomVariable`. Details on each method are provided in the API. 