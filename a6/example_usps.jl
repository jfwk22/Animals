using Printf
using JLD
using PyPlot
using Statistics
include("misc.jl")
# Load X and y variable
data = load("uspsData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

(n,d) = size(X)
t = size(Xtest,1)

# Standardize columns and add bias variable to input layer
(X,mu,sigma) = standardizeCols(X)
X = [ones(n,1) X]
d += 1

# Apply the same transformation to test data
Xtest = standardizeCols(Xtest,mu=mu,sigma=sigma)
Xtest = [ones(t,1) Xtest]

# Let 'k' be the number of classes, and 'Y' be a matrix of binary labels
k = maximum(y)
Y = zeros(n,k)
for i in 1:n
	Y[i,y[i]] = 1
end

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [10]
@show nHidden
nParams = NeuralNetMulti_nParams(d,k,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
maxIter = 100000
stepSize = 1e-3
for iter in 1:maxIter

	# The stochastic gradient update:
	i = rand(1:n)
	(f,g) = NeuralNetMulti_backprop(w,X[i,:],Y[i,:],k,nHidden)
	global w = w - stepSize*g

	# Every few iterations, plot the data/model:
	if (mod(iter,round(maxIter/50)) == 0)
		yhat = NeuralNet_predict(w,Xtest,k,nHidden)
		@printf("Training iteration = %d, test error = %f\n",iter,sum(yhat .!= ytest)/t)
	end
end
