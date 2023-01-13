using Printf

# Load X and y variable
using JLD
using PyPlot
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])
n = size(X,1)
X = [ones(n,1) X]
d = 2

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [10]
nParams = NeuralNet_nParams(d,nHidden)
w = .00001*randn(nParams,1) 

# Train with stochastic gradient
maxIter = 30000
stepSize = 1e-4
w2 = w
for t in 1:maxIter

	# The stochastic gradient update:
	i = rand(1:n)
	(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
	w3 = w
	global w = w - stepSize*g +0.70(w3-w2)
	global w2 = w3

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		@printf("train error = %d\n",f)
		@printf("Training iteration = %d\n",t-1)

		figure(1)
		clf()
		Xhat = -10:.05:10
		yhat = NeuralNet_predict(w,[ones(length(Xhat),1) Xhat],nHidden)
		
		plot(X[:,2],y,".")
		plot(Xhat,yhat,"g-")
		sleep(.1)
	end
end
