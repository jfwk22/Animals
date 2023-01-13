using DelimitedFiles
using LinearAlgebra
using Printf

include("PCA.jl")

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

# Standardize columns
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)
Xhat = [-4 3; 0 1; -2 2; 4 -1; 2 0]
(Xhat, mu2, sigma2) = standardizeCols(Xhat)

# Plot matrix as image
using PyPlot
figure(1)
clf()
imshow(X)

# Show scatterplot of 2 random features or latent z_i
#j1 = rand(1:d)
#j2 = rand(1:d)
figure(2)

Xtest_1 = [-3 2.5]
Xtest_2 = [-3 2]

model = PCA(X, 2)
model_2 = PCA(Xhat, 1)
Z = model.compress(X)

Z_2 = model_2.compress(Xtest_1)
rerror = norm(model_2.expand(Z_2) - Xtest_1, 2)

@printf("reconstruction error 1 = %.3f\n", rerror)



Z_2 = model_2.compress(Xtest_2)
rerror = norm(model_2.expand(Z_2) - Xtest_2, 2)

@printf("reconstruction error 2 = %.3f\n", rerror)

rerror = norm(model.expand(Z) - X, 2)
rerror /= norm(X, 2)





clf()
plot(Z[:,1],Z[:,2],".")
for i in 1:n
    annotate(dataTable[i+1,1],
	xy=[Z[i,1],Z[i,2]],
	xycoords="data")
end



k=2

while rerror > 0.5

global k +=1

model = PCA(X, k)

Z = model.compress(X)
rerror = norm(model.expand(Z) - X, 2)
rerror /= norm(X, 2)

@printf("when k = %d, ratio = %.3f\n", k,rerror)
end