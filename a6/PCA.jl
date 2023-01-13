using Printf
using Statistics
using LinearAlgebra
include("misc.jl")
include("findMin.jl")

function PCA(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,dims=1)
    X -= repeat(mu,n,1)

    (U,S,V) = svd(X)
    W = V[:,1:k]'

    compress(Xhat) = compressFunc(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compressFunc(Xhat,W,mu)
    (t,d) = size(Xhat)
    Xcentered = Xhat - repeat(mu,t,1)
    return Xcentered*W' # Assumes W has orthogonal rows
end

function expandFunc(Z,W,mu)
    (t,k) = size(Z)
    return Z*W + repeat(mu,t,1)
end

function PCA_gradient(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,dims=1)
    X -= repeat(mu,n,1)

    # Initialize W and Z
    W = randn(k,d)
    Z = randn(n,k)

    R = Z*W - X
    f = sum(R.^2)
    funObjZ(z) = pcaObjZ(z,X,W)
    funObjW(w) = pcaObjW(w,X,Z)
    for iter in 1:50
        fOld = f

        # Update Z
        Z[:] = findMin(funObjZ,Z[:],verbose=false,maxIter=10)

        # Update W
        W[:] = findMin(funObjW,W[:],verbose=false,maxIter=10)

        R = Z*W - X
        f = sum(R.^2)
        @printf("Iteration %d, loss = %f\n",iter,f/length(X))

        if (fOld - f)/length(X) < 1e-2
            break
        end
    end


    # We didn't enforce that W was orthogonal so we need to optimize to find Z
    compress(Xhat) = compress_gradientDescent(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compress_gradientDescent(Xhat,W,mu)
    (t,d) = size(Xhat)
    k = size(W,1)
    Xcentered = Xhat - repeat(mu,t,1)
    Z = zeros(t,k)

    funObj(z) = pcaObjZ(z,Xcentered,W)
    Z[:] = findMin(funObj,Z[:],verbose=false)
    return Z
end


function pcaObjZ(z,X,W)
    # Rezie vector of parameters into matrix
    n = size(X,1)
    k = size(W,1)
    Z = reshape(z,n,k)

    # Comptue function value
    R = Z*W - X
    f = (1/2)sum(R.^2)

    # Comptue derivative with respect to each residual
    dR = R

    # Multiply by W' to get elements of gradient
    G = dR*W'

    # Return function and gradient vector
    return (f,G[:])
end

function pcaObjW(w,X,Z)
    # Rezie vector of parameters into matrix
    d = size(X,2)
    k = size(Z,2)
    W = reshape(w,k,d)

    # Comptue function value
    R = Z*W - X
    f = (1/2)sum(R.^2)

    # Comptue derivative with respect to each residual
    dR = R

    # Multiply by Z' to get elements of gradient
    G = Z'dR

    # Return function and gradient vector
    return (f,G[:])
end







function RPCA(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,dims=1)
    X -= repeat(mu,n,1)

    # Initialize W and Z
    W = randn(k,d)
    Z = randn(n,k)

    R = Z*W - X
    f = sum(map(huberL,R))
    funObjZR(z) = pcaObjZR(z,X,W)
    funObjWR(w) = pcaObjWR(w,X,Z)
    for iter in 1:50
        fOld = f

        # Update Z
        Z[:] = findMin(funObjZR,Z[:],verbose=false,maxIter=10)

        # Update W
        W[:] = findMin(funObjWR,W[:],verbose=false,maxIter=10)

        R = Z*W - X
        f = sum(map(huberL, R))
        @printf("Iteration %d, loss = %f\n",iter,f/length(X))

        if (fOld - f)/length(X) < 1e-2
            break
        end
    end


    # We didn't enforce that W was orthogonal so we need to optimize to find Z
    compress(Xhat) = compress_gradientDescentR(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compress_gradientDescentR(Xhat,W,mu)
    (t,d) = size(Xhat)
    k = size(W,1)
    Xcentered = Xhat - repeat(mu,t,1)
    Z = zeros(t,k)

    funObj(z) = pcaObjZR(z,Xcentered,W)
    Z[:] = findMin(funObj,Z[:],verbose=false)
    return Z
end


function pcaObjZR(z,X,W)
    # Rezie vector of parameters into matrix
    n = size(X,1)
    k = size(W,1)
    Z = reshape(z,n,k)

    # Comptue function value
    R = Z*W - X
    f = sum(map(huberL,R))

    # Comptue derivative with respect to each residual
    dR = map(huberLD,R)

    # Multiply by W' to get elements of gradient
    G = dR*W'

    # Return function and gradient vector
    return (f,G[:])
end

function pcaObjWR(w,X,Z)
    # Rezie vector of parameters into matrix
    d = size(X,2)
    k = size(Z,2)
    W = reshape(w,k,d)

    # Comptue function value
    R = Z*W - X
    f = sum(map(huberL,R))

    # Comptue derivative with respect to each residual
    dR = map(huberLD,R)

    # Multiply by Z' to get elements of gradient
    G = Z'dR

    # Return function and gradient vector
    return (f,G[:])
end

function huberL(x)
	if abs(x) <= 0.01
		return (1/2)*x^2
	else
		return 0.01*abs(x)-0.00005
	end


end

function huberLD(x)
	if abs(x) > 0.01
		return 0.01*sign(x)

	end
	return x
end

function derivativeCheck(funObj, w)
# Evalluate the intial objective and gradient
	(f,g) = funObj(w)

	
	
		g2 = numGrad(funObj,w)

		if maximum(abs.(g-g2)) > 1e-4
			@show([g g2])
			@printf("User and numerical derivatives differ\n")
			sleep(1)
		else
			@printf("User and numerical derivatives agree\n")
		end
	


end