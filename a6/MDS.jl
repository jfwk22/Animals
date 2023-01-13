include("misc.jl")
include("PCA.jl")
include("findMin.jl")
using Printf

function MDS(X)
    (n,d) = size(X)

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stress(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2

            # Gradient
            df = s
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
        end
    end
    return (f,G[:])
end

function ISOMAP(X)
	#Compute all Distances
	(n,d) = size(X)
	k=2
	D = distancesSquared(X,X)
	D = sqrt.(abs.(D))
	

	#Set distance to oneself to be infinity (so it doesn't get picked as a neighbour)
	D += diagm(fill(Inf,n))
	
	#Find neighbours
	adjm = fill(Inf, n,n)

	for i in 1:n
		near = sortperm(D[:,i])
		adjm[near[1:k+1],i] = D[near[1:k+1],i]
		near = sortperm(D[i,:])
		adjm[i,near[1:k+1]] = D[i,near[1:k+1]]

	end
	
	#Replace distances with geodesic distances - Modification for disconnected graph
	x=0

	for i in 1:n
		for j in 1:n
			z = dijkstra(adjm,i,j)
			
			
			D[i,j] = z
			if !isinf(z) & (x<z)
				x=z
			end


		end
	end



	for i in 1:n
		for j in 1:n
			if isinf(D[i,j])
				D[i,j] = x
			end
		end
	end
	
	
	
	model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)
	@printf("x =%d",x)
    Z[:] = findMin(funObj,Z[:])

    return Z




end
