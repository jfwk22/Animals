# We use nHidden as a vector, containing the number of hidden units in each layer
# Definitely not the most efficient implementation!

# Function that returns total number of parameters
function NeuralNet_nParams(d,nHidden)

	# Connections from inputs to first hidden layer
	nParams = d*nHidden[1]

	# Connections between hidden layers
	for h in 2:length(nHidden)
		nParams += nHidden[h-1]*nHidden[h]
	end

	# Connections from last hidden layer to output
	nParams += nHidden[end]

end

# Compute squared error and gradient
# for a single training example (x,y)
# (x is assumed to be a column-vector)
function NeuralNet_backprop(bigW,x,y,nHidden)

	
	d = length(x)
	nLayers = length(nHidden)
	
	#### Reshape 'bigW' into vectors/matrices
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = Array{Any}(undef,nLayers-1)
	for layer in 2:nLayers
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		ind += nHidden[layer]*nHidden[layer-1]
	end
	v = bigW[ind+1:end]

	#### Define activation function and its derivative
	#h(z) = tanh.(z)
	#dh(z) = (sech.(z)).^2

	h(z) = map((x) -> (x)/(1 + exp(x)), z)
	dh(z) = map((x) -> (x)/(1 + exp(x)) + (1 - (x)/(1 + exp(x)))/(1+exp(x)),z)
	
	#h(z) = max.(0,z)
	#dh(z) = ((max.(0,z)) .>0)


	#### Forward propagation
	z = Array{Any}(undef,nLayers)
	z[1] = W1*x
	for layer in 2:nLayers
		z[layer] = Wm[layer-1]*h(z[layer-1])
	end
	yhat = v'*h(z[end])

	r = yhat - y
	f = (1/2)r^2

	#### Backpropagation
	dr = r
	err = dr

	# Output weights
	Gout = err*h(z[end])

	Gm = Array{Any}(undef,nLayers-1)
	if nLayers > 1
		# Last Layer of Hidden Weights
		backprop = err*(dh(z[end]).*v)
		Gm[end] = backprop*h(z[end-1])'

		# Other Hidden Layers
		for layer in nLayers-2:-1:1
			backprop = (Wm[layer+1]'*backprop).*dh(z[layer+1])
			Gm[layer] = backprop*h(z[layer])'
		end

		# Input Weights
		backprop = (Wm[1]'*backprop).*dh(z[1])
		G1 = backprop*x'
	else
		# Input weights
		G1 = err*(dh(z[1]).*v)*x'
	end

	#### Put gradients into vector
	g = zeros(size(bigW))
	g[1:nHidden[1]*d] = G1
	ind = nHidden[1]*d
	for layer in 2:nLayers
		g[ind+1:ind+nHidden[layer]*nHidden[layer-1]] = Gm[layer-1]
		ind += nHidden[layer]*nHidden[layer-1]
	end
	g[ind+1:end] = Gout

	return (f,g)
end

# Computes predictions for a set of examples X
function NeuralNet_predict(bigW,Xhat,nHidden)
	
	(t,d) = size(Xhat)
	nLayers = length(nHidden)
	

	#### Reshape 'bigW' into vectors/matrices
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = Array{Any}(undef,nLayers-1)
	for layer in 2:nLayers
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		ind += nHidden[layer]*nHidden[layer-1]
	end
	v = bigW[ind+1:end]

	#### Define activation function and its derivative
	#h(z) = tanh.(z)
	#dh(z) = (sech.(z)).^2
	
	h(z) = map((x) -> (x)/(1 + exp(x)), z)
	dh(z) = map((x) -> (x)/(1 + exp(x)) + (1 - (x)/(1 + exp(x)))/(1+exp(x)),z)

	#h(z) = max.(0,z)
	#dh(z) = ((max.(0,z)) .>0)




	#### Forward propagation on each example to make predictions
	yhat = zeros(t,1)
	for i in 1:t
		# Forward propagation
		z = Array{Any}(undef,1nLayers)
		z[1] = W1*Xhat[i,:]
		for layer in 2:nLayers
			z[layer] = Wm[layer-1]*h(z[layer-1])
		end
		yhat[i] = v'*h(z[end])
	end
	return yhat
end



#### Copies of the above for multi-output versions
#### (would be refactored if not for teaching purposes)

# Function that returns total number of parameters
function NeuralNetMulti_nParams(d,k,nHidden)

	# Connections from inputs to first hidden layer
	nParams = d*nHidden[1]

	# Connections between hidden layers
	for h in 2:length(nHidden)
		nParams += nHidden[h-1]*nHidden[h]
	end

	# Connections from last hidden layer to outputs
	nParams += nHidden[end]*k

end

# Compute squared error and gradient
# for a single training example (x,y)
# (x is assumed to be a column-vector)
function NeuralNetMulti_backprop(bigW,x,y,k,nHidden)
	d = length(x)
	nLayers = length(nHidden)

	#### Reshape 'bigW' into vectors/matrices
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = Array{Any}(undef,nLayers-1)
	for layer in 2:nLayers
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		ind += nHidden[layer]*nHidden[layer-1]
	end
	v = bigW[ind+1:end]
	v = reshape(v,nHidden[end],k)

	#### Define activation function and its derivative
	h(z) = map((x) -> (x)/(1 + exp(x)), z)
	dh(z) = map((x) -> (x)/(1 + exp(x)) + (1 - (x)/(1 + exp(x)))/(1+exp(x)),z)


	#### Forward propagation
	z = Array{Any}(undef,nLayers)
	z[1] = W1*x
	for layer in 2:nLayers
		z[layer] = Wm[layer-1]*h(z[layer-1])
	end
	yhat = v'*h(z[end])

	r = yhat-y
	f = (1/2)sum(r.^2)

	#### Backpropagation
	dr = r
	err = dr

	# Output weights
	Gout = zeros(nHidden[end],k)
	for c in 1:k
		Gout[:,c] = err[c]*h(z[end])
	end

	Gm = Array{Any}(undef,nLayers-1)
	if nLayers > 1
		# Last Layer of Hidden Weights
		backprop = zeros(k,nHidden[end])
		Gm[end] = zeros(nHidden[end],nHidden[end-1])
		for c in 1:k
			backprop[c,:] = err[c]*(dh(z[end]).*v[:,c])
			Gm[end] += backprop[c,:]*h(z[end-1])'
		end
		backprop = sum(backprop,dims=1)'

		# Other Hidden Layers
		for layer in nLayers-2:-1:1
			backprop = (Wm[layer+1]'*backprop).*dh(z[layer+1])
			Gm[layer] = backprop*h(z[layer])'
		end

		# Input Weights
		backprop = (Wm[1]'*backprop).*dh(z[1])
		G1 = backprop*x'
	else
		# Input weights
		G1 = zeros(size(W1))
		for c in 1:k
			G1 += err[c]*(dh(z[1]).*v[:,c])*x'
		end
	end

	#### Put gradients into vector
	g = zeros(size(bigW))
	g[1:nHidden[1]*d] = G1
	ind = nHidden[1]*d
	for layer in 2:nLayers
		g[ind+1:ind+nHidden[layer]*nHidden[layer-1]] = Gm[layer-1]
		ind += nHidden[layer]*nHidden[layer-1]
	end
	g[ind+1:end] = Gout

	return (f,g)
end


# Computes predictions for a set of examples X
function NeuralNet_predict(bigW,Xhat,k,nHidden)
	(t,d) = size(Xhat)
	nLayers = length(nHidden)

	#### Reshape 'bigW' into vectors/matrices
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = Array{Any}(undef,nLayers-1)
	for layer in 2:nLayers
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		ind += nHidden[layer]*nHidden[layer-1]
	end
	v = bigW[ind+1:end]
	v = reshape(v,nHidden[end],k)

	#### Define activation function and its derivative
	h(z) = map((x) -> (x)/(1 + exp(x)), z)
	dh(z) = map((x) -> (x)/(1 + exp(x)) + (1 - (x)/(1 + exp(x)))/(1+exp(x)),z)

	#### Forward propagation on each example to make predictions
	yhat = zeros(t,k)
	for i in 1:t
		# Forward propagation
		z = Array{Any}(undef,1nLayers)
		z[1] = W1*Xhat[i,:]
		for layer in 2:nLayers
			z[layer] = Wm[layer-1]*h(z[layer-1])
		end
		yhat[i,:] = v'*h(z[end])
	end
	return mapslices(argmax,yhat,dims=2)
end
