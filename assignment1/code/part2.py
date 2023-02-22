import random
import math


def forward_prop(network, input, target):
    network["inputs"] = input
    fp = Forwardpropagate()
    network["outputs"][0] = fp.factivate_layer(network["inputs"], network["weights"][0], network["biases"][0], fp.fsigmoid)
    network["outputs"][1] = fp.factivate_layer(network["outputs"][0], network["weights"][1], network["biases"][1], fp.fsoftmax)
    network["loss"] = fp.fcross_entropy_loss(network["outputs"][-1][target])
    return network

def backward_prop(network):
    bp = Backpropagate()

    # backpropagate cross entropy loss and softmax in one step
    gly = [bp.b_cel_sm(network["outputs"][-1],[1.0, 0.0])]

    # backpropagate layer activation outputs
    glx = bp.bweights(gly,[network["outputs"][-2]])
        # bp.bweights(gly, network["outputs"][-2])  # this is for adjustment of weights
    # print(glx)
    glw = bp.bactivate(gly, network["weights"][-1])
    glb2 = bp.bbias(gly)  # bias gradient

    # backpropagate sigmoid
    glv = [bp.bsigmoid(network["outputs"][-2], glw)]

    # backpropagate first layer's weights
    glu = bp.bweights(glv, [network["inputs"]])  # this is for adjustment of 1st layer weights
    glb1 = bp.bbias(glv)  # bias gradient
    return glx, glu, glb2, glb1

def adjust_weights(learning_rate, weights, biases, gradient_weight, gradient_bias):
    for z in range(len(weights)):
        for y in range(len(weights[0])):
            weights[z][y] += -learning_rate * gradient_weight[z][y]
    for x in range(len(biases)):
        biases[x] += -learning_rate * gradient_bias[x]
    return weights, biases

def matrix_mult(matrix1, matrix2):
	result = [[0 for x in range(len(matrix2[0]))] for y in range(len(matrix1))]
	for i in range(len(matrix1)):
		for j in range(len(matrix2[0])):
			for k in range(len(matrix2)):
				result[i][j] += matrix1[i][k] * matrix2[k][j]
	return result

def matrix_add(X, Y):
	# iterate through rows
	result = [[0 for x in range(len(Y[0]))] for y in range(len(X))]
	for i in range(len(X)):
	# iterate through columns
		for j in range(len(X[0])):
			result[i][j] = X[i][j] + Y[i][j]
	return result

def transpose_vector(inputv):
	return [[x] for x in inputv]

def initialize_network_q3(inputs=None):
	if inputs is None:
		inputs = []
	network = {}
	first_layer = [[1, 1, 1], [-1, -1, -1]]
	first_outs = [0, 0, 0]
	first_bias = [0, 0, 0]
	second_layer = [[1, 1], [-1, -1], [-1, -1]]
	second_outs = [0, 0]
	second_bias = [0, 0]

	network["inputs"]  = inputs
	network["weights"] = [first_layer, second_layer]
	network["biases"]  = [first_bias, second_bias]
	network["outputs"] = [first_outs, second_outs]
	network["loss"] = 1.0
	return network

def initialize_network():
	network = {}
	first_layer = [
		[random.randrange(-1,1), random.randrange(-1,1), random.randrange(-1,1)],
		[random.randrange(-1,1), random.randrange(-1,1), random.randrange(-1,1)]
	]
	first_outs = [0, 0, 0]
	first_bias = [0, 0, 0]
	second_layer = [
		[random.randrange(-1,1), random.randrange(-1,1)],
		[random.randrange(-1,1), random.randrange(-1,1)],
		[random.randrange(-1,1), random.randrange(-1,1)]
	]
	second_outs = [0, 0]
	second_bias = [0, 0]

	network["inputs"]  = [0.0, 0.0]
	network["weights"] = [first_layer, second_layer]
	network["biases"]  = [first_bias, second_bias]
	network["outputs"] = [first_outs, second_outs]
	network["loss"] = 1.0
	return network

class Forwardpropagate:
	def fsigmoid(self, inputs):
		return [1.0/(1.0 + math.exp(-x)) for x in inputs]

	def fsoftmax(self, inputs):
		exp_inputs = [math.exp(x) for x in inputs]
		return [(math.exp(x) / math.fsum(exp_inputs)) for x in inputs]

	def fcross_entropy_loss(self, pred):
		return -math.log(pred)

	def fcross_entropy_loss_full(self, inputs, target):  # where target is a one-hot-encoded vector of the true class
		return -math.fsum([(math.log(inputs[x])) * target[x] for x in range(len(inputs))])

	def factivate_layer(self, inputs, weights, bias, function=None):
		output = matrix_mult([inputs], weights)
		for x in range(len(output)):
			output[0][x] += bias[x]
		if function:
			output = function(output[0])
		return output

class Backpropagate:
	def bcross_entropy_loss(self, inputs, target):
		output = [-(1/inputs[x]) if target[x] == 1.0 else 0.0 for x in range(len(inputs))]
		return output

	def bsigmoid(self, sig, sigder):
		return [sigder[x] * sig[x] * (1-sig[x]) for x in range(len(sig))]

	def bsoftmax(self, y):
		output = [[0.0 for k in range(len(y))] for l in range(len(y))]
		for i in range(len(y)):
			for j in range(len(y)):
				if i == j:
					output[i][j] = y[i] * (1 - y[i])
				else:
					output[i][j] = -y[j] * y[i]
		return output

	def b_cel_sm(self,pred, true):
		return [a - b for a, b in zip(pred, true)]

	def bweights(self, inputs, prev_inputs):

		output = [[0.0 for k in range(len(inputs[0]))] for l in range(len(prev_inputs[0]))]
		for i in range(len(prev_inputs[0])):
			for j in range(len(inputs[0])):
				output[i][j] = inputs[0][j] * prev_inputs[0][i]
		return output

	def bactivate(self, inputs, weights):
		vin = transpose_vector(inputs[0])
		for j in range(len(vin)):
			for i in range(len(weights)):
				vin[j].append(vin[j][0])

		vout = matrix_mult(weights,vin)
		for j in range(len(vout)):
			vout[j] = math.fsum(vout[j])
		return vout

	def bbias(self,inputs):
		return inputs
