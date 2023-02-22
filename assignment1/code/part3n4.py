import random

import numpy as np

def one_hot(num_classes:int, target_class:int):
    target_vector = [0] * num_classes
    target_vector[target_class] = 1.0
    return np.array(target_vector)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def stable_sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def softmax(inputs):
    exp_inputs = [np.exp(x) for x in inputs]
    return [(np.exp(x) / np.sum(exp_inputs)) for x in inputs]

def stable_softmax(inputs):
    x = inputs - max(inputs)
    numerator = np.exp(x)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def train_network(input_data, input_targets, test_data, test_target, n_classes, n_input=784, n_hidden=300, n_output=10,
                  n_epochs=1000, batch_size=1, learning_rate=0.2, moving_average=10):
    # initialize
    layer1_weights = np.random.normal(loc=0, size=(n_input, n_hidden),)
    layer2_weights = np.random.normal(loc=0, size=(n_hidden, n_output))
    layer1_bias = np.zeros(shape=n_hidden)
    layer2_bias = np.zeros(shape=n_output)

    batch = list(range(len(input_data)))
    av_loss = []
    accuracy = []
    av_loss2 = []
    accuracy2 = []

    for i in range(n_epochs):
        minibatch1 = random.sample(batch,batch_size)

        sum_batch_loss = 0.0
        acc = 0.0
        sum_batch_loss2 = 0.0
        acc2 = 0.0

        wsum_weight_gradients1 = np.zeros(shape=(n_input, n_hidden))
        wsum_weight_gradients2 = np.zeros(shape=(n_hidden, n_output))
        wsum_bias_gradients1 = np.zeros(shape=n_hidden)
        wsum_bias_gradients2 = np.zeros(shape=n_output)

        for x in minibatch1:
            # connect layers --> feedforward
            layer1_in = np.matmul(input_data[x], layer1_weights) + layer1_bias
            layer1_out = stable_sigmoid(layer1_in)

            layer2_in = np.matmul(layer1_out, layer2_weights) + layer2_bias
            layer2_out = softmax(layer2_in)  # This is the prediction
            layer2_out = np.clip(layer2_out, 0.0001, 0.9999)  # this is to make the algorithm more numerically stable

            cross_entropy_loss = (1/n_classes) * -np.log(layer2_out[input_targets[x]])

            # one-hot-encoding of target, we use this for calculating accuracy, but also for calculating gradient wrt loss and softmax
            target_v = one_hot(n_classes, input_targets[x])
            if np.argmax(layer2_out) == np.argmax(target_v):
                acc += 1.0

            sum_batch_loss += cross_entropy_loss

            # GRADIENT CALCULATIONS (backpropagation of error)
            # first step is gradient descent of Error wrt both the cross entropy loss and the softmax activation
            # this is simplified to the target_vector minus the prediction_vector (i.e. the output layer outputs)
            dL_dSM = layer2_out - target_v
            dL_db2 = dL_dSM.copy()  # gradient bias

            # next is the gradient wrt the weights, i.e. nabla-y dot h-transposed
            dL_dW2 = np.outer(dL_dSM, layer1_out)
            dL_dh2 = (dL_dSM[None,:] * layer2_weights).sum(axis=1)

            # next is gradient of Sigmoid (which is also equal to the gradient wrt bias)
            dL_dSG = dL_dh2 * layer1_out * (1 - layer1_out)
            dL_db1 = dL_dSG.copy()

            # now the gradient wrt first layer weights
            dL_dW1 = np.outer(dL_dSG, input_data[x])

            wsum_weight_gradients2 = (wsum_weight_gradients2 + dL_dW2.T)
            wsum_weight_gradients1 = (wsum_weight_gradients1 + dL_dW1.T)
            wsum_bias_gradients2 = (wsum_bias_gradients2 + dL_db2.T)
            wsum_bias_gradients1 = (wsum_bias_gradients1 + dL_db1.T)

        # validation
        for x in range(len(test_target)):
            # connect layers --> feedforward
            layer1_in = np.matmul(test_data[x], layer1_weights) + layer1_bias
            layer1_out = stable_sigmoid(layer1_in)

            layer2_in = np.matmul(layer1_out, layer2_weights) + layer2_bias
            layer2_out = softmax(layer2_in)  # This is the prediction
            layer2_out = np.clip(layer2_out, 0.0001, 0.9999)  # this is to make the algorithm more numerically stable

            cross_entropy_loss = (1 / n_classes) * -np.log(layer2_out[test_target[x]])

            target_v = one_hot(n_classes, test_target[x])

            if np.argmax(layer2_out) == np.argmax(target_v):
                acc2 += 1.0

            sum_batch_loss2 += cross_entropy_loss

        av_loss.append(sum_batch_loss/batch_size)
        accuracy.append(acc/batch_size)

        av_loss2.append(sum_batch_loss2/len(test_target))
        accuracy2.append(acc2/len(test_target))

        mv_Loss = sum(av_loss[-moving_average:])/moving_average

        print("[INFO]:   Epoch ", i)
        print("Training Loss: ", av_loss[-1], "   -   Training Accuracy:", accuracy[-1])
        print("Validation Loss: ", av_loss2[-1], "   -   Validation Accuracy:", accuracy2[-1])
        print("[INFO]:   ...updating weights...\n")

        layer2_weights = layer2_weights - (learning_rate * (wsum_weight_gradients2/batch_size))
        layer1_weights = layer1_weights - (learning_rate * (wsum_weight_gradients1/batch_size))
        layer2_bias = layer2_bias - (learning_rate * (wsum_bias_gradients2/batch_size))
        layer1_bias = layer1_bias - (learning_rate * (wsum_bias_gradients1/batch_size))

    return av_loss, accuracy, av_loss2, accuracy2
