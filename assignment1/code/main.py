from part2 import *
from part3n4 import *
import numpy as np
import matplotlib.pyplot as plt
from util import *


def run1(xtrain, ytrain, n_epochs=1, learning_rate=0.01, init=initialize_network_q3()):

    nn1 = init
    nn1["inputs"] = xtrain

    for n in range(n_epochs):

        for x in range(1000):
            nn1 = forward_prop(nn1, xtrain, ytrain)
            gl2, gl1, glb2, glb1 = backward_prop(nn1)


        nn1["weights"][-1], nn1["biases"][-1] = adjust_weights(learning_rate, nn1["weights"][-1], nn1["biases"][-1],
                                                               gl2, glb2[0])
        nn1["weights"][-2], nn1["biases"][-2] = adjust_weights(learning_rate, nn1["weights"][-2], nn1["biases"][-2],
                                                               gl1, glb1[0])
    return nn1, gl2, glb2[0], gl1, glb1[0]

# run 2 is with SGD
def run2(xtrain, ytrain, n_epochs=1, learning_rate=0.1, init=initialize_network()):

    change_loss = []
    nn1 = init

    for n in range(n_epochs):
        ind = random.choice(list(range(len(xtrain))))

        nn1 = forward_prop(nn1, xtrain[ind], ytrain[ind])
        gl2, gl1, glb2, glb1 = backward_prop(nn1)

        nn1["weights"][-1], nn1["biases"][-1] = adjust_weights(learning_rate, nn1["weights"][-1], nn1["biases"][-1],
                                                               gl2, glb2[0])
        nn1["weights"][-2], nn1["biases"][-2] = adjust_weights(learning_rate, nn1["weights"][-2], nn1["biases"][-2],
                                                               gl1, glb1[0])
        change_loss.append(nn1["loss"])
    return nn1, change_loss

# run 3 is with minibatch
def run3():
    (xtrain, ytrain), (xval, yval), num_cls = load_synth()
    n_epochs = 100
    learning_rate = 0.1
    nn1 = initialize_network()

    for n in range(n_epochs):
        av_gl2 = [[0.0 for x in range(len(nn1["weights"][-1][0]))] for y in range(len(nn1["weights"][-1]))]
        av_gl1 = [[0.0 for x in range(len(nn1["weights"][-2][0]))] for y in range(len(nn1["weights"][-2]))]
        av_glb2 = [[0.0 for y in range(len(nn1["biases"][-1]))]]
        av_glb1 = [[0.0 for y in range(len(nn1["biases"][-2]))]]

        losses = [0.0] * len(xtrain)
        rando = list(range(len(xtrain)))
        random.shuffle(rando)

        for x in range(1000):
            nn1 = forward_prop(nn1, xtrain[rando[x]].tolist(), ytrain[rando[x]].tolist())
            losses[x] = nn1["loss"]
            gl2, gl1, glb2, glb1 = backward_prop(nn1)
            av_gl2 = matrix_add(gl2, av_gl2)
            av_gl1 = matrix_add(av_gl1, gl1)
            av_glb2 = matrix_add(av_glb2, glb2)
            av_glb1 = matrix_add(av_glb1, glb1)

        av_gl2 = [[y / (x + 1) for y in z] for z in av_gl2]
        av_gl1 = [[y / (x + 1) for y in z] for z in av_gl1]
        av_glb2 = [[y / (x + 1) for y in z] for z in av_glb2]
        av_glb1 = [[y / (x + 1) for y in z] for z in av_glb1]
        av_loss = math.fsum(losses) / (x + 1)
        print(av_loss)

        nn1["weights"][-1], nn1["biases"][-1] = adjust_weights(learning_rate, nn1["weights"][-1], nn1["biases"][-1],
                                                               av_gl2, av_glb2[0])
        nn1["weights"][-2], nn1["biases"][-2] = adjust_weights(learning_rate, nn1["weights"][-2], nn1["biases"][-2],
                                                               av_gl1, av_glb1[0])


if __name__ == '__main__':

    """ QUESTION 3 """
    # network1, gl2, glb2, gl1, glb1 = run1([1,-1],0)
    # print("Layer 1 weight gradients:", gl1)
    # print("Layer 1 bias gradients  :", glb1)
    # print("Layer 1 weights         :", network1["weights"][0])
    # print("Layer 1 bias            :", network1["biases"][0], "\n")
    #
    # print("Layer 2 weight gradients:", gl2)
    # print("Layer 2 bias gradients  :", glb2)
    # print("Layer 2 weights         :", network1["weights"][1])
    # print("Layer 2 bias            :", network1["biases"][1])


    """ QUESTION 4 """
    (xtrain, ytrain), (xt,yt), f = load_synth(seed=1)
    xtr = xtrain.tolist()
    ytr = ytrain.tolist()
    n_ep = 500
    network1, d_loss = run2(xtrain, ytrain, n_epochs=n_ep, learning_rate=0.05)

    epochs = range(n_ep)
    plt.plot(epochs, d_loss, 'g', label='Training loss  -  Learning Rate = 0.05, SGD')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    """ QUESTION 5 """

    (train_x, train_y), (test_x, test_y), num_cls = load_mnist(flatten=True, final=True)
    n_runs = 6
    n_ep = 5
    lrs = [0.001, 0.01, 0.03, 0.1, 0.3]

    # do a few runs each with different values for learning rate and store outputs
    statstuff = []

    for lr in range(len(lrs)):

        tr_loss = [0]*n_runs; tr_acc = [0]*n_runs; ts_loss = [0]*n_runs; ts_acc = [0]*n_runs

        for n in range(n_runs):
            tr_loss[n], tr_acc[n], ts_loss[n], ts_acc[n] = train_network(train_x, train_y, test_x, test_y, num_cls, n_input=784,
                        n_hidden=300, n_output=10, n_epochs=n_ep, batch_size=50, learning_rate=lrs[lr], moving_average=2)

        tr_losses = np.array(tr_loss)
        tr_accs = np.array(tr_acc)
        ts_losses = np.array(ts_loss)
        ts_accs = np.array(ts_acc)

        # means & StDs
        trl_m = np.mean(tr_losses, axis=0)
        tra_m = np.mean(tr_accs, axis=0)
        tsl_m = np.mean(ts_losses, axis=0)
        tsa_m = np.mean(ts_accs, axis=0)

        trl_s = np.std(tr_losses, axis=0)
        tra_s = np.std(tr_accs, axis=0)
        tsl_s = np.std(ts_losses, axis=0)
        tsa_s = np.std(ts_accs, axis=0)

        epochs = range(n_ep)
        fig, (ax1, ax2) = plt.subplots(1,2,sharex='all',sharey='all')
        fig.suptitle('Training and Validation loss')

        ax1.errorbar(epochs, trl_m, yerr=trl_s)
        ax1.set_title('Training loss - LR='+str(lrs[lr]))
        ax2.errorbar(epochs, tsl_m, yerr=tsl_s)
        ax2.set_title('Validation loss - LR='+str(lrs[lr]))
        # plt.plot(epochs, tra_m, tra_s, 'g', label='Training accuracy')
        # plt.plot(epochs, tsa_m, tsa_s, 'b', label='validation accuracy')
        # plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    tr_loss, tr_acc, ts_loss, ts_acc = train_network(train_x, train_y, test_x, test_y, num_cls, n_input=784,
                        n_hidden=300, n_output=10, n_epochs=1000, batch_size=50, learning_rate=0.3)

    epochs = range(1000)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharex='all')
    fig.suptitle('Training and Validation loss')

    ax1.plot(epochs, tr_loss, 'b')
    ax1.set_title('Training loss - LR=0.3, Batch size=50')
    ax2.plot(epochs, ts_loss, 'g')
    ax2.set_title('Validation loss - LR=0.3, Batch size=50')
    ax3.plot(epochs, ts_acc, 'r')
    ax3.set_title('Validation accuracy - LR=0.3, Batch size=50')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
