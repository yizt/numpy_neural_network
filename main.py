# Author: Thang Vu
# Date: 26/Nov/2017
# Desription: Load dataset, train with dnn, store best model based on val_set
#             finally test best model with test_set

from __future__ import print_function
import numpy as np

from sgd import SGD
from dnn import DNN
from load_mnist import load_mnist_datasets
from layers import softmax_cross_entropy_loss

def check_acc(model, X, y):
    score = model.forward(X)
    preds = np.argmax(score, axis=1)
    acc = np.mean(preds == y)
    return acc

def main():
    print('=========================================')
    print('               Numpy DNN                 ')
    print('              26/Nov/2017                ')
    print('    By Thang Vu (thangvubk@gmail.com)    ')
    print('=========================================')

    # load datasets
    path = 'data/mnist.pkl.gz'
    train_set, val_set, test_set = load_mnist_datasets(path)
    batch_size = 128
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set

    # bookeeping for best model based on validation set
    best_val_acc = -1
    best_model = None

    # create model and optimization method
    dnn = DNN()
    sgd = SGD(lr=0.1, lr_decay=0.1, weight_decay=1e-3, momentum=0.9)
    
    # Train 
    batch_size = 128
    for epoch in range(20):
        dnn.train_mode() # set model to train mode (because of dropout)
        
        num_train = X_train.shape[0]
        num_batch = num_train//batch_size
        for batch in range(num_batch):
            # get batch data
            batch_mask = np.random.choice(num_train, batch_size)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]
           
            # forward
            output = dnn.forward(X_batch)
            loss, dout = softmax_cross_entropy_loss(output, y_batch)
            if batch%100 == 0:
                print("Epoch %2d Iter %3d Loss %.5f" %(epoch, batch, loss))

            # backward and update
            grads = dnn.backward(dout)
            sgd.step(dnn.params, grads)
                                
        sgd.decay_learning_rate() # decay learning rate after one epoch
        dnn.eval_mode() # set model to eval mode 
        train_acc = check_acc(dnn, X_train, y_train)
        val_acc = check_acc(dnn, X_val, y_val)

        if(best_val_acc < val_acc):
            best_val_acc = val_acc
            best_model = dnn

        # store best model based n acc_val
        print('Epoch finish. ')
        print('Train acc %.3f' %train_acc)
        print('Val acc %.3f' %val_acc)
        print('-'*30)
        print('')

    print('Train finished. Best acc %.3f' %best_val_acc)
    test_acc = check_acc(best_model, X_test, y_test)
    print('Test acc %.3f' %test_acc)

if __name__ == '__main__':
    main()
