# -*- coding: utf-8 -*-
"""
Created on 2018/9/01 15:03

@author: mick.yi

测试案例
"""
import numpy as np
from dnn import Mnist, LinearRegression

from load_mnist import load_mnist_datasets
import utils


def dnn_mnist():
    # load datasets
    path = 'mnist.pkl.gz'
    train_set, val_set, test_set = load_mnist_datasets(path)
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set

    # 转为稀疏分类
    y_train, y_val,y_test =utils.to_categorical(y_train,10),utils.to_categorical(y_val,10),utils.to_categorical(y_test,10)

    # bookeeping for best model based on validation set
    best_val_acc = -1
    mnist = Mnist()

    # Train
    batch_size = 32
    lr = 1e-1
    for epoch in range(10):
        num_train = X_train.shape[0]
        num_batch = num_train // batch_size
        for batch in range(num_batch):
            # get batch data
            batch_mask = np.random.choice(num_train, batch_size)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]
            # 前向及反向
            mnist.forward(X_batch)
            loss = mnist.backward(X_batch, y_batch)
            if batch % 200 == 0:
                print("Epoch %2d Iter %3d Loss %.5f" % (epoch, batch, loss))

            # 更新梯度
            for w in ["W1", "b1", "W2", "b2", "W3", "b3"]:
                mnist.weights[w] -= lr * mnist.gradients[w]

        train_acc = mnist.get_accuracy(X_train, y_train)
        val_acc = mnist.get_accuracy(X_val, y_val)

        if(best_val_acc < val_acc):
            best_val_acc = val_acc

        # store best model based n acc_val
        print('Epoch finish. ')
        print('Train acc %.3f' % train_acc)
        print('Val acc %.3f' % val_acc)
        print('-' * 30)
        print('')

    print('Train finished. Best acc %.3f' % best_val_acc)
    test_acc = mnist.get_accuracy(X_test, y_test)
    print('Test acc %.3f' % test_acc)


if __name__ == '__main__':
    #dnn_mnist()
    m = LinearRegression()
    m.train()
