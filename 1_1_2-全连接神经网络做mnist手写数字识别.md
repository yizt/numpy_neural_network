[TOC]

## 一、定义前向、后向传播

本文将用numpy实现dnn, 并测试mnist手写数字识别

如果对神经网络的反向传播过程还有不清楚的，可以[0_1-全连接层、损失函数的反向传播](0_1-全连接层、损失函数的反向传播.md)

网络结构如下,包括3个fc层：
input(28\*28)=> fc (256) => relu => fc(256) => relu => fc(10)


```python
import numpy as np
# 定义权重、神经元、梯度
weights={}
weights_scale=1e-3
weights["W1"]=weights_scale*np.random.randn(28*28,256)
weights["b1"]=np.zeros(256)
weights["W2"]=weights_scale*np.random.randn(256,256)
weights["b2"]=np.zeros(256)
weights["W3"]=weights_scale*np.random.randn(256,10)
weights["b3"]=np.zeros(10)

nuerons={}
gradients={}

```


```python
from nn.layers import fc_forward
from nn.activations import relu_forward

# 定义前向过程
def forward(X):
    nuerons["z2"]=fc_forward(X,weights["W1"],weights["b1"])
    nuerons["z2_relu"]=relu_forward(nuerons["z2"])
    nuerons["z3"]=fc_forward(nuerons["z2_relu"],weights["W2"],weights["b2"])
    nuerons["z3_relu"]=relu_forward(nuerons["z3"])
    nuerons["y"]=fc_forward(nuerons["z3_relu"],weights["W3"],weights["b3"])
    return nuerons["y"]
```


```python
from nn.losses import cross_entropy_loss
from nn.layers import fc_backward
from nn.activations import relu_backward

# 定义后向过程
def backward(X,y_true):
    loss,dy=cross_entropy_loss(nuerons["y"],y_true)
    gradients["W3"],gradients["b3"],gradients["z3_relu"]=fc_backward(dy,weights["W3"],nuerons["z3_relu"])
    gradients["z3"]=relu_backward(gradients["z3_relu"],nuerons["z3"])
    gradients["W2"],gradients["b2"],gradients["z2_relu"]=fc_backward(gradients["z3"],
                                                                     weights["W2"],nuerons["z2_relu"])
    gradients["z2"]=relu_backward(gradients["z2_relu"],nuerons["z2"])
    gradients["W1"],gradients["b1"],_=fc_backward(gradients["z2"],
                                                    weights["W1"],X)
    return loss
```


```python
# 获取精度
def get_accuracy(X,y_true):
    y_predict=forward(X)
    return np.mean(np.equal(np.argmax(y_predict,axis=-1),
                            np.argmax(y_true,axis=-1)))
```

## 二、加载数据

mnist.pkl.gz数据源： http://deeplearning.net/data/mnist/mnist.pkl.gz   


```python
from nn.load_mnist import load_mnist_datasets
from nn.utils import to_categorical
train_set, val_set, test_set = load_mnist_datasets('mnist.pkl.gz')
train_y,val_y,test_y=to_categorical(train_set[1]),to_categorical(val_set[1]),to_categorical(test_set[1])
```


```python
# 随机选择训练样本
train_num = train_set[0].shape[0]
def next_batch(batch_size):
    idx=np.random.choice(train_num,batch_size)
    return train_set[0][idx],train_y[idx]

x,y= next_batch(16)
print("x.shape:{},y.shape:{}".format(x.shape,y.shape))
```

    x.shape:(16, 784),y.shape:(16, 10)
​    


```python
# 可视化
import matplotlib.pyplot as plt
digit=train_set[0][3]
plt.imshow(np.reshape(digit,(28,28)))
plt.show()
```


![png](pic/dnn_mnist_1.png)


## 三、训练


```python
# 初始化变量
batch_size=32
epoch = 3
steps = train_num // batch_size
lr = 0.1

for e in range(epoch):
    for s in range(steps):
        X,y=next_batch(batch_size)
        
        # 前向过程
        forward(X)
        loss=backward(X,y)
        
        # 更新梯度
        for k in ["W1","b1","W2","b2","W3","b3"]:
            weights[k]-=lr*gradients[k]
        
        if s % 500 ==0:
            print("\n epoch:{} step:{} ; loss:{}".format(e,s,loss))
            print(" train_acc:{};  val_acc:{}".format(get_accuracy(X,y),get_accuracy(val_set[0],val_y)))

            
print("\n final result test_acc:{};  val_acc:{}".
      format(get_accuracy(test_set[0],test_y),get_accuracy(val_set[0],val_y)))
```


     epoch:0 step:0 ; loss:2.302584820875885
     train_acc:0.1875;  val_acc:0.103
    
     epoch:0 step:200 ; loss:2.3089974735813046
     train_acc:0.0625;  val_acc:0.1064
    
     epoch:0 step:400 ; loss:2.3190137162037106
     train_acc:0.0625;  val_acc:0.1064
    
     epoch:0 step:600 ; loss:2.29290016314387
     train_acc:0.1875;  val_acc:0.1064
    
     epoch:0 step:800 ; loss:2.2990879829286004
     train_acc:0.125;  val_acc:0.1064
    
     epoch:0 step:1000 ; loss:2.2969247354797817
     train_acc:0.125;  val_acc:0.1064
    
     epoch:0 step:1200 ; loss:2.307249383676819
     train_acc:0.09375;  val_acc:0.1064
    
     epoch:0 step:1400 ; loss:2.3215380862102757
     train_acc:0.03125;  val_acc:0.1064
    
     epoch:1 step:0 ; loss:2.2884130059797547
     train_acc:0.25;  val_acc:0.1064
    
     epoch:1 step:200 ; loss:1.76023258152068
     train_acc:0.34375;  val_acc:0.2517
    
     epoch:1 step:400 ; loss:1.4113708080481038
     train_acc:0.40625;  val_acc:0.3138
    
     epoch:1 step:600 ; loss:1.4484238805860425
     train_acc:0.53125;  val_acc:0.5509
    
     epoch:1 step:800 ; loss:0.4831932927037818
     train_acc:0.9375;  val_acc:0.7444
    
     epoch:1 step:1000 ; loss:0.521746944367524
     train_acc:0.84375;  val_acc:0.8234
    
     epoch:1 step:1200 ; loss:0.5975823718636631
     train_acc:0.875;  val_acc:0.8751
    
     epoch:1 step:1400 ; loss:0.39426304417143254
     train_acc:0.9375;  val_acc:0.8939
    
     epoch:2 step:0 ; loss:0.3392397455325375
     train_acc:0.9375;  val_acc:0.8874
    
     epoch:2 step:200 ; loss:0.2349061434167009
     train_acc:0.96875;  val_acc:0.9244
    
     epoch:2 step:400 ; loss:0.1642980488678663
     train_acc:0.96875;  val_acc:0.9223
    
     epoch:2 step:600 ; loss:0.18962678031295344
     train_acc:1.0;  val_acc:0.9349
    
     epoch:2 step:800 ; loss:0.1374088809322303
     train_acc:1.0;  val_acc:0.9365
    
     epoch:2 step:1000 ; loss:0.45885105735878895
     train_acc:0.96875;  val_acc:0.939
    
     epoch:2 step:1200 ; loss:0.049076886226820146
     train_acc:1.0;  val_acc:0.9471
    
     epoch:2 step:1400 ; loss:0.3464252344080918
     train_acc:0.9375;  val_acc:0.9413
    
     epoch:3 step:0 ; loss:0.2719433362166901
     train_acc:0.96875;  val_acc:0.9517
    
     epoch:3 step:200 ; loss:0.06844332074679768
     train_acc:1.0;  val_acc:0.9586
    
     epoch:3 step:400 ; loss:0.16346902137921188
     train_acc:1.0;  val_acc:0.9529
    
     epoch:3 step:600 ; loss:0.15661875582989374
     train_acc:1.0;  val_acc:0.9555
    
     epoch:3 step:800 ; loss:0.10004190054365474
     train_acc:1.0;  val_acc:0.9579
    
     epoch:3 step:1000 ; loss:0.20624793312023684
     train_acc:0.96875;  val_acc:0.9581
    
     epoch:3 step:1200 ; loss:0.016292493383161803
     train_acc:1.0;  val_acc:0.9602
    
     epoch:3 step:1400 ; loss:0.08761421046492293
     train_acc:1.0;  val_acc:0.9602
    
     epoch:4 step:0 ; loss:0.23058956036352923
     train_acc:0.9375;  val_acc:0.9547
    
     epoch:4 step:200 ; loss:0.14973880899309255
     train_acc:0.96875;  val_acc:0.9674
    
     epoch:4 step:400 ; loss:0.4563995699690676
     train_acc:0.9375;  val_acc:0.9667
    
     epoch:4 step:600 ; loss:0.03818259411193518
     train_acc:1.0;  val_acc:0.9641
    
     epoch:4 step:800 ; loss:0.18057951765239755
     train_acc:1.0;  val_acc:0.968
    
     epoch:4 step:1000 ; loss:0.05313018618481231
     train_acc:1.0;  val_acc:0.9656
    
     epoch:4 step:1200 ; loss:0.07373341371929959
     train_acc:1.0;  val_acc:0.9692
    
     epoch:4 step:1400 ; loss:0.0499225679993673
     train_acc:1.0;  val_acc:0.9696
    
     final result test_acc:0.9674;  val_acc:0.9676



```python
# 查看预测结果
x,y=test_set[0][5],test_y[5]
plt.imshow(np.reshape(x,(28,28)))
plt.show()

y_predict = np.argmax(forward([x])[0])

print("y_true:{},y_predict:{}".format(np.argmax(y),y_predict))


```


![png](pic/dnn_mnist_2.png)


    y_true:1,y_predict:1
​    
