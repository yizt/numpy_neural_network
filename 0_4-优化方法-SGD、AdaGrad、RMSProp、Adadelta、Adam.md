[TOC]

## SGD

​         随机梯度下降，注意这里的随机梯度下降是mini-batch gradient descent。一般深度学习中就用sgd代表。

a)  权重参数$w$ , 权重梯度$\nabla_w$ 

b)  学习率 $\eta$ , 学习率衰减$decay$ (一般设置很小)

c)  动量大小$\gamma$ (一般设置为0.9) , t次迭代时累积的动量为$v_t$

​           则学习率的更新公式为:
$$
\eta_t = \eta /(1+t \cdot decay)  \tag 1
$$
​           累积动量和权重的更新公式如下：
$$
\begin{align}
&v_t=\gamma \cdot v_{t-1} + \eta_t \cdot \nabla_w   & 其中v_0=0    \tag 2 \\
&w = w - v_t  \tag 3
\end{align}
$$


​          

## AdaGrad

​         这一节我们介绍 Adagrad 算法，它根据自变量在每个维度的梯度值的大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题。

​        Adagrad 的算法会使用一个小批量随机梯度按元素平方的累加变量每次迭代中，首先将梯度$\nabla_w$  按元素平方后累加到变量 $s_t$
$$
s_t = s_{t-1} + \nabla_w^2  \tag4
$$
​        梯度的更新公式为:
$$
w = w - \frac {\eta_t} {\sqrt{s_t + \epsilon }} \cdot \nabla_w  \tag 5
$$
​        $\epsilon$ 是为了维持数值稳定性(避免除零)而添加的常数，例如 $10^{-6}$ ; $\eta_t$ 可以是常数，也可以像公式(1)样为衰减学习率。

​        需要强调的是，梯度按元素平方的累加变量 $s$ 出现在学习率的分母项中。因此，如果目标函数有关自变量中某个元素的偏导数一直都较大，那么就让该元素的学习率下降快一点；反之，如果目标函数有关自变量中某个元素的偏导数一直都较小，那么就让该元素的学习率下降慢一点。然而，由于 $s$ 一直在累加按元素平方的梯度，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。所以，当学习率在迭代早期降得较快且当前解依然不佳时，Adagrad 在迭代后期由于学习率过小，可能较难找到一个有用的解。

### 特点

- Adagrad 在迭代过程中不断调整学习率，并让目标函数自变量中每个元素都分别拥有自己的学习率。
- 使用 Adagrad 时，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。



## RMSProp

​        当学习率在迭代早期降得较快且当前解依然不佳时，Adagrad 在迭代后期由于学习率过小，可能较难找到一个有用的解。为了应对这一问题，RMSProp 算法对 Adagrad 做了一点小小的修改。

​        不同于 Adagrad 里状态变量$s$是到目前时间步里所有梯度按元素平方和，RMSProp 将过去时间步里梯度按元素平方做指数加权移动平均。公式如下：
$$
s_t =\gamma \cdot s_{t-1} + (1-\gamma) \cdot \nabla_w^2  \ ;  \ \ \ \ 其中   0<\gamma <1\  \tag6
$$
​        权重更新公式仍然如AdaGrad
$$
w = w - \frac {\eta_t} {\sqrt{s_t + \epsilon }} \cdot \nabla_w  \tag 5
$$

​                

​                                                                              未完待续... ...

## 参考

a) [An overview of gradient descent optimization](http://ruder.io/optimizing-gradient-descent/)

b) [优化算法](http://zh.gluon.ai/chapter_optimization/optimization-intro.html)



